"""Official split loading. Training does not access test arrays."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
import tarfile
import zipfile
import shutil
import os
import warnings
import re
from filelock import FileLock
import numpy as np
from sklearn.preprocessing import QuantileTransformer, StandardScaler


def hash_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):h.update(chunk)
    return h.hexdigest()


def read_config(repo, name):
    path=Path(repo)/'configs'/name/'dataset.json'
    if not path.is_file():raise FileNotFoundError(f'Missing required dataset config: {path}')
    cfg=json.loads(path.read_text())
    if cfg.get('dataset')!=name or cfg.get('schema')!='dpn_dataset_v2':
        raise ValueError('Wrong dataset name or unsupported config schema')
    for key in ('expected','source','preprocessing','pruning','reference_mlp','training','regularization'):
        if key not in cfg:raise ValueError(f'Missing required dataset field: {key}')
    ex=cfg['expected']
    if any(int(ex[k])<1 for k in ('input_features','classes','output_logits')):raise ValueError('Invalid output/input metadata')
    if ex['output_logits'] != (1 if ex['classes']==2 else ex['classes']):
        raise ValueError('Unexpected output-logit count; specify binary or multiclass')
    if not cfg['reference_mlp'].get('hidden_widths'):
        raise ValueError('Dataset config must state reference MLP shape; no model architecture guesses')
    acquisition = cfg.get('data_acquisition', cfg['source'])
    if acquisition.get('type') == 'official_rtdl_archive':
        _official_pins(acquisition)
    return cfg


def data_folder(repo,cfg):
    source=cfg['source']
    if source.get('type') not in ('prepared_numpy','prepared_numpy_zip','prepared_numpy_tar','official_rtdl_archive'):
        raise ValueError('Unsupported dataset source type; no implicit data acquisition')
    path=Path(source['path'])
    if not path.is_absolute():path=Path(repo)/path
    return path


# These six filenames are the official train/validation/test partitions.
# Never infer a new random split for a published benchmark dataset.
_OFFICIAL_FILES = tuple(
    f'{kind}_{split}.npy'
    for split in ('train', 'val', 'test')
    for kind in ('N', 'y')
)


_OFFICIAL_REQUIRED = frozenset(_OFFICIAL_FILES) | {'info.json'}


def _official_pins(source):
    def valid(value):
        return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None
    pins = source.get('file_sha256')
    if not valid(source.get('sha256')):
        raise ValueError('Official archive requires a pinned SHA256')
    if not isinstance(pins, dict) or set(pins) != _OFFICIAL_REQUIRED or not all(valid(v) for v in pins.values()):
        raise ValueError('Official dataset requires seven pinned file SHA256 values')
    return pins


def _verify_official_files(folder, source):
    """Trust config pins, never a mutable local receipt. Hash raw bytes only."""
    pins = _official_pins(source)
    observed = {}
    for name, expected in sorted(pins.items()):
        path = folder / name
        if not path.is_file():
            raise FileNotFoundError(f'Missing pinned official file: {path}')
        digest = hash_file(path)
        if digest != expected:
            raise ValueError(f'Official file SHA256 mismatch: {path}. Move the dataset directory aside before reacquiring.')
        observed[name] = digest
    receipt = folder / '.source_provenance.json'
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous.get('archive_sha256') != source['sha256'] or previous.get('files') != observed:
            raise ValueError(f'Official dataset provenance mismatch: {receipt}')
    return observed


def _official_dataset_ready(folder):
    return all((folder / name).is_file() for name in _OFFICIAL_REQUIRED)


def _download_official_archive(repo, source):
    """Cache the official shared archive safely across workers and datasets.

    Enforce the project-verified archive pin before accepting a download.
    The pin is not a publisher-issued checksum.
    """
    _official_pins(source)
    expected = source['sha256']
    url = source.get('download_url')
    if not url or not str(url).startswith(('https://', 'file://')):
        raise ValueError('official_rtdl_archive needs an explicit HTTPS URL')
    cache = Path(repo) / 'data' / '.archives'
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / 'rtdl_revisiting_models_data.tar.gz'
    with FileLock(str(cache / 'rtdl_download.lock'), timeout=7200):
        if not archive.exists():
            temporary = cache / f'.rtdl_download_{os.getpid()}.tmp'
            try:
                with urlopen(url, timeout=120) as stream, temporary.open('wb') as target:
                    shutil.copyfileobj(stream, target, length=1024*1024)
                # Catch expired provider links or HTML error pages.
                with temporary.open('rb') as f:
                    if f.read(2) != b'\x1f\x8b':
                        raise ValueError('Official URL did not return a gzip archive')
                if hash_file(temporary) != expected:
                    raise ValueError('Official archive SHA256 differs from dataset config')
                temporary.replace(archive)
            finally:
                temporary.unlink(missing_ok=True)
        observed = hash_file(archive)
        if observed != expected:
            raise ValueError('Official archive SHA256 differs from dataset config')
        receipt = cache / 'rtdl_archive_manifest.json'
        if receipt.is_file():
            previous = json.loads(receipt.read_text())
            if previous.get('sha256') != observed:
                raise ValueError('Cached official archive differs from earlier run')
        else:
            temp_receipt = cache / f'.rtdl_manifest_{os.getpid()}.tmp'
            temp_receipt.write_text(json.dumps({
                'url':url, 'sha256':observed,
                'verified_against_config_hash': True,
            }, indent=2, sort_keys=True))
            temp_receipt.replace(receipt)
    return archive, observed


def _extract_official_dataset(repo, cfg):
    """Extract only the configured dataset; preserve published split indices."""
    source = cfg.get('data_acquisition', cfg['source'])
    folder = data_folder(repo, cfg)
    _official_pins(source)
    # Manually supplied or cached files must match the same trusted pins.
    if _official_dataset_ready(folder):
        _verify_official_files(folder, source)
        return folder
    prefix = source.get('archive_subdir')
    if not isinstance(prefix, str) or prefix.strip('/') != f'data/{cfg["dataset"]}':
        raise ValueError('Archive subdirectory must match data/<dataset> exactly')
    lockdir = Path(repo) / 'data' / '.locks'
    lockdir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lockdir / (cfg['dataset'] + '.lock')), timeout=7200):
        if _official_dataset_ready(folder):
            _verify_official_files(folder, source)
            return folder
        # Do not mix old and newly downloaded split files.
        if folder.exists() and any(folder.iterdir()):
            raise RuntimeError(
                f'Incomplete dataset directory: {folder}. Move it aside before acquiring.')
        archive, archive_hash = _download_official_archive(repo, source)
        temporary = folder.parent / f'.{folder.name}.extract_{os.getpid()}'
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        wanted = set(_OFFICIAL_FILES) | {'info.json'}
        found = set()
        try:
            # Streaming mode handles archives larger than memory.
            with tarfile.open(archive, mode='r|gz') as container:
                for member in container:
                    path = Path(member.name)
                    if path.is_absolute() or '..' in path.parts:
                        raise ValueError('Unsafe path in official dataset archive')
                    parts = tuple(x for x in path.parts if x != '.')
                    if len(parts) != 3 or parts[:2] != ('data', cfg['dataset']):
                        continue
                    name = parts[2]
                    if name not in wanted:
                        continue
                    if not member.isfile() or name in found:
                        raise ValueError(f'Unsafe or duplicated official dataset member: {name}')
                    stream = container.extractfile(member)
                    if stream is None:
                        raise ValueError(f'Unreadable official dataset member: {name}')
                    with stream, (temporary / name).open('wb') as dst:
                        shutil.copyfileobj(stream, dst, length=1024*1024)
                    found.add(name)
            missing = _OFFICIAL_REQUIRED - found
            if missing:
                raise ValueError(f'Official dataset archive is missing {sorted(missing)}')
            verified_files = _verify_official_files(temporary, source)
            # Check all headers and dimensions before installing the dataset.
            expected = cfg['expected']
            split_rows = 0
            for split in ('train', 'val', 'test'):
                features = np.load(temporary/f'N_{split}.npy',
                                   mmap_mode='r', allow_pickle=False)
                labels = np.load(temporary/f'y_{split}.npy',
                                 mmap_mode='r', allow_pickle=False)
                if features.ndim != 2 or features.shape[1] != int(expected['input_features']):
                    raise ValueError(f'Wrong {split} feature dimensions')
                if labels.shape != (len(features),):
                    raise ValueError(f'Wrong {split} label dimensions')
                split_rows += len(features)
                del features, labels
            if split_rows != int(expected['total_rows']):
                raise ValueError('Official dataset row count differs from config')
            provenance = {
                'dataset': cfg['dataset'],
                'source_url': source['download_url'],
                'archive_sha256': archive_hash,
                'archive_subdir': prefix,
                'split_policy': 'publisher train/val/test splits, no resampling',
                'files': verified_files,
            }
            (temporary/'.source_provenance.json').write_text(
                json.dumps(provenance, indent=2, sort_keys=True))
            if folder.exists():
                folder.rmdir()  # Only empty directories permitted here.
            temporary.rename(folder)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return folder


def acquire_explicit_archive(repo,cfg):
    """Obtain prepared split data from an explicit config source."""
    source=cfg['source'];folder=data_folder(repo,cfg)
    if cfg.get('data_acquisition', {}).get('type')=='official_rtdl_archive':
        return _extract_official_dataset(repo,cfg)
    if source['type']=='official_rtdl_archive':
        return _extract_official_dataset(repo,cfg)
    if source['type']=='prepared_numpy':return folder
    if (folder/'N_train.npy').is_file():return folder
    url=source.get('download_url');sha=source.get('sha256')
    if not url or not sha or len(sha)!=64:
        raise ValueError('Archive source requires a direct download_url and verified sha256')
    cache=Path(repo)/'data'/'.archives';cache.mkdir(parents=True,exist_ok=True)
    dest=cache/(cfg['dataset']+'.download')
    if not dest.exists():
        tmp=dest.with_suffix('.tmp')
        with urlopen(url,timeout=90) as stream,tmp.open('wb') as out:
            shutil.copyfileobj(stream,out)
        tmp.replace(dest)
    if hash_file(dest)!=sha:raise ValueError('Dataset archive checksum mismatch')
    # Extract only into a new directory; reject traversal and links.
    folder.mkdir(parents=True,exist_ok=True)
    if source['type']=='prepared_numpy_zip':
        with zipfile.ZipFile(dest) as archive:
            for item in archive.infolist():
                p=Path(item.filename)
                if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe archive path')
                if item.is_dir():continue
                target=(folder/p).resolve()
                if not target.is_relative_to(folder.resolve()):raise ValueError('Archive escapes destination')
                target.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(item) as f,target.open('wb') as out:shutil.copyfileobj(f,out)
    else:
        with tarfile.open(dest,'r:*') as archive:
            for item in archive:
                p=Path(item.name)
                if p.is_absolute() or '..' in p.parts or not item.isfile():
                    raise ValueError('Only safe regular archive files are supported')
                target=(folder/p).resolve()
                if not target.is_relative_to(folder.resolve()):raise ValueError('Archive escapes destination')
                target.parent.mkdir(parents=True,exist_ok=True)
                f=archive.extractfile(item)
                if f is None:raise ValueError('Unreadable archive entry')
                with f,target.open('wb') as out:shutil.copyfileobj(f,out)
    return folder


def split_manifest(repo,cfg,include_test=False):
    folder=acquire_explicit_archive(repo,cfg)
    splits=('train','val','test') if include_test else ('train','val')
    files={}
    for split in splits:
        for name in ('N','y'):
            p=folder/f'{name}_{split}.npy'
            if not p.is_file():raise FileNotFoundError(f'No official {split} split: {p}')
            files[p.name]={'bytes':p.stat().st_size,'sha256':hash_file(p)}
    info=folder/'info.json'
    if info.exists():files['info.json']={'bytes':info.stat().st_size,'sha256':hash_file(info)}
    provenance=folder/'.source_provenance.json'
    if provenance.exists():files['.source_provenance.json']={'bytes':provenance.stat().st_size,'sha256':hash_file(provenance)}
    return {'dataset':cfg['dataset'],'files':files,'test_included':bool(include_test),
            'source_type':cfg['source']['type'],'path':str(folder.resolve())}


def load_splits(repo,cfg,seed,include_test=False):
    """Fit preprocessing on training data only and transform validation/test."""
    manifest=split_manifest(repo,cfg,include_test=include_test)
    folder=Path(manifest['path'])
    splits=('train','val','test') if include_test else ('train','val')
    X={p:np.array(np.load(folder/f'N_{p}.npy',allow_pickle=False),dtype=np.float32,copy=True) for p in splits}
    y={p:np.asarray(np.load(folder/f'y_{p}.npy',allow_pickle=False),dtype=np.int64) for p in splits}
    expected=cfg['expected']; nfeat=int(expected['input_features']);nclass=int(expected['classes'])
    for p in splits:
        if X[p].ndim !=2 or X[p].shape[1]!=nfeat or y[p].shape!=(len(X[p]),):
            raise ValueError(f'{cfg["dataset"]}/{p}: invalid feature or label dimensions')
        if np.any((y[p]<0)|(y[p]>=nclass)):
            raise ValueError(f'{cfg["dataset"]}/{p}: labels do not match class range')
        if np.isinf(X[p]).any():raise ValueError('Infinite numeric features not supported')
    if expected.get('total_rows') and include_test:
        if sum(len(y[p]) for p in splits)!=int(expected['total_rows']):
            raise ValueError('Prepared dataset has incorrect official number of records')
    if (folder/'info.json').is_file():
        info=json.loads((folder/'info.json').read_text())
        for p in splits:
            if f'{p}_size' in info and int(info[f'{p}_size']) != len(y[p]):
                raise ValueError('Official split size differs from info.json')
    # Preserve source memory order until the final transform, as in the legacy
    # loader: forcing C order changes float32 nanmean reductions on F-order data.
    if any(np.isnan(arr).any() for arr in X.values()):
        mean=np.nanmean(X['train'],axis=0)
        if not np.isfinite(mean).all():raise ValueError('Missing values span an entire training feature')
        for arr in X.values():
            bad=np.where(np.isnan(arr));arr[bad]=mean[bad[1]]
    prep=cfg['preprocessing'];norm=prep['normalization']
    if norm=='quantile':
        n=max(min(len(X['train'])//30,1000),10)
        fit=X['train'].copy()
        noise=.001/np.maximum(np.std(fit,axis=0,keepdims=True),.001)
        fit+=noise*np.random.default_rng(seed).standard_normal(fit.shape)
        scaler=QuantileTransformer(output_distribution='normal',n_quantiles=n,
                                    subsample=1_000_000_000,random_state=seed)
    elif norm=='standard':scaler=StandardScaler();fit=X['train']
    elif norm=='none':scaler=None;fit=None
    else:raise ValueError('Unsupported normalization; specify quantile, standard or none')
    if scaler is not None:
        scaler.fit(fit)
        X={k:np.ascontiguousarray(scaler.transform(v),dtype=np.float32) for k,v in X.items()}
    if any(not np.isfinite(v).all() for v in X.values()):raise ValueError('Nonfinite normalized features')
    return X,y,manifest
