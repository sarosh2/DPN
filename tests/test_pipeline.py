"""Small, fully synthetic arrays test config discovery, training, CSV, tests."""
import copy
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from src.dpn import DPN
from src.config_registry import ensure_model,per_seed_dir,json_hash,atomic_json
from src.data_io import hash_file,read_config
from src.training import atomic_save,run_training,run_staged,isolated_latency
from src.model_factory import instantiate,model_counts
from src.results import read_rows,make_plots
from src.statistics import compute


def _toy(tmp_path):
    root=tmp_path/'toy_repo';root.mkdir()
    shutil.copytree(Path(__file__).resolve().parents[1]/'src',root/'src',ignore=shutil.ignore_patterns('__pycache__'))
    cfg=copy.deepcopy(read_config(Path(__file__).resolve().parents[1],'jannis'))
    cfg.pop('data_acquisition', None)  # synthetic data is not the official archive
    cfg['dataset']='toy';cfg['source']['path']='data/toy'
    cfg['expected']={'input_features':5,'classes':3,'output_logits':3,'total_rows':120,'task_type':'multiclass'}
    cfg['preprocessing']['normalization']='standard'
    cfg['reference_mlp']['hidden_widths']=[9,9]
    cfg['pruning']['initial_hidden_blocks']=30;cfg['pruning']['reference_target_diagnostic_only']=1000
    cfg['training'].update(max_epochs=3,patience=2,batch_size=16,learning_rate=.002)
    cfg['regularization']['lth_mlp']['pretrain_epochs_per_round']=1
    cfg['regularization']['block_matched_dpn']['normalization']='batchnorm'
    folder=root/'configs'/'toy';folder.mkdir(parents=True)
    (folder/'dataset.json').write_text(json.dumps(cfg))
    data=root/'data'/'toy';data.mkdir(parents=True)
    rng=np.random.default_rng(42)
    for part,n in [('train',70),('val',25),('test',25)]:
        np.save(data/f'N_{part}.npy',rng.normal(size=(n,5)).astype('float32'))
        np.save(data/f'y_{part}.npy',rng.integers(0,3,size=n,dtype=np.int64))
    source=DPN(5,29,3,minimal=False,hidden_dims=[10,8,8,3])
    with torch.no_grad():source.mask_1[:,0]=0
    ticket=per_seed_dir(root,'toy',42);ticket.mkdir(parents=True)
    actual=sum(p.numel() for p in source.parameters())
    atomic_save(ticket/'accepted_pruned_ticket.pt',{
        'schema':'dpn_accepted_pruning_v2','dataset':'toy','seed':42,
        'features':5,'output_logits':3,'widths':[10,8,8,3],
        'state':{k:v.detach().cpu() for k,v in source.state_dict().items()},
        'parameters_allocated':actual,'parameters_active':source.count_parameters(),
        'source_dataset_config_sha256':json_hash(cfg)})
    atomic_json(ticket/'pruning_metadata.json',{
        'dataset':'toy','seed':42,'dataset_config_sha256':json_hash(cfg),
        'ticket_sha256':hash_file(ticket/'accepted_pruned_ticket.pt'),
        'parameters_allocated':actual,'parameters_active':source.count_parameters(),
        'hidden_widths':[10,8,8],'output_widths':[3],'pruning_completed':True})
    return root,cfg


def test_seed_model_creation_and_training(tmp_path):
    root,cfg=_toy(tmp_path)
    a=ensure_model(root,'toy',42,'block_matched_dpn',cfg)
    assert (root/'configs'/'toy'/'42'/'matched_mlp.config').exists()
    assert len(a['architecture']['hidden_widths'])==2
    assert a['architecture']['relative_error']<.005
    for name in ('matched_mlp','pruned_dpn','block_matched_dpn','lth_mlp'):
        config=ensure_model(root,'toy',42,name,cfg)
        outcome=run_training(root,cfg,config,device='cpu')
        assert outcome is not None and outcome['result']['best_epoch']>=1
        second=run_training(root,cfg,config,device='cpu')
        assert outcome['result']['checkpoint_fingerprint']==second['result']['checkpoint_fingerprint']
        latency=isolated_latency(root,cfg,config,device='cpu')
        assert latency>0.
    rows=read_rows(root,'training')
    assert len(rows)==4
    images=make_plots(root,['toy'])
    assert images==['toy_test_accuracy.png']
    assert (root/'results'/'images'/'README.md').is_file()


def test_cli_scheduler_and_frozen_test_gate(tmp_path, monkeypatch):
    """Run the notebook CLI backend end-to-end, then verify no test reuse."""
    from src.runner import run
    from src.test_evaluation import evaluate_frozen
    root, _ = _toy(tmp_path)
    ret = run(root, ['toy'], ['42'], ['matched_mlp', 'block_matched_dpn'],
              device='cpu', max_workers_per_gpu=1)
    assert ret['complete'] is True
    rows = read_rows(root, 'training')
    assert len(rows) == 2
    assert all(r['inference_latency_batch1_ms'] for r in rows)
    assert all(not r['test_accuracy'] for r in rows)
    with __import__('pytest').raises(RuntimeError, match='sealed'):
        evaluate_frozen(root, ['toy'], [42], ['matched_mlp', 'block_matched_dpn'],
                        device='cpu', confirm_frozen=False)
    # Reject a stale or incomplete grid before opening any test observations.
    import pytest
    import src.test_evaluation as evaluator
    from src.results import upsert
    original_loader=evaluator.load_splits
    opened=[]
    def tracked_loader(*args, **kwargs):
        if kwargs.get('include_test'):
            opened.append(True)
            assert (root/'results'/'TEST_EVALUATION_FROZEN.json').is_file()
        return original_loader(*args, **kwargs)
    monkeypatch.setattr(evaluator,'load_splits',tracked_loader)
    from filelock import FileLock, Timeout
    with FileLock(str(root/'checkpoints'/'.runner.lock'),timeout=0):
        with pytest.raises(Timeout):
            evaluate_frozen(root,['toy'],[42],['matched_mlp'],confirm_frozen=True)
    assert not opened
    cp=root/'checkpoints'/'toy'/'seed_42'/'matched_mlp'/'best.pt'
    original=cp.read_bytes()
    saved=torch.load(cp,map_location='cpu',weights_only=False)
    saved['fingerprint']='invalid'
    atomic_save(cp,saved)
    with pytest.raises(RuntimeError,match='best checkpoint fingerprint'):
        evaluate_frozen(root,['toy'],[42],['matched_mlp','block_matched_dpn'],confirm_frozen=True)
    cp.write_bytes(original)
    row=next(r for r in rows if r['model']=='matched_mlp')
    upsert(root,'training',{**row,'checkpoint_fingerprint':'invalid'})
    with pytest.raises(RuntimeError,match='matching completed CSV row'):
        evaluate_frozen(root,['toy'],[42],['matched_mlp','block_matched_dpn'],confirm_frozen=True)
    upsert(root,'training',row)
    with pytest.raises(RuntimeError,match='incomplete|config missing'):
        evaluate_frozen(root,['toy'],[42],['matched_mlp','staged_dpn'],confirm_frozen=True)
    assert not opened
    assert not (root/'results'/'TEST_EVALUATION_FROZEN.json').exists()
    evaluate_frozen(root, ['toy'], [42], ['matched_mlp', 'block_matched_dpn'],
                    device='cpu', confirm_frozen=True)
    rows = read_rows(root, 'training')
    assert all(r['test_accuracy'] for r in rows)
    assert len(opened)==1
    frozen=json.loads((root/'results'/'TEST_EVALUATION_FROZEN.json').read_text())
    assert all(len(item['best_sha256'])==64 for item in frozen['frozen_checkpoints'])
    # Alter bytes without changing the checkpoint's embedded fingerprint.
    cp.write_bytes(original+b'changed')
    with pytest.raises(RuntimeError,match='DIFFERENT comparison grid'):
        evaluate_frozen(root,['toy'],[42],['matched_mlp','block_matched_dpn'],confirm_frozen=True)
    assert len(opened)==1
    cp.write_bytes(original)
    with __import__('pytest').raises(RuntimeError, match='frozen'):
        run(root, ['toy'], ['42'], ['matched_mlp'], device='cpu')


def test_welch_raw_seed_confidence_intervals(tmp_path):
    """Measure from per-seed CSV records. No rounded summary input."""
    from src.results import upsert
    from scipy import stats
    root = tmp_path / 'stats_repo'
    root.mkdir()
    for i,seed in enumerate((42,77,53,31,18)):
        upsert(root, 'training', {'dataset':'toy','seed':seed,
              'model':'matched_resnet','status':'complete','val_accuracy':.80+i*.005})
        upsert(root, 'training', {'dataset':'toy','seed':seed,
              'model':'pruned_dpn','status':'complete','val_accuracy':.78+i*.004})
    out = compute(root,split='val')
    assert len(out)==1
    row=out[0]
    assert row['mean_difference_pp']>0
    assert row['ci95_lower_pp'] <= row['mean_difference_pp'] <= row['ci95_upper_pp']
    assert 0 <= row['p_value_two_sided'] <= 1
    assert row['paired_n']==5
    assert row['holm_p_value']==row['p_value_two_sided']


def test_one_time_legacy_import_exact_configs_and_results(tmp_path):
    """Mock the actual Step83 + Step89 artifact layout; no fake scores shipped."""
    import pytest
    imp=pytest.importorskip('scripts.import_legacy_tickets',
        reason='One-time legacy importer has been deleted after migration')
    from src.model_factory import MODEL_NAMES
    from src.config_registry import model_path
    import csv
    root=tmp_path/'portable';root.mkdir()
    shutil.copytree(Path(__file__).resolve().parents[1]/'src',root/'src',ignore=shutil.ignore_patterns('__pycache__'))
    config=copy.deepcopy(read_config(Path(__file__).resolve().parents[1],'jannis'))
    config['pruning']['initial_hidden_blocks']=30
    (root/'configs'/'jannis').mkdir(parents=True)
    (root/'configs'/'jannis'/'dataset.json').write_text(json.dumps(config))
    seed=42;ds='jannis';ex=config['expected']
    canonical=DPN(ex['input_features'],34,ex['output_logits'],minimal=False,
                  hidden_dims=[1]*34)
    source=tmp_path/'old_results'/'step83_jannis_full_4h'/'tasks'/ds/f'seed_{seed}'
    folder=source/'discovery';folder.mkdir(parents=True)
    allocated=sum(x.numel() for x in canonical.parameters())
    active=int(canonical.count_parameters())
    raw={'dataset':ds,'seed':seed,'widths':[1]*34,
         'state':{k:v.detach().cpu().clone() for k,v in canonical.state_dict().items()}}
    atomic_save(folder/'ticket.pt',raw)
    atomic_json(folder/'discovery.json',{
        'dataset':ds,'seed':seed,'ticket_sha256':hash_file(folder/'ticket.pt'),
        'qualification':{'near_reference_scale':True},
        'ticket':{'parameters_allocated':allocated,'parameters_active':active},
        'pruning_result':{'wall_seconds':12.0}})
    # Build temporary model specs so mock frozen historical contracts describe
    # true geometry. They will then be discarded and re-imported.
    target=per_seed_dir(root,ds,seed);target.mkdir(parents=True)
    atomic_save(target/'accepted_pruned_ticket.pt',{
        'schema':'dpn_accepted_pruning_v2','dataset':ds,'seed':seed,
        'features':ex['input_features'],'output_logits':ex['output_logits'],
        'widths':[1]*34,'state':raw['state'],
        'parameters_allocated':allocated,'parameters_active':active,
        'source_dataset_config_sha256':json_hash(config)})
    atomic_json(target/'pruning_metadata.json',{
        'dataset':ds,'seed':seed,'dataset_config_sha256':json_hash(config),
        'ticket_sha256':hash_file(target/'accepted_pruned_ticket.pt'),
        'parameters_allocated':allocated,'parameters_active':active,
        'hidden_widths':[1]*30,'output_widths':[1]*4,
        'pruning_completed':True})

    def write_arm(arm,spec,variant=None,base=None):
        where=(base or source)/'arms'/(arm+('__'+variant if variant else ''))
        where.mkdir(parents=True,exist_ok=True)
        model=instantiate(spec,config,seed,target,variant)
        act,alloc=model_counts(model)
        arch=spec['architecture'] if variant is None else spec['architecture']['candidates'][variant]
        if arm in ('matched_mlp','lth_mlp'):
            original={'widths':arch['hidden_widths'],'paired_init_sha256':'toy-matched'}
        elif arm=='matched_resnet':
            original={k:arch[k] for k in ('d','n_layers','d_hidden_factor','activation')}
        elif arm in ('staged_dpn','parameter_matched_max_dpn','block_matched_dpn'):
            original={'config':{k:arch.get(k,False if k=='raw_input_bypass' else [])
                                for k in ('hidden_widths','output_widths','stage_boundaries',
                                          'transition_widths','raw_input_bypass')}}
        else:original={}
        atomic_json(where/'contract.json',{
            'dataset':ds,'seed':seed,'arm':arm,'spec':original,
            'hyperparameters':{'learning_rate':.0004,'batch_size':512,
                               'weight_decay':.0001,'patience':16}})
        result={'dataset':ds,'seed':seed,'arm':arm,'variant':variant,
             'completed':True,'metrics_split':'validation','test_data_accessed':False,
             'epochs':10,'best_epoch':7,'train_wall_seconds':120,
             'allocated':alloc,'active':act,'budget':allocated,
             'relative_gap':abs(allocated-alloc)/allocated,
             'metrics':{'accuracy':.70,'macro_f1':.51,'log_loss':.85}}
        atomic_json(where/'result.json',result)
        (where/'result.sha256').write_text(hash_file(where/'result.json'))
        return where
    for arm in ('pruned_dpn','matched_mlp','lth_mlp','matched_resnet',
                'parameter_matched_max_dpn','block_matched_dpn','staged_dpn'):
        spec=ensure_model(root,ds,seed,arm,config)
        if arm=='staged_dpn':
            valid=[]
            for v,arch in spec['architecture']['candidates'].items():
                if arch['status']=='infeasible':
                    d=source/'arms'/f'staged_dpn__{v}';d.mkdir(parents=True,exist_ok=True)
                    atomic_json(d/'INFEASIBLE.json',{'reason':'candidate infeasible'})
                else:
                    write_arm(arm,spec,v)
                    valid.append(v)
            if not valid:
                raise AssertionError('Toy budget must support staged candidate')
            d=source/'arms'/'staged_dpn';d.mkdir(parents=True,exist_ok=True)
            atomic_json(d/'selected.json',{'variant':valid[0]})
        elif arm=='block_matched_dpn':
            write_arm(arm,spec,base=tmp_path/'old_results'/'step89_shallow_block_regularized_4gpu'/'tasks'/ds/f'seed_{seed}')
        else:write_arm(arm,spec)
    shutil.rmtree(target)
    from unittest.mock import patch
    with patch('sys.argv',['import_legacy_tickets.py',
            '--previous-results',str(tmp_path/'old_results'),
            '--new-root',str(root),'--datasets','jannis','--seeds','42']):
        imp.main()
        imp.main()  # safe idempotency
    rows=read_rows(root,'training')
    assert len(rows)==7
    assert len(list(target.glob('*.config')))==7
    assert all(r['status']=='complete' and not r['test_accuracy'] for r in rows)
    assert any(r['model']=='block_matched_dpn' and r['inference_latency_batch1_ms']=='' for r in rows)
    assert (root/'results'/'images'/'jannis_validation_accuracy.png').is_file()
    assert (root/'results'/'csvs'/'welch_tests.csv').is_file()


def test_small_outer_pruning_on_cpu(tmp_path):
    """Regression: missing pruning_checkpoint and hard-coded CUDA no longer break discovery."""
    from src.pruning import discover
    root,cfg=_toy(tmp_path)
    cfg['pruning'].update(initial_hidden_blocks=7,outer_initial_rounds=2,
                          candidate_training_epochs=1)
    (root/'configs'/'toy'/'dataset.json').write_text(json.dumps(cfg))
    folder=per_seed_dir(root,'toy',42)
    (folder/'accepted_pruned_ticket.pt').unlink()
    (folder/'pruning_metadata.json').unlink()
    result=discover(root,cfg,42,'cpu')
    assert result['actual_rounds']>0
    assert result['pruning_completed'] is True
    assert (folder/'pruning_rounds_2.pt').is_file()
    # Using the same exact pruned ticket does not start structural discovery again.
    assert discover(root,cfg,42,'cpu')['ticket_sha256']==result['ticket_sha256']


def test_official_data_acquisition_reuses_model_config_hash_and_splits(tmp_path):
    """Mock official tar: config-only acquisition, stable imported-ticket hash."""
    import tarfile
    import hashlib
    import json
    import copy
    from src.config_registry import json_hash
    from src.data_io import read_config, load_splits, split_manifest
    source_root = Path(__file__).resolve().parents[1]
    legacy = json.loads((source_root/'configs'/'covtype'/'dataset.json').read_text())
    unmodified = copy.deepcopy(legacy)
    unmodified.pop('data_acquisition')
    assert json_hash(legacy) == json_hash(unmodified)
    cfg = copy.deepcopy(legacy)
    cfg['dataset'] = 'toy'
    cfg['source']['path'] = 'data/toy'
    cfg['expected'].update(total_rows=23, input_features=5, classes=3, output_logits=3)
    cfg['preprocessing']['normalization'] = 'standard'
    cfg['data_acquisition']['archive_subdir'] = 'data/toy'
    staging = tmp_path/'archive_staging'/'data'/'toy'
    staging.mkdir(parents=True)
    rng = np.random.default_rng(123)
    for part, n in [('train', 11), ('val', 7), ('test', 5)]:
        np.save(staging/f'N_{part}.npy', rng.normal(size=(n, 5)).astype('float32'))
        np.save(staging/f'y_{part}.npy', rng.integers(0, 3, size=n, dtype=np.int64))
    (staging/'info.json').write_text(json.dumps({'train_size':11, 'val_size':7, 'test_size':5}))
    archive = tmp_path/'data.tar.gz'
    with tarfile.open(archive, 'w:gz') as dst:
        for member in sorted(staging.iterdir()):
            dst.add(member, arcname=f'data/toy/{member.name}')
    cfg['data_acquisition']['download_url'] = archive.as_uri()
    cfg['data_acquisition']['sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
    cfg['data_acquisition']['file_sha256'] = {p.name:hash_file(p) for p in staging.iterdir()}
    configdir = tmp_path/'configs'/'toy'
    configdir.mkdir(parents=True)
    (configdir/'dataset.json').write_text(json.dumps(cfg))
    actual = read_config(tmp_path, 'toy')
    X, y, manifest = load_splits(tmp_path, actual, seed=42, include_test=False)
    assert X['train'].shape == (11,5) and X['val'].shape == (7,5)
    assert 'test' not in X and 'N_test.npy' not in manifest['files']
    downloaded=tmp_path/'data'/'toy'
    assert (downloaded/'N_test.npy').exists()  # sealed on disk, not loaded
    receipt = json.loads((downloaded/'.source_provenance.json').read_text())
    assert receipt['split_policy'] == 'publisher train/val/test splits, no resampling'
    assert len(receipt['files']) == 7
    assert (tmp_path/'data'/'.archives'/'rtdl_archive_manifest.json').exists()
    # Must reuse prepared arrays without reading or requiring the remote archive.
    archive.unlink()
    X2, y2, manifest2 = load_splits(tmp_path, actual, seed=42, include_test=False)
    assert np.array_equal(X['train'], X2['train'])
    assert manifest == manifest2

    # Cached files are verified even when the local receipt is missing.
    import pytest
    provenance = downloaded/'.source_provenance.json'
    provenance.unlink()
    split_manifest(tmp_path, actual, include_test=False)
    # Test bytes are hashed but not deserialized in validation-only runs.
    from unittest.mock import patch
    original_load = np.load
    def guarded_load(path, *args, **kwargs):
        assert '_test.npy' not in str(path)
        return original_load(path, *args, **kwargs)
    with patch('src.data_io.np.load', side_effect=guarded_load):
        load_splits(tmp_path, actual, seed=42, include_test=False)
    for name in ('N_train.npy', 'y_val.npy', 'N_test.npy', 'info.json'):
        path = downloaded/name
        original = path.read_bytes()
        path.write_bytes(original + b'corruption')
        with pytest.raises(ValueError, match='Official file SHA256 mismatch'):
            split_manifest(tmp_path, actual, include_test=False)
        path.write_bytes(original)
    # A forged receipt is not an authority.
    provenance.write_text(json.dumps({**receipt, 'archive_sha256':'0'*64}))
    with pytest.raises(ValueError, match='provenance mismatch'):
        split_manifest(tmp_path, actual, include_test=False)
    provenance.unlink()
    # Fresh extraction must reject incorrect pins and leave no installed data.
    shutil.rmtree(downloaded)
    bad = copy.deepcopy(actual)
    bad['data_acquisition']['file_sha256']['N_train.npy'] = '0'*64
    with pytest.raises(ValueError, match='Official file SHA256 mismatch'):
        split_manifest(tmp_path, bad, include_test=False)
    assert not downloaded.exists()
    # Cached archive integrity is enforced when extraction is required.
    cached = tmp_path/'data'/'.archives'/'rtdl_revisiting_models_data.tar.gz'
    original = cached.read_bytes()
    cached.write_bytes(original + b'corruption')
    with pytest.raises(ValueError, match='Official archive SHA256'):
        split_manifest(tmp_path, actual, include_test=False)
    cached.unlink()
    # A bad download must not poison the archive cache.
    archive.write_bytes(original + b'corruption')
    with pytest.raises(ValueError, match='Official archive SHA256'):
        split_manifest(tmp_path, actual, include_test=False)
    assert not cached.exists() and not downloaded.exists()
    archive.write_bytes(original)
    split_manifest(tmp_path, actual, include_test=True)
    assert downloaded.exists()


def test_builtin_data_pins_and_legacy_fingerprints():
    from src.data_io import _official_pins
    root = Path(__file__).resolve().parents[1]
    archives = set()
    for ds in ('covtype', 'jannis', 'helena', 'higgs_small'):
        cfg = read_config(root, ds)
        pins = _official_pins(cfg['data_acquisition'])
        assert len(pins) == 7
        archives.add(cfg['data_acquisition']['sha256'])
        legacy = copy.deepcopy(cfg)
        legacy.pop('data_acquisition')
        assert json_hash(cfg) == json_hash(legacy)
    assert len(archives) == 1


def test_legacy_quantile_preprocessing_preserves_layout(tmp_path):
    from sklearn.preprocessing import QuantileTransformer
    from src.data_io import load_splits

    # Reproduce the historical prepare_train_val sequence independently.
    rng = np.random.default_rng(42)
    cfg = {'dataset':'toy', 'source':{'type':'prepared_numpy','path':'data/toy'},
           'expected':{'input_features':28,'classes':2,'total_rows':7275},
           'preprocessing':{'normalization':'quantile'}}
    folder = tmp_path/'data'/'toy'
    folder.mkdir(parents=True)
    for order in ('C', 'F'):
        raw = {}
        for part, n in (('train',6275),('val',1000)):
            x = np.array(rng.normal(loc=3,size=(n,28)), dtype=np.float32, order=order)
            x[::11,0] = np.nan
            np.save(folder/f'N_{part}.npy', x)
            np.save(folder/f'y_{part}.npy', np.zeros(n,dtype=np.int64))
            raw[part] = np.load(folder/f'N_{part}.npy', allow_pickle=False)
        assert raw['train'].flags.f_contiguous == (order == 'F')
        for seed in (42,77,53,31,18):
            expected = {k:np.array(v,dtype=np.float32,copy=True) for k,v in raw.items()}
            means = np.nanmean(expected['train'], axis=0)
            for values in expected.values():
                mask = np.where(np.isnan(values))
                values[mask] = means[mask[1]]
            fit = expected['train'].copy()
            noise = 1e-3
            stds = np.std(fit,axis=0,keepdims=True)
            noise_std = noise / np.maximum(stds,noise)
            fit += noise_std*np.random.default_rng(seed).standard_normal(fit.shape)
            scaler = QuantileTransformer(output_distribution='normal',
                n_quantiles=max(min(len(fit)//30,1000),10),
                subsample=1_000_000_000,random_state=seed).fit(fit)
            actual, labels, manifest = load_splits(tmp_path,cfg,seed=seed,include_test=False)
            for part in ('train','val'):
                reference = np.ascontiguousarray(scaler.transform(expected[part]),dtype=np.float32)
                np.testing.assert_array_equal(actual[part],reference)
                assert actual[part].flags.c_contiguous and actual[part].dtype == np.float32
            assert set(actual) == set(labels) == {'train','val'} and not manifest['test_included']
