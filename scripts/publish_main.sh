#!/usr/bin/env bash
# Preview locally by default. Review the remote diff, then explicitly publish.
set -euo pipefail
MODE="${1:---preview}"
if [[ $# -gt 1 || ! "$MODE" =~ ^--(preview|review|publish)$ ]]; then
    echo 'Usage: bash scripts/publish_main.sh [--preview|--review|--publish]' >&2
    exit 2
fi
URL="${DPN_GITHUB_URL:-https://github.com/sarosh2/DPN.git}"
SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${DPN_PYTHON:-python}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/dpn-release-XXXXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# Explicit publication snapshot; never copy raw data, training checkpoints,
# unreviewed files, logs, runtime locks, caches, credentials, or freeze state.
"$PY" - "$SOURCE" "$WORK/stage" <<'PY'
import csv, hashlib, json, shutil, sys
from pathlib import Path
from collections import Counter
source, stage = map(Path, sys.argv[1:])
datasets = ('covtype', 'jannis', 'helena', 'higgs_small')
seeds = (42, 77, 53, 31, 18)
models = ('matched_mlp', 'matched_resnet', 'lth_mlp', 'pruned_dpn',
          'block_matched_dpn', 'parameter_matched_max_dpn', 'staged_dpn')

def require(ok, message):
    if not ok:
        raise SystemExit('RELEASE REJECTED: ' + message)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def cfg_hash(obj):
    if obj.get('schema') == 'dpn_dataset_v2':
        obj = {k: v for k, v in obj.items() if k != 'data_acquisition'}
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()

require(not (source / 'results/TEST_EVALUATION_FROZEN.json').exists(),
        'this release is validation-only; frozen test manifest found')
files = {Path(p) for p in ('.gitignore', 'README.md', 'requirements.txt',
    'run.py', 'DPN_Benchmark.ipynb', 'data/README.md', 'checkpoints/README.md',
    'configs/README.md', 'results/README.md', 'results/csvs/README.md',
    'results/images/README.md', 'scripts/README.md', 'scripts/publish_main.sh',
    'scripts/import_legacy_tickets.py', 'scripts/slurm_example.sbatch')}
# Enumerate the reviewed implementation and tests, not arbitrary new files.
files.update(Path(p) for p in (
    'src/README.md', 'src/__init__.py', 'src/config_registry.py',
    'src/data_io.py', 'src/dpn.py', 'src/model_factory.py',
    'src/pruning.py', 'src/pruning_checkpoint.py', 'src/results.py',
    'src/runner.py', 'src/statistics.py', 'src/test_evaluation.py',
    'src/training.py', 'src/utils.py', 'src/worker.py',
    'src/models/README.md', 'src/models/__init__.py', 'src/models/dpn.py',
    'src/models/dpn_flexible.py', 'src/models/mlp.py', 'src/models/resnet.py',
    'tests/README.md', 'tests/test_geometry.py', 'tests/test_pipeline.py'))
for name in ('LICENSE', 'LICENSE.md', 'LICENSE.txt', 'COPYING', 'COPYING.md',
             'NOTICE', 'NOTICE.md', 'NOTICE.txt', 'CITATION.cff'):
    if (source / name).exists():
        files.add(Path(name))
for name in ('pruning', 'training', 'welch_tests', 'historical_validation_summary',
             'historical_complexity_summary'):
    files.add(Path('results/csvs') / (name + '.csv'))
for d in datasets:
    files.update((Path('configs') / d / 'README.md', Path('configs') / d / 'dataset.json'))
    for split in ('validation', 'test'):
        files.add(Path('results/images') / f'{d}_{split}_accuracy.png')
    for s in seeds:
        base = Path('configs') / d / str(s)
        files.update(base / n for n in ('README.md', 'accepted_pruned_ticket.pt', 'pruning_metadata.json'))
        files.update(base / (m + '.config') for m in models)

for rel in sorted(files):
    p = source / rel
    require(p.is_file(), f'missing {rel}')
    require(not any((source / q).is_symlink() for q in (rel, *rel.parents)),
            f'symlink not permitted: {rel}')
    require(p.stat().st_size < 50 * 2**20, f'file exceeds 50 MiB release limit: {rel}')
    dest = stage / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dest)

# Validate the staged copy rather than merely validating its source paths.
def read_json(rel):
    return json.loads((stage / rel).read_text())
def rows(name):
    with (stage / 'results/csvs' / (name + '.csv')).open(newline='') as f:
        return list(csv.DictReader(f))

def unique_grid(records, key, expected, name):
    got = Counter(key(r) for r in records)
    require(set(got) == expected and all(n == 1 for n in got.values()),
            f'{name}: missing, duplicated or unexpected records')

expected_tickets = {(d,s) for d in datasets for s in seeds}
expected_models = {(d,s,m) for d,s in expected_tickets for m in models}
training, pruning, welch = rows('training'), rows('pruning'), rows('welch_tests')
unique_grid(training, lambda r:(r['dataset'],int(r['seed']),r['model']), expected_models,'training')
unique_grid(pruning, lambda r:(r['dataset'],int(r['seed'])), expected_tickets,'pruning')
require(len(welch) == 48 and Counter(r['dataset'] for r in welch) == Counter({d:12 for d in datasets}),
        'expected 48 Welch comparisons')
for row in training:
    require(row['status'] == 'complete' and row['split'] == 'validation' and row.get('val_accuracy','') != '',
            'non-complete validation row')
    require(all(not v for k,v in row.items() if k.startswith('test_') and k not in ('test_accessed','test_evaluation_frozen'))
            and all(str(row.get(k,'')).lower() in ('','false','0') for k in ('test_accessed','test_evaluation_frozen')),
            'test result or test-access flag found')
by_model = {(r['dataset'],int(r['seed']),r['model']):r for r in training}
by_ticket = {(r['dataset'],int(r['seed'])):r for r in pruning}
for d,s in sorted(expected_tickets):
    cfg = read_json(Path('configs') / d / 'dataset.json')
    dh = cfg_hash(cfg)
    base = Path('configs') / d / str(s)
    meta = read_json(base / 'pruning_metadata.json')
    th = sha(stage / base / 'accepted_pruned_ticket.pt')
    require(meta['dataset'] == d and meta['seed'] == s and meta['dataset_config_sha256'] == dh
            and meta['ticket_sha256'] == th, f'ticket identity mismatch {d}/{s}')
    require(by_ticket[d,s]['ticket_sha256'] == th and by_ticket[d,s]['source_config_sha256'] == dh,
            f'pruning CSV mismatch {d}/{s}')
    for m in models:
        spec = read_json(base / (m + '.config'))
        row = by_model[d,s,m]
        require((spec['dataset'],spec['seed'],spec['model']) == (d,s,m) and
                spec['ticket_sha256'] == th and spec['dataset_config_sha256'] == dh and
                row['ticket_sha256'] == th and row['dataset_config_sha256'] == dh and
                row['model_config_sha256'] == cfg_hash(spec), f'model/CSV mismatch {d}/{s}/{m}')

print('Verified staged release: 20 tickets, 140 model configs, 140 validation rows, 48 Welch comparisons.')
print('No raw data, runtime checkpoints, caches, locks or test-evaluation state staged.')
print(f'Total: {len(files)} files, {sum((stage/p).stat().st_size for p in files)/2**20:.2f} MiB')
# Existing original-path metadata is provenance, not a runtime dependency.
# Flag it for inspection, rather than silently altering hashed configurations.
for rel in sorted(files):
    if rel.suffix in ('.json', '.config', '.csv', '.md', '.ipynb'):
        if '/scratch/' in (stage / rel).read_text(errors='replace'):
            print('Review historical/cluster path in:', rel)
PY

if [[ "$MODE" == '--preview' ]]; then
    echo 'LOCAL RELEASE PREVIEW PASSED — no remote contacted or changed.'
    exit 0
fi

# Remote review and explicit publication use the same staging procedure.
git clone --quiet --branch main --single-branch "$URL" "$WORK/repo"
cd "$WORK/repo"
OLD_MAIN="$(git rev-parse HEAD)"
check_remote() {
    local current backup
    current="$(git ls-remote --heads origin refs/heads/main | awk '{print $1}')"
    backup="$(git ls-remote --heads origin refs/heads/dpn1.0 | awk '{print $1}')"
    [[ "$current" == "$OLD_MAIN" ]] || { echo 'Remote main moved; stop.' >&2; exit 1; }
    [[ -z "$backup" || "$backup" == "$OLD_MAIN" ]] || {
        echo 'Existing dpn1.0 differs from old main; stop without replacing it.' >&2; exit 1; }
}
check_remote
# Preserve existing root legal/citation files if the release has no replacement.
for name in LICENSE LICENSE.md LICENSE.txt COPYING COPYING.md NOTICE NOTICE.md NOTICE.txt CITATION.cff; do
    if [[ -f "$name" && ! -L "$name" && ! -e "$WORK/stage/$name" ]]; then
        cp -p "$name" "$WORK/stage/$name"
        echo "Preserved existing $name"
    fi
done
if ! compgen -G "$WORK/stage/LICENSE*" >/dev/null && ! compgen -G "$WORK/stage/COPYING*" >/dev/null; then
    echo 'NOTE: no root license found; review licensing before publishing.'
fi
git rm -q -r --ignore-unmatch -- .
cp -a "$WORK/stage/." .
git add -A
# No unexpected or ignored files in the actual Git index.
"$PY" - "$WORK/stage" <<'PY'
from pathlib import Path
import subprocess,sys
root=Path(sys.argv[1]); expected={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
actual={}
for record in subprocess.check_output(['git','ls-files','--stage','-z']).split(b'\0'):
    if record:
        meta, path = record.split(b'\t',1)
        actual[path.decode()] = meta.split()[1].decode()
if set(actual) != expected:
    raise SystemExit(f'Git index differs from release snapshot: missing={expected-set(actual)}; extra={set(actual)-expected}')
for path, oid in actual.items():
    digest=subprocess.check_output(['git','hash-object','--no-filters',str(root/path)],text=True).strip()
    if digest != oid:
        raise SystemExit(f'Git index content differs (possible Git filter): {path}')
PY
git diff --cached --stat
printf '\nOld main: %s\n' "$OLD_MAIN"
if [[ "$MODE" == '--review' ]]; then
    echo 'REMOTE REVIEW PASSED — no branches pushed.'
    exit 0
fi
if git diff --cached --quiet; then
    echo 'No changes to publish.'; exit 0
fi
git commit -m 'Release configuration-driven DPN benchmark v2'
check_remote
# Both references update together or neither updates. No force options.
git push --atomic origin HEAD:refs/heads/main "$OLD_MAIN:refs/heads/dpn1.0"
echo "PUBLISHED: $URL"
echo "BACKUP: dpn1.0 -> $OLD_MAIN"
echo "NEW main -> $(git rev-parse HEAD)"
