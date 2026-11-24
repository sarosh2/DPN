# Denser Perceptron Networks — reproducible benchmark

This repository trains seven models on tabular datasets. It uses the **original DPN outer pruning routine** and the regularized model policies. It creates missing model configs from a verified pruned DPN. The shallow Block DPN matches the depth and allocated parameter count of the matched MLP, not the pruned DPN.

**Current benchmark:** Covertype, Jannis, Helena and **HIGGS Small** (98,050 samples; not HIGGS-500k). Five seeds: 42, 77, 53, 31, 18. Training uses 300 epochs at most, with early stopping after more than 16 epochs without better validation accuracy. The test set stays sealed during training.

## Repository layout

- `src/`: Original `utils.py`, `dpn.py`, original model classes, data loader, config resolver, pruning bridge, trainer, GPU scheduler and Welch tests.
- `configs/<dataset>/dataset.json`: The **only required config**. The runner creates `<seed>/<model>.config` files when their dependencies are ready.
- `DPN_Benchmark.ipynb`: Preview datasets/seeds/models and device assignments, display saved results, or explicitly enable training and final evaluation.
- `run.py`: Non-interactive version of the notebook.
- `results/csvs/`: Per-seed pruning and training CSVs and Welch test results.
- `results/images/`: Validation figures and test-accuracy boxplots built from CSV values; test results stay pending until real test evaluation.
- `checkpoints/`: Pruning snapshots and per-epoch model/optimizer/RNG state. Large checkpoint files are ignored by Git.
- `scripts/`: Slurm example, old-run ticket migration, and safe GitHub publication commands.


## Included benchmark artifacts

The public release includes **20 accepted, seed-specific pruning tickets**,
**140 model configs**, **140 historical per-seed validation records**, and
48 validation-accuracy Welch comparisons. The tickets supply the actual
pruned geometries and masks; the ordinary training checkpoints are not
included. The historical measurements are from the original regularized
benchmark (including corrected Step 89 shallow Block DPN), not new DPN-v2
training runs. Rounded five-seed summaries are also retained separately in
`results/csvs/historical_validation_summary.csv` and
`historical_complexity_summary.csv`.

Reviewers do not need the original research directory or migration scripts
to use the included tickets. The optional importer is retained for provenance;
see `scripts/README.md`. To preserve the historical results when retraining,
work in a separate clone or copy of the repository.

## Download and prepare the benchmark datasets

No datasets are stored in Git. The dataset configs for Covertype, Jannis,
Helena and HIGGS Small link to the original published prepared-data archive.
The first real run downloads and caches the shared archive, extracts the
requested official splits, and records file hashes. The archive already has
train/validation/test partitions: **no additional random splitting occurs**.
Preprocessing fits on train only. Validation-only training never loads test
observations or labels for modeling; raw test-file bytes are hashed for
integrity, and fresh extraction checks their array headers.

The first download needs internet access and archive-sized local disk space.
If the publisher link fails, update only the dataset config download URL.
The configs pin the project-verified archive SHA256 and all seven per-dataset
file SHA256 values (`data_acquisition.sha256` and `file_sha256`). These were
matched against the original benchmark data on 2026-11-24, not supplied by
the publisher. Both cached prepared files and new extractions must match the
pins. The archive itself is verified whenever extraction is needed.

Existing imported accepted tickets and model configs remain valid:
acquisition metadata is excluded from the legacy scientific config hash.
Data bytes are checked against the pins and recorded in split manifests.
Changing data pins is not a supported way to reuse tickets for different
data; use a new experiment. See `data/README.md`.

## Portable local execution (CUDA, Apple Metal, CPU)

The notebook previews the requested five training families by default: matched MLP,
ResMLP, LTH-MLP, corrected shallow regularized Block DPN, and staged DPN.
All seven configurations are still generated during migration. The imported
pruning ticket is a **dependency**, not a request to retrain pruned DPN.
With `--no-pruning`, a missing ticket stops with an error instead of starting
structural discovery.

```bash
python run.py --datasets jannis helena higgs_small covtype \
  --seeds 42 77 53 31 18 \
  --models matched_mlp matched_resnet lth_mlp block_matched_dpn staged_dpn \
  --device auto --max-workers-per-gpu 8 --no-pruning
```

`--device auto` detects CUDA, then Apple MPS, then CPU. On CUDA it uses
visible-device assignment and accelerator telemetry when provided. On Apple
Metal it uses a bounded concurrent worker limit; on CPU it uses detected
cores. The same command runs outside Slurm. There is no RTX/GH200/MacBook
profile. Resource-adaptive scheduling is best-effort, not a guarantee of
100% busy time, and RTX 4060 execution has not yet been validated.

For Slurm, `scripts/slurm_example.sbatch` is **optional**. Set `DPN_NO_PRUNING=1` to forbid structural discovery. Supply
site-specific allocation flags at submission time, for example:

```bash
sbatch --partition=YOUR_PARTITION --gres=gpu:1 \
  --time=02:00:00 scripts/slurm_example.sbatch
```

No training command depends on Slurm, and no new script is needed for a
different GPU. Notebook and CLI both call `src.runner.run`. The only normal
code changes should be to existing shared modules; add dataset configs to
introduce data sources.

## Install

Use an appropriate Python/PyTorch environment for your system, then:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

On CSC Roihu, load `python-pytorch` first and use the site-supported environment.
Dependency ranges are not a fully pinned environment; the verification record
below specifies the principal versions used on Roihu.

Alternatively, enable `INSTALL_REQUIREMENTS` in the first code cell of `DPN_Benchmark.ipynb` if dependencies are missing. The first real run downloads the official prepared splits if they are absent. You can also place the original files in `data/<dataset>/` manually. The program does not create new splits.

## Notebook workflow

Open `DPN_Benchmark.ipynb` from the repository root using the intended Python kernel. By default **Run All** previews the selection and displays saved CSVs and figures; it does not install packages, download data, train, refresh reports or evaluate test observations. Enable `REFRESH_REPORTS` to regenerate figures and validation statistics. Enable `RUN_TRAINING` to train or resume; leave `ALLOW_DISCOVERY=False` to require imported tickets. LTH-MLP still runs its internal mask/rewind training. Use a compute allocation for actual training on a cluster. The notebook defaults to eight workers per GPU, configurable for the hardware. Retraining can replace imported result rows; use a separate experiment copy to preserve them. Enable `RUN_FROZEN_TEST` only in a separate execution with training disabled and compatible completed checkpoints. The freeze blocks later training. On non-CUDA systems the notebook evaluates on CPU.

## Run

```bash
# Show the available GPU allocation and selected tasks without training.
python run.py --datasets all --seeds 42 77 --models matched_mlp block_matched_dpn --dry-run

# Train only the corrected, regularized shallow Block DPN; create dependencies.
python run.py --datasets covtype jannis helena higgs_small \
  --seeds 42 77 53 31 18 --models block_matched_dpn \
  --max-workers-per-gpu 8 --gpu-util-target 0.92 --gpu-memory-limit 0.90

# Run all seven families; use the current resource allocation.
python run.py --datasets all --seeds all --models all

# Update plots from existing CSV files without training.
python run.py --plots-only

# Compute two-sided Welch tests on per-seed validation accuracy.
python -m src.statistics --repo . --split val --metric accuracy
```

The runner checks `CUDA_VISIBLE_DEVICES` and available GPU memory. It assigns one dataset per GPU when enough GPUs exist. It admits more independent tasks when the GPU is not full. The CLI default is 16 tasks per GPU; eight was sufficient for the measured Covertype workload on four GH200s. The runner does not create duplicate tasks to increase load. The optional Slurm example is `scripts/slurm_example.sbatch`.

## Config dependencies

1. The runner reads `configs/<dataset>/dataset.json`. It checks the **specified** dataset source and its official split.
2. If `configs/<dataset>/<seed>/accepted_pruned_ticket.pt` is absent, the runner starts the saved original `src.utils.apply_lth_pruning()` outer search. It uses 50, 45, …, 5 rounds, with three training epochs per candidate. It stores intermediate `pruning_rounds_*.pt` files.
3. The accepted pruning ticket fixes the **real allocated** parameter budget for that dataset/seed.
4. The resolver creates the MLP, ResMLP, LTH-MLP, maximal DPN and staged DPN configs from that ticket. The LTH-MLP shares the MLP's initial weights and runs two mask/rewind rounds (50% and 90%). Two staged candidates are trained; validation selects one.
5. **Block DPN exception:** The resolver first creates the matched MLP config. It then makes a cumulative-connectivity DPN with the **same hidden depth** and within 0.5% of that seed's MLP parameter count.
6. A worker trains a requested model only when its config is ready. It saves a checkpoint after each epoch. It resumes only when dataset, model, split, code and optimizer fingerprint agree. It refuses incompatible checkpoints.

To add a dataset, create `configs/my_dataset/dataset.json`. Specify the original prepared split path or a direct URL and SHA-256, dimensions, target task, preprocessing, pre-pruning DPN, MLP reference shape, optimizer and all seven regularization policies. The program will **not guess** missing details. See `configs/jannis/dataset.json` for a full example.

## Metrics and evaluation

`results/csvs/pruning.csv` has the requested pruning rounds, connection/allocated parameter counts, block reduction, final layout, duration and latency. `results/csvs/training.csv` has one record for each dataset/seed/model. It records accuracy, macro-F1, balanced accuracy, log-loss, Brier score, 15-bin calibration error, AUROC when defined, training details, active/allocated counts, and batch-1 inference latency. Missing metrics remain blank.

The training command never opens a test `.npy` array. **Test evaluation is a separate, explicit freeze action:**

```bash
python -m src.test_evaluation --repo . \
  --datasets covtype jannis helena higgs_small \
  --seeds 42 77 53 31 18 \
  --models pruned_dpn matched_mlp matched_resnet lth_mlp block_matched_dpn parameter_matched_max_dpn staged_dpn \
  --device cuda --confirm-frozen
```

Use this only when the selected comparison grid is complete. It freezes the validation selections and blocks later training in this result directory. Test figures use only genuine per-seed test accuracies. Before any test observations are loaded, the evaluator verifies every selected completion record, current training fingerprint, matching CSV row, and best checkpoint fingerprint and epoch. The freeze manifest records the SHA-256 of each selected best checkpoint. Training and evaluation share the same exclusive runner lock. Historical imported validation rows alone are not sufficient for test evaluation; compatible trained checkpoints are required.

## Device-neutral discovery and its scope

The original outer pruning algorithm is preserved. Its one hard-coded
`.cuda()` device transfer now uses `.to(device)` so a small dataset may also
run discovery on CPU or Apple Metal. This changes device handling only, not
model selection or the pruning schedule. Do not run Covertype structural
discovery on a laptop. Use `--no-pruning` with imported tickets.

## Verification status and scientific limits

This version uses the original canonical DPN and outer pruning logic with
the documented device-handling correction and the later regularized model
policies. The following checks were performed on CSC Roihu before release
(2026-11-24; PyTorch 2.13.0, NumPy 2.5.2, scikit-learn 1.9.0):

| Check | Observed result |
| --- | --- |
| Official dataset acquisition and source integrity | All 28 files identical to legacy source files |
| Training/validation preprocessing | Exact numerical equality for 20 dataset/seed pairs |
| CPU regression suite | 11 passed |
| Single-GH200 Jannis pilot and resume (jobs 3387155, 3387431) | Three models completed; Pruned DPN resumed from epoch 115; isolated latency measured |
| Four-GH200 Covertype pilot (job 3387718) | Eight-worker limit per GPU; mean device loads 95.61–96.07%; 28 checkpointed and four additional launched conditions, no new completed models |
| Historical reporting | 20 pruning rows, 140 unique validation rows, 48 Welch comparisons, four validation figures and four pending test figures |
| Final-evaluation safeguard regression tests | Passed on synthetic data; actual held-out evaluation not performed |
| Notebook default dry run | Seven code cells passed; CSVs unchanged |

The four-GPU pilot was intentionally time-limited. It does not prove optimal
throughput, completion of the staged DPN, or successful full LTH-MLP pruning.
Fresh structural pruning, complete LTH pruning validation, dynamic matching
from fresh tickets, full 140-model reproduction and real held-out test
evaluation are deferred. CUDA execution was tested on GH200; other GPU models,
Apple Metal and cross-platform performance are not independently verified.
The released CSVs contain historical **validation** measurements. Test
accuracy plots are placeholders. The results must not be reported as held-out
performance or as a complete new-code reproduction. HIGGS Small is not HIGGS-500k.

## Publication (maintainers only)

Review the clean release snapshot locally with `bash scripts/publish_main.sh --preview`.
The allowlist includes the accepted pruning tickets, configurations, historical
CSVs, figures, source, tests and documentation; it excludes raw data, runtime
checkpoints, temporary files and freeze state. Historical paths in metadata are
flagged for review and not silently rewritten.

`bash scripts/publish_main.sh --review` clones the configured remote to a
temporary directory, checks the old main and existing backup branch, and prints
the staged diff without pushing. After reviewing that diff, confirming rights
to publish the files, and verifying Git authentication, a maintainer may run
`bash scripts/publish_main.sh --publish`. The script uses an atomic normal
Git push to preserve the previous main as `dpn1.0` and advance `main` together;
if the remote does not support atomic updates, it fails rather than falling
back to a partial push. No force-push is used. Existing root licensing and
citation files are retained if not replaced. Older Git history is preserved;
this does not purge any previously committed data from history.
