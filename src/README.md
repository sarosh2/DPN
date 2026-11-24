# Source code

`dpn.py` exports the canonical original `DPN` and the regularized `FlexibleDPN`. `utils.py` contains the original outer and inner pruning code, plus MLP/ResMLP/DPN parameter matching and staged geometry builders. Do not modify the original pruning functions in a replication run.

`data_io.py` loads official prepared splits and fits transformations on training data only. `config_registry.py` resolves missing model configs in dependency order. `pruning.py` runs the original outer pruning wrapper with the binary-safe, pre-linear normalization training callback. `model_factory.py` constructs all seven model families. `training.py` trains with per-epoch checkpoints. `runner.py` schedules tasks on available GPUs. `worker.py` runs one task per process. `results.py` saves CSVs and figures. `statistics.py` computes Welch confidence intervals and p-values. `test_evaluation.py` performs opt-in sealed test evaluation.

Example: `python -m src.statistics --repo . --split val`.

`pruning_checkpoint.py` is the original required structural-pruning snapshot
module. The device-specific `.cuda()` call in `utils.apply_lth_pruning()`
was changed to `.to(device)`; the original pruning decisions are unchanged.
`runner.py` supports CUDA, Apple MPS, and CPU; Slurm is not a dependency.
