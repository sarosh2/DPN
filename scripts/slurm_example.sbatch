#!/usr/bin/env bash
# Optional Slurm example. Choose --partition, --account, --gres, and --time
# through sbatch flags for YOUR site. This file does not select a GPU model.
#SBATCH --job-name=dpn-benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=dpn-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Works with an installed Python environment; no cluster-specific modules.
# Set DPN_DATASETS, DPN_SEEDS, DPN_MODELS for a custom selection.
read -r -a DATASETS <<< "${DPN_DATASETS:-covtype jannis helena higgs_small}"
read -r -a SEEDS <<< "${DPN_SEEDS:-42 77 53 31 18}"
read -r -a MODELS <<< "${DPN_MODELS:-matched_mlp matched_resnet lth_mlp block_matched_dpn staged_dpn}"

PRUNING_ARGS=()
if [[ "${DPN_NO_PRUNING:-0}" == "1" ]]; then PRUNING_ARGS=(--no-pruning); fi
TIME_ARGS=()
if [[ -n "${DPN_MAX_SECONDS:-}" ]]; then TIME_ARGS=(--max-seconds "$DPN_MAX_SECONDS"); fi
# SLURM_TIMELIMIT is a suggestion. For a strict limit, pass --max-seconds.
# The runner also stops before its own deadline and writes per-epoch checkpoints.
python -u run.py --datasets "${DATASETS[@]}" --seeds "${SEEDS[@]}" \
       --models "${MODELS[@]}" --device auto \
       --max-workers-per-gpu "${DPN_MAX_WORKERS:-16}" \
       "${PRUNING_ARGS[@]}" \
       "${TIME_ARGS[@]}"
