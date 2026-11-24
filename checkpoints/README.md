# Restart-safe checkpoints

The runner creates `checkpoints/<dataset>/seed_<seed>/<model>/latest.pt`, `best.pt`, and a completion marker. It writes files atomically. The fingerprint includes the dataset config, model config, data manifest, source implementation files, and training conditions. The runner resumes a checkpoint **only if the fingerprint matches**. A mismatched checkpoint causes a hard error; use a separate run root or archive it before changing protocol settings. The pruning routine has its original `pruning_rounds_*.pt` intermediate snapshots, and the final pruning ticket is saved under the seed's config directory and referenced by hash.

Do not commit ordinary training checkpoints or private datasets to Git. The
20 accepted discovery tickets under `configs/` are intentionally published;
these contain the accepted DPN state and masks, not final trained checkpoints.
