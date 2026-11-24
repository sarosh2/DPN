# Dataset storage and automatic preparation

No dataset files are committed to Git. On first use, the four built-in
`configs/<dataset>/dataset.json` files instruct `src/data_io.py` to download
the official NeurIPS 2021 RTDL prepared-data archive. The downloader caches
that shared archive under `data/.archives/`, extracts only the requested
dataset, and checks row counts and feature dimensions. If an official split
is already present, all seven pinned file digests are checked before reuse,
without downloading anything.

The publisher supplies **train, validation, and test** splits in the archive.
We do not generate new random splits. The loader fits quantile or standard
normalization on the training split only, then transforms validation and
(optionally during frozen final evaluation) test. Test labels are not loaded
in normal training. The download writes `.source_provenance.json` with SHA256
hashes for each extracted split and the observed archive digest.

The built-in configs pin `data_acquisition.sha256` (archive) and
`data_acquisition.file_sha256` (all six split arrays and `info.json`). These
project-verified fingerprints matched the original benchmark files on Roihu
on 2026-11-24. They are not publisher-issued checksums and do not independently
prove publisher authenticity. A mismatch fails closed; move a corrupted dataset
directory or cached archive aside and rerun acquisition rather than editing the
pins. The cached archive is checked when extraction is needed, not when valid
prepared data is reused. Local provenance receipts are not trusted in place of
the configured pins. Downloading accepts component licenses as explained
at https://github.com/yandex-research/rtdl-revisiting-models.

Manual prepared NumPy arrays also work. Put `N_train.npy`, `N_val.npy`,
`N_test.npy`, `y_train.npy`, `y_val.npy`, `y_test.npy`, and `info.json` in
`data/<dataset>/`. They must match the pins for the built-in datasets.
Validation-only loading hashes raw test-file bytes for integrity but does not
deserialize test arrays, fit preprocessing on them or calculate test metrics.
Fresh extraction also checks all array headers and dimensions.

Example (first use downloads the configured official archive):

```python
from src.data_io import read_config, split_manifest
cfg = read_config('.', 'jannis')
print(split_manifest('.', cfg, include_test=False))
```

The first run needs internet access and enough disk space for the official
archive. If a provider blocks automated downloads, update the URL in the
same dataset config or place the original prepared split manually. The
program reports failures; it does not silently replace the dataset or split.
