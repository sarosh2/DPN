# Results

`csvs/pruning.csv` and `csvs/training.csv` contain per-seed observations. `csvs/welch_tests.csv` stores Welch comparisons when requested. `images/` contains test-accuracy boxplots derived strictly from the training CSV. The figures show an explicit pending state before held-out test evaluation.

Example: `python run.py --plots-only`.

The release includes both the rounded historical summaries and the 140
imported per-seed validation records. Available seed-level timings are from
the original Roihu runs, not new-code measurements. Validation figures must
not be described as test results. Retraining can replace rows; keep this
release as a historical snapshot and use a separate copy for new runs.
