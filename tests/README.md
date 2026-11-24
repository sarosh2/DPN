# Tests

`test_geometry.py` checks the preserved outer pruning API, GPU-independent DPN masks, and the four benchmark depth/parameter matching policies. `test_pipeline.py` makes a small synthetic prepared dataset to test config creation, checkpoint reuse, regularized training, LTH masks, isolated latency, CSVs and figure generation without touching the real test splits.

Example: `python -m pytest -q`.
