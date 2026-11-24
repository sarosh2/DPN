# Result CSVs

- `pruning.csv`: original per-seed pruning counts, sizes, timings (only when measured), and layouts.
- `training.csv`: one verified result per dataset, seed and model. Training and validation metrics, isolated inference latency where present, and provenance. Empty metrics have not been measured.
- `welch_tests.csv`: Welch confidence intervals and p-values from **per-seed** results, with Holm correction and paired diagnostics.
- `historical_validation_summary.csv`: author's **rounded five-seed validation** means, SDs and F1 for all seven regularized models. The Block DPN uses corrected Step 89 values. **Not** raw seed values.
- `historical_complexity_summary.csv`: author's rounded five-seed active/allocated parameters and uncontended latency.

The first two tables are populated in the release from verified legacy files. Example: `python -m src.statistics --repo . --split val --metric accuracy`.
