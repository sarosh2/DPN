# Optional launchers and maintainer tools

- `import_legacy_tickets.py`: retained one-time migration utility for provenance. The released artifacts are already imported; reviewers do not need to run it. No runtime code imports it.
- `slurm_example.sbatch`: optional site-neutral Slurm wrapper. The regular `python run.py` command does not require Slurm.
- `publish_main.sh`: maintainer-only publication helper. Default `--preview` stages and verifies an allowlisted snapshot locally; `--review` inspects the remote and staged diff without pushing; `--publish` explicitly requests an atomic non-force push of the new main and its old-main backup `dpn1.0`.

Optional legacy migration example (requires actual original results):
`PYTHONPATH=. python scripts/import_legacy_tickets.py --previous-results /path/to/old/results/final --new-root .`
