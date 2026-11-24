# higgs_small, seed 31

Generated `*.config` files contain frozen, seed-specific architecture, initialization and regularization details. `pruning_metadata.json` and `accepted_pruned_ticket.pt` record the verified original outer-pruning winner. Delete neither during a checkpoint resume.

Example: `python run.py --datasets higgs_small --seeds 31 --models matched_mlp block_matched_dpn`.
