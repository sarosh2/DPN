# covtype

`dataset.json` defines the official split, preprocessing, maximal singleton DPN, model regularization, and training protocol. The dataset config is the **only required config**. The pipeline creates `<seed>/<model>.json` when the model dependency is ready, and saves the accepted pruned ticket as `<seed>/pruned_ticket.pt`.

Example: `python run.py --datasets covtype --seeds 42 --models block_matched_dpn`. This first resolves the pruned ticket and matched MLP when missing. It does not access the test split during training.
