# Model definitions

`dpn.py` is the preserved canonical DPN with `DPNAutoGrad`, cumulative connections, and masks. `dpn_flexible.py` adds non-affine LayerNorm, dropout, masks, and staged transitions. `resnet.py` contains the residual MLP. `mlp.py` contains the earlier numeric MLP; the final paper-shaped, one-logit-capable `PaperMLP` is in `src/model_factory.py`.

Example: `from src.dpn import DPN, FlexibleDPN`.
