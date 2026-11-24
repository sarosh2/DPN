
import os

import random

from pathlib import Path



import numpy as np

import torch





def save_checkpoint(path, state):

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)



    payload = dict(state)

    payload["rng_state"] = {

        "python": random.getstate(),

        "numpy": np.random.get_state(),

        "torch": torch.get_rng_state(),

        "cuda": (

            torch.cuda.get_rng_state_all()

            if torch.cuda.is_available()

            else []

        ),

    }



    temporary = path.with_name(path.name + ".tmp")



    try:

        torch.save(payload, temporary)

        os.replace(temporary, path)

    finally:

        temporary.unlink(missing_ok=True)





def load_checkpoint(path):

    # Only load checkpoints produced by our own experiment.

    return torch.load(

        path,

        map_location="cpu",

        weights_only=False,

    )





def restore_rng(checkpoint):

    state = checkpoint["rng_state"]



    random.setstate(state["python"])

    np.random.set_state(state["numpy"])

    torch.set_rng_state(state["torch"])



    if torch.cuda.is_available() and state["cuda"]:

        torch.cuda.set_rng_state_all(state["cuda"])






def snapshot_dpn(model):

    """Capture a DPN with potentially changing weight-block shapes."""

    return {

        "weights": [w.detach().cpu().clone() for w in model.weights],

        "masks": [

            getattr(model, f"mask_{i}").detach().cpu().clone()

            for i in range(len(model.weights))

        ],

        "biases": model.biases.detach().cpu().clone(),

    }





def restore_dpn(model, snapshot):

    """Restore a structurally pruned DPN in place."""

    from torch import nn



    device = next(model.parameters()).device

    weights = snapshot["weights"]

    masks = snapshot["masks"]

    biases = snapshot["biases"]



    assert len(weights) == len(masks)



    total_nodes = sum(w.shape[0] for w in weights)

    assert biases.numel() == total_nodes



    expected_inputs = model.input_features

    for w, mask in zip(weights, masks):

        assert w.shape == mask.shape

        assert w.shape[1] == expected_inputs

        expected_inputs += w.shape[0]



    assert total_nodes >= model.output_nodes



    model.weights = nn.ParameterList([

        nn.Parameter(w.to(device).clone())

        for w in weights

    ])



    for name in list(model._buffers):

        if name.startswith("mask_"):

            delattr(model, name)



    for i, mask in enumerate(masks):

        model.register_buffer(

            f"mask_{i}",

            mask.to(device).clone()

        )



    model.biases = nn.Parameter(biases.to(device).clone())

    model.total_nodes = total_nodes



    return model

