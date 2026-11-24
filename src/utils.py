import torch
import time
import numpy as np
import math
import copy
import torch.nn.utils.prune as prune
import torch.nn as nn
from src.models.dpn import DPN

def evaluate(model, dataloader, criterion, flatten, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.view(x_batch.size(0), -1).to(device) if flatten else x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            
            loss = criterion(outputs, y_batch)
            loss_sum += loss.item() * y_batch.size(0)
            
            preds = outputs.argmax(dim=1) if outputs.dim() > 1 and outputs.shape[1] > 1 else (outputs > 0.5).float()
            y_batch_labels = y_batch.argmax(dim=1) if y_batch.dim() > 1 and y_batch.shape[1] > 1 else y_batch
            correct += (preds == y_batch_labels).sum().item()
            total += y_batch.size(0)

        test_loss, test_acc = loss_sum / total, correct / total
    return test_loss, test_acc

def train(model, train_loader, val_loader, epochs, optimizer, criterion, flatten=True, device=None):
    train_metrics = np.empty((0, 3))  # list of (time_per_batch, loss, accuracy) per batch
    val_metrics = np.empty((0, 2))    # list of (val_loss, val_accuracy) per epoch

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        batch_metrics = np.empty((0, 3))

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.view(x_batch.size(0), -1).to(device) if flatten else x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            batch_start = time.time()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            batch_time = time.time() - batch_start
            
            running_loss += loss.item() * y_batch.size(0)
            preds = outputs.argmax(dim=1) if outputs.dim() > 1 and outputs.shape[1] > 1 else (outputs > 0.5).float()
            y_batch_labels = y_batch.argmax(dim=1) if y_batch.dim() > 1 and y_batch.shape[1] > 1 else y_batch
            correct += (preds == y_batch_labels).sum().item()
            total += y_batch.size(0)

            new_metrics = np.array([batch_time, loss.item(), (preds == y_batch_labels).float().mean().item()]).reshape(1, -1)
            batch_metrics = np.concatenate([batch_metrics, new_metrics])

        train_metrics = np.concatenate([train_metrics, batch_metrics.sum(axis=0, keepdims=True)])

        total_time = train_metrics[-1, 0]  # Sum of all batch times
        avg_time_per_batch = total_time / batch_metrics.shape[0]
        train_loss = running_loss / total
        train_acc = correct / total

        if val_loader:
            val_loss, val_acc = evaluate(model, val_loader, criterion, flatten, device)
            val_metrics = np.concatenate([val_metrics, np.array([val_loss, val_acc]).reshape(1, -1)])

        print(f"\nEpoch: {epoch} Total_Time: {total_time:.4f} Average_Time_per_batch: {avg_time_per_batch:.4f} "
            f"Train_Accuracy: {train_acc:.4f} Train_Loss: {train_loss:.4f} ", end='')
        
        if val_loader:
            print(f"Validation_Accuracy: {val_acc:.4f} Validation_Loss: {val_loss:.4f}", end='')

    return train_metrics, val_metrics

def prune_by_percent_once(percent, mask, final_weight, input_size):
    # Get the absolute values of weights where mask == 1
    masked_weights = final_weight[mask == 1].abs()

    # Sort the unmasked weights
    sorted_weights, _ = torch.sort(masked_weights)
    nonzero_idx = max(mask.size(1) - 1 - (mask.flip(1) != 0).int().argmax(dim=1)).item()
    if sorted_weights.shape[0] != 0 and nonzero_idx >= input_size:
        # Determine the cutoff index for pruning
        cutoff_index = min(int(round(percent * sorted_weights.shape[0])), sorted_weights.shape[0] - 1)
        cutoff = sorted_weights[cutoff_index]
        new_mask = torch.where(final_weight.abs() <= cutoff, torch.zeros_like(mask), mask)
        new_mask = torch.cat([mask[:, :nonzero_idx], new_mask[:, nonzero_idx:]], dim=1)

    else:
        new_mask = mask

    return new_mask

def prune_by_percent_once_param(percent, mask, final_weight):
   # Get the absolute values of weights where mask == 1
    masked_weights = final_weight[mask == 1].abs()

    # Sort the unmasked weights
    sorted_weights, _ = torch.sort(masked_weights)
    if sorted_weights.shape[0] != 0:
        # Determine the cutoff index for pruning
        cutoff_index = min(int(round(percent * sorted_weights.shape[0])), sorted_weights.shape[0] - 1)
        cutoff = sorted_weights[cutoff_index]
        new_mask = torch.where(final_weight.abs() <= cutoff, torch.zeros_like(mask), mask)

    else:
        new_mask = mask

    return new_mask

def prune_by_percent(model, masks, percent):

    blocks = model.layers
    input_size = masks[0].shape[1]
    for i in range(len(blocks)):
        mask = []
        for j in range(masks[i].shape[0]):
            mask.append(prune_by_percent_once(percent, masks[i][j: j + 1, :], blocks[i].weight[j: j + 1, :], input_size))
        masks[i] = torch.cat(mask, dim = 0)

    return masks

def prune_by_percent_param(model, masks, percent):

    blocks = model.layers
    for i in range(len(blocks)):
        masks[i] = prune_by_percent_once_param(percent, masks[i], blocks[i].weight)

    return masks

def rebuild_with_masks(model, weights, masks, device):
    new_layers = torch.nn.ModuleList()

    for w, mask in zip(weights, masks):
        layer = torch.nn.Linear(
            w.shape[1],
            w.shape[0]
        ).to(device)

        # Register PyTorch's internal pruning machinery:
        # weight = weight_orig * weight_mask
        prune.custom_from_mask(
            layer,
            name="weight",
            mask=mask.to(device)
        )

        # Put your desired values into weight_orig
        with torch.no_grad():
            layer.weight_orig.copy_(w.to(device))

        new_layers.append(layer)

    model.layers = new_layers

def apply_lth_pruning(model, train_loader, val_loader, epochs, optimizer, criterion, percent, rounds, checkpoint_path=None):

    device = next(model.parameters()).device
    step = rounds // 10



    from pathlib import Path



    def pass_checkpoint(n):

        if checkpoint_path is None:

            return None

        return Path(checkpoint_path).with_name(

            f"pruning_rounds_{n}.pt"

        )


    #save initial model
    initial_weights = [w.detach().clone() for w in model.weights]
    initial_biases = model.biases.detach().clone()

    hidden_dims = [weight.shape[0] for weight in model.weights]

    # Initial training is unnecessary when resuming a saved pass.

    first_checkpoint = pass_checkpoint(rounds)



    if first_checkpoint is not None and first_checkpoint.is_file():

        initial_val_acc = 0.0

        # The inner pruner restores the actual saved accuracy.

        print("[pruning-resume] Skipping initial validation training", flush=True)

    else:

        print("Initial Validation Phase", end="")

        _, val_metrics = train(

            model, train_loader, val_loader,

            epochs, optimizer, criterion

        )

        initial_val_acc = val_metrics[-1][1]



    #reset model
    model.weights = nn.ParameterList([nn.Parameter(w.to(device)) for w in initial_weights])
    with torch.no_grad():
        model.biases.copy_(initial_biases)

    trial_model = model
    optimizer = type(optimizer)(trial_model.parameters(), **optimizer.defaults)
    
    #prune once
    trial_optimizer, final_round, prune_percent, final_val_acc = apply_lth_pruning_param(trial_model, train_loader, val_loader, epochs, optimizer, criterion, percent, rounds, val_accuracy = initial_val_acc, checkpoint_path=pass_checkpoint(rounds))

    hidden_dims_pruned = [weight.shape[0] for weight in trial_model.weights]

    #prune in loops
    while hidden_dims != hidden_dims_pruned and rounds > step:

        rounds -= step
        #percent *= 1.1
        model = trial_model
        optimizer = trial_optimizer
        hidden_dims = hidden_dims_pruned
        
        trial_model = DPN(model.input_features, sum(hidden_dims_pruned), model.output_nodes, minimal=False, hidden_dims = hidden_dims_pruned).to(device)
        trial_optimizer = type(optimizer)(trial_model.parameters(),**optimizer.defaults)
        trial_optimizer, final_round, prune_percent, final_val_acc = apply_lth_pruning_param(trial_model, train_loader, val_loader, epochs, trial_optimizer, criterion, percent, rounds, val_accuracy = final_val_acc * 0.99, checkpoint_path=pass_checkpoint(rounds))
        hidden_dims_pruned = [weight.shape[0] for weight in trial_model.weights]

    return trial_model, trial_optimizer, final_round, prune_percent, final_val_acc

def apply_lth_pruning_param(model, train_loader, val_loader, epochs, optimizer, criterion, percent, rounds, val_accuracy = None, end_thresh = 0.90, checkpoint_path=None):

    from pathlib import Path

    device = next(model.parameters()).device

    # --------------------------------------------------
    # Save original lottery-ticket initialization
    # --------------------------------------------------
    initial_weights = [
        w.detach().clone()
        for w in model.weights
    ]

    initial_biases = model.biases.detach().clone()

    # Fraction of remaining weights removed each round
    p_per_round = (1 - (1 - percent) ** (1 / rounds))

    # --------------------------------------------------
    # Initial training
    # --------------------------------------------------

    if val_accuracy == None:
    
        print("Initial Validation Phase", end="")
    
        _, val_metrics = train(
            model,
            train_loader,
            val_loader,
            epochs,
            optimizer,
            criterion
        )
    
        val_accuracy = val_metrics[-1][1]

    final_val_accuracy = val_accuracy
    final_round = 0

    # Best accepted ticket
    final_weights = [
        w.detach().clone()
        for w in initial_weights
    ]

    final_masks = [
        getattr(model, f"mask_{i}").detach().clone()
        for i in range(len(model.weights))
    ]

    # ==================================================
    # Pruning rounds
    # ==================================================


    start_round = 0



    if checkpoint_path is not None and Path(checkpoint_path).is_file():

        from src.pruning_checkpoint import load_checkpoint, restore_dpn, restore_rng



        checkpoint = load_checkpoint(checkpoint_path)



        if checkpoint.get("phase") != "inner_pruning":

            raise RuntimeError("Unexpected pruning checkpoint phase")



        if (

            checkpoint["rounds"] != rounds

            or abs(checkpoint["percent"] - percent) > 1e-12

            or abs(checkpoint["end_thresh"] - end_thresh) > 1e-12

        ):

            raise RuntimeError("Pruning checkpoint configuration mismatch")



        restore_dpn(model, checkpoint["current_model"])



        initial_weights = [

            w.to(device).clone() for w in checkpoint["initial_weights"]

        ]

        initial_biases = checkpoint["initial_biases"].to(device).clone()



        final_weights = [

            w.to(device).clone() for w in checkpoint["final_weights"]

        ]

        final_masks = [

            m.to(device).clone() for m in checkpoint["final_masks"]

        ]



        val_accuracy = checkpoint["val_accuracy"]

        final_val_accuracy = checkpoint["final_val_accuracy"]

        final_round = checkpoint["final_round"]



        start_round = checkpoint["next_round"]

        if not 0 <= start_round <= rounds:

            raise RuntimeError("Invalid saved pruning round")



        if checkpoint.get("stop_requested", False):

            start_round = rounds



        restore_rng(checkpoint)



        print(

            f"[pruning-resume] Restored checkpoint; "

            f"next round={start_round}/{rounds}",

            flush=True

        )



    for round_idx in range(start_round, rounds):



        print(
            "\n\nPruning Round:",
            round_idx + 1,
            end=""
        )

        # --------------------------------------------------
        # Global L1 pruning over currently active weights
        # --------------------------------------------------
        active_values = []

        for i, w in enumerate(model.weights):

            mask = getattr(
                model,
                f"mask_{i}"
            )

            active_values.append(
                w.detach()
                 .abs()[mask.bool()]
                 .flatten()
            )

        active_values = torch.cat(active_values)

        n_prune = int(
            active_values.numel() * p_per_round
        )

        if n_prune > 0:

            threshold = torch.kthvalue(
                active_values,
                n_prune
            ).values

            with torch.no_grad():

                for i, w in enumerate(model.weights):

                    mask = getattr(
                        model,
                        f"mask_{i}"
                    )

                    new_mask = (
                        mask.bool()
                        & (w.abs() > threshold)
                    )

                    mask.copy_(new_mask)

        # --------------------------------------------------
        # Collect current masks
        # --------------------------------------------------
        current_masks = [
            getattr(model, f"mask_{i}")
            .detach()
            .clone()
            for i in range(len(model.weights))
        ]

        # --------------------------------------------------
        # Apply masks to current trained weights
        # --------------------------------------------------
        pruned_weights = [
            w.detach().clone() * m
            for w, m in zip(
                model.weights,
                current_masks
            )
        ]

        # --------------------------------------------------
        # Apply same masks to lottery-ticket initialization
        # --------------------------------------------------
        pruned_initial_weights = [
            w * m
            for w, m in zip(
                initial_weights,
                current_masks
            )
        ]

        # --------------------------------------------------
        # Merge current weights, masks, and initialization
        # --------------------------------------------------
        merged_weights = merge_weights(
            [w.clone() for w in pruned_weights],
            model.output_nodes
        )

        merged_masks = merge_weights(
            [m.clone() for m in current_masks],
            model.output_nodes
        )

        merged_initial_weights = merge_weights(
            [w.clone() for w in pruned_initial_weights],
            model.output_nodes
        )

        # Number of masks before changing structure
        old_num_weights = len(model.weights)

        # --------------------------------------------------
        # Replace model weight structure
        # --------------------------------------------------
        model.weights = nn.ParameterList([
            nn.Parameter(w.to(device))
            for w in merged_weights
        ])

        # Remove old mask buffers
        for i in range(old_num_weights):
            name = f"mask_{i}"

            if hasattr(model, name):
                delattr(model, name)

        # Register merged masks
        for i, mask in enumerate(merged_masks):

            model.register_buffer(
                f"mask_{i}",
                mask.to(device)
            )

        # --------------------------------------------------
        # Initialization must follow new merged structure
        # --------------------------------------------------
        initial_weights = [
            w.detach().clone().to(device)
            for w in merged_initial_weights
        ]

        dims = [m.shape[0] for m in merged_weights]
        print(" ", len(dims), dims, p_per_round)

        # --------------------------------------------------
        # Lottery-ticket rewind
        # --------------------------------------------------
        with torch.no_grad():

            for i, w in enumerate(model.weights):
                w.copy_(initial_weights[i])

            model.biases.copy_(
                initial_biases
            )

        # Model parameters changed -> new optimizer
        optimizer = type(optimizer)(
            model.parameters(),
            **optimizer.defaults
        )

        # --------------------------------------------------
        # Train merged/pruned ticket
        # --------------------------------------------------
        _, val_metrics = train(
            model,
            train_loader,
            val_loader,
            epochs,
            optimizer,
            criterion
        )

        new_val_accuracy = val_metrics[-1][1]

        # --------------------------------------------------
        # Accept ticket
        # --------------------------------------------------
        should_stop = False
        if new_val_accuracy >= val_accuracy:

            final_round = round_idx + 1

            final_val_accuracy = (
                new_val_accuracy
            )

            val_accuracy = (
                new_val_accuracy
            )

            # Save initialization in the CURRENT structure
            final_weights = [
                w.detach().clone()
                for w in initial_weights
            ]

            final_masks = [
                getattr(
                    model,
                    f"mask_{i}"
                ).detach().clone()
                for i in range(
                    len(model.weights)
                )
            ]

        # --------------------------------------------------
        # Stop if degradation > 2%
        # --------------------------------------------------
        elif new_val_accuracy < val_accuracy * end_thresh:
            should_stop = True



        if checkpoint_path is not None:

            from src.pruning_checkpoint import save_checkpoint, snapshot_dpn

            save_checkpoint(checkpoint_path, {

                "version": 1,

                "phase": "inner_pruning",

                "rounds": rounds,

                "next_round": round_idx + 1,
                "stop_requested": should_stop,

                "percent": percent,

                "end_thresh": end_thresh,

                "val_accuracy": val_accuracy,

                "final_val_accuracy": final_val_accuracy,

                "final_round": final_round,

                "current_model": snapshot_dpn(model),

                "initial_weights": [w.detach().cpu().clone() for w in initial_weights],

                "initial_biases": initial_biases.detach().cpu().clone(),

                "final_weights": [w.detach().cpu().clone() for w in final_weights],

                "final_masks": [m.detach().cpu().clone() for m in final_masks],

            })



        if should_stop:

            break

    # ==================================================
    # Restore best accepted ticket
    # ==================================================

    old_num_weights = len(model.weights)

    model.weights = nn.ParameterList([
        nn.Parameter(w.to(device))
        for w in final_weights
    ])

    # Remove current masks
    for i in range(old_num_weights):

        name = f"mask_{i}"

        if hasattr(model, name):
            delattr(model, name)

    # Restore best masks
    for i, mask in enumerate(final_masks):

        model.register_buffer(
            f"mask_{i}",
            mask.to(device)
        )

    with torch.no_grad():

        model.biases.copy_(
            initial_biases
        )

    # --------------------------------------------------
    # Fresh optimizer for final ticket
    # --------------------------------------------------
    optimizer = type(optimizer)(
        model.parameters(),
        **optimizer.defaults
    )

    # --------------------------------------------------
    # Actual sparsity of final ticket
    # --------------------------------------------------
    total_weights = sum(
        mask.numel()
        for mask in final_masks
    )

    active_weights = sum(
        mask.count_nonzero().item()
        for mask in final_masks
    )

    actual_pruning = (
        1 - active_weights / total_weights
    )

    return (
        optimizer,
        final_round,
        actual_pruning,
        final_val_accuracy
    )

def apply_lth_pruning_base(model, train_loader, val_loader, epochs, optimizer, criterion, percent, rounds):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initial_state = copy.deepcopy(model.state_dict())
    p_per_round = 1 - (1 - percent) ** (1 / rounds)
    final_round = 0
    final_model = initial_state

    print("Initial Validation Phase", end='')
    _, val_metrics = train(model, train_loader, val_loader, epochs, optimizer, criterion)
    val_accuracy = val_metrics[-1][1]
    final_val_accuracy = val_accuracy

    parameters_to_prune = tuple((layer, "weight") for layer in model.layers)

    for round_idx in range(rounds):
        print("\n\nPruning Round:", round_idx + 1, end='')
                
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=p_per_round)

        with torch.no_grad():
            for i, layer in enumerate(model.layers):
                weight_key = (f"layers.{i}.weight_orig" if f"layers.{i}.weight_orig" in initial_state else f"layers.{i}.weight")
        
                layer.weight_orig.copy_(initial_state[weight_key].to(device))
                layer.bias.copy_(initial_state[f"layers.{i}.bias"].to(device))
                
        optimizer = type(optimizer)(model.parameters(), **optimizer.defaults)
        ticket_state = copy.deepcopy(model.state_dict())

        _, val_metrics = train(model, train_loader, val_loader, epochs, optimizer, criterion)
        new_val_accuracy = val_metrics[-1][1]

        if new_val_accuracy >= val_accuracy:
            final_model = copy.deepcopy(ticket_state)
            final_round = round_idx + 1
            final_val_accuracy = new_val_accuracy
            val_accuracy = new_val_accuracy
        elif new_val_accuracy < val_accuracy - (0.02 * val_accuracy):
            break

    if final_round == 0:
        # No pruned ticket was accepted.
        # Convert the currently-pruned model back to ordinary Linear weights.
        for layer in model.layers:
            if hasattr(layer, "weight_orig"):
                prune.remove(layer, "weight")
    
        model.load_state_dict(initial_state)
    
    else:
        # final_model is a pruned ticket and matches the
        # current weight_orig / weight_mask structure.
        model.load_state_dict(final_model)
        
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            weight_key = (f"layers.{i}.weight_orig" if f"layers.{i}.weight_orig" in initial_state else f"layers.{i}.weight")

            if f"layers.{i}.weight_orig" in initial_state:
                layer.weight_orig.copy_(initial_state[weight_key].to(device))
            else:
                layer.weight.copy_(initial_state[weight_key].to(device))
            layer.bias.copy_(initial_state[f"layers.{i}.bias"].to(device))
    optimizer = type(optimizer)(model.parameters(), **optimizer.defaults)
    
    return model, optimizer, final_round, 1 - (1 - p_per_round) ** (final_round), final_val_accuracy

def last_nonzero_idx(row):
    # row shape: [1, n]
    nz = torch.nonzero(row != 0, as_tuple=False)
    if nz.numel() == 0:
        return -1
    return nz[:, 1].max().item()

def merge_weights(weights, num_classes):

    if not weights:
        return weights

    merged_weights = [weights[0]]
    current_node = weights[0].shape[0]
    total_nodes = sum([w.shape[0] for w in weights])
    
    for i in range(1, len(weights)):
        W = weights[i]
        for j in range(W.shape[0]):
            row = W[j : j + 1, :]
            nonzero_idx = last_nonzero_idx(row)
            target_shape = merged_weights[-1].shape[1]

            if current_node == total_nodes - num_classes:
                merged_weights.append(row)
                
            elif nonzero_idx < target_shape:
                merged_weights[-1] = torch.cat([merged_weights[-1], row[:, :target_shape]], dim=0)
                W = torch.cat([W, torch.zeros((weights[i].shape[0], 1), dtype=W.dtype, device=W.device)], dim = 1)
                
            else:
                merged_weights.append(row)
            current_node += 1
            
    return merged_weights

def clean_weights(weights, size_limit):
    zero_indices = [i + size_limit for i, t in enumerate(weights) if torch.all(t == 0)]
    print("Empty Blocks: ", zero_indices)


    while len(zero_indices) > 0:
        weights = [t for t in weights if not torch.all(t == 0)]

        for i in range(len(weights)):
            tensor = weights[i]
            columns_to_keep = [i for i in range(tensor.shape[1]) if i not in zero_indices]
            weights[i] = tensor[:, columns_to_keep]

        zero_indices = [i + size_limit for i, t in enumerate(weights) if torch.all(t == 0)]
        print("Empty Blocks: ", zero_indices)

    return weights

def count_parameters(model):
    total = 0

    for layer in model.layers:
        # Weights
        if hasattr(layer, "weight_mask"):
            total += layer.weight_mask.count_nonzero().item()
        elif hasattr(layer, "weight"):
            total += layer.weight.numel()
    
        # Biases
        if getattr(layer, "bias", None) is not None:
            total += layer.bias.numel()
    
    return total

# ---------------------------------------------------------------------------
# DPN Paper 2: model geometry and dependency utilities (reproducible builders).
# The original apply_lth_pruning/apply_lth_pruning_param above are unchanged.
# ---------------------------------------------------------------------------

def allocated_parameters(input_features, output_logits, widths):
    """MLP allocated parameters, including every bias and prediction head."""
    dimensions = [int(input_features), *(int(x) for x in widths), int(output_logits)]
    if min(dimensions) < 1:
        raise ValueError('Dimensions must be positive')
    return sum((a + 1) * b for a, b in zip(dimensions, dimensions[1:]))


def dpn_allocated_parameters(input_features, output_logits, hidden_widths,
                             output_widths=None):
    """Exact cumulative-concatenation capacity, not a paper-budget proxy."""
    widths = [*map(int, hidden_widths),
              *map(int, output_widths if output_widths is not None else [output_logits])]
    if not widths or min(widths) < 1 or sum(widths[-len(output_widths or [output_logits]):]) != output_logits:
        raise ValueError('Invalid DPN widths')
    current, result = int(input_features), 0
    for width in widths:
        result += (current + 1) * width
        current += width
    return result


def singleton_capacity(features, hidden, output):
    nodes = int(hidden) + int(output)
    return nodes * (int(features) + 1) + nodes * (nodes - 1) // 2


def match_mlp_to_ticket(features, output, paper_widths, target, tolerance=.01):
    """Scaled paper-shaped MLP; search integers, never substitute paper count."""
    shape = tuple(int(w) for w in paper_widths)
    if not shape or min(shape) < 1 or target < 1:
        raise ValueError('Paper-shaped MLP and actual ticket budget required')
    candidates = []
    for k in range(100, 3501):
        widths = tuple(max(1, round(w * k / 1000)) for w in shape)
        p = allocated_parameters(features, output, widths)
        candidates.append((abs(p - target), p, widths))
    _, _, best = min(candidates)
    for _ in range(3):
        candidates = []
        for i in range(len(best)):
            for delta in range(-15, 16):
                w = list(best)
                w[i] = max(1, w[i] + delta)
                p = allocated_parameters(features, output, w)
                candidates.append((abs(p - target), p, tuple(w)))
        candidates.append((abs(allocated_parameters(features, output, best)-target),
                           allocated_parameters(features, output, best), best))
        _, _, best = min(candidates)
    got = allocated_parameters(features, output, best)
    # Small and highly asymmetric datasets need a wider integer search than
    # the paper-proportional local search. Refine two coordinates analytically
    # rather than silently accept an over-budget architecture.
    if abs(got-target)/target > tolerance:
        proposals=[]
        for i in range(len(best)):
            for j in range(len(best)):
                if i==j:continue
                for wi in range(1,min(600,max(best[i]*4,40))):
                    trial=list(best);trial[i]=wi
                    p0=allocated_parameters(features,output,trial)
                    slope=allocated_parameters(features,output,[*(trial[:j]),trial[j]+1,*(trial[j+1:])])-p0
                    wj=max(1,trial[j]+round((target-p0)/slope))
                    trial[j]=wj
                    p=allocated_parameters(features,output,trial)
                    proposals.append((abs(p-target),p,tuple(trial)))
        if proposals:
            _,got,best=min(proposals)
    if abs(got - target) / target > tolerance:
        raise RuntimeError(f'Paper-shaped MLP cannot match actual budget: {got}/{target}')
    return {'hidden_widths': list(best), 'target_allocated': int(target),
            'actual_allocated': int(got), 'relative_error': abs(got-target)/target}


def match_shallow_block_dpn(features, output, mlp_widths, mlp_allocation, tolerance=.005):
    """Correct Step 89 control: exactly the MLP hidden depth AND allocation.

    The output logits are one full-width linear block, not singleton output
    blocks. We optimize integer hidden widths proportional to MLP widths.
    """
    shape = tuple(int(x) for x in mlp_widths)
    if allocated_parameters(features, output, shape) != int(mlp_allocation):
        raise ValueError('Seed-specific MLP shape does not match allocation')
    found = {}
    def consider(widths):
        widths = tuple(int(x) for x in widths)
        if min(widths) < 1 or widths in found:
            return
        p = dpn_allocated_parameters(features, output, widths)
        distortion = sum(((x / max(1, y)) ** .5 - 1) ** 2 for x,y in zip(widths,shape))
        found[widths] = (abs(p - mlp_allocation)/mlp_allocation,
                         distortion, abs(p-mlp_allocation), widths, p)
    for k in range(40,2501):
        s = k/1000
        w = tuple(max(1, int(round(x*s))) for x in shape)
        p = dpn_allocated_parameters(features, output, w)
        if abs(p-mlp_allocation) > mlp_allocation*.055:
            continue
        consider(w)
        for i in range(len(w)):
            coeff = features + 1 + output + sum(w)-w[i]
            delta = round((mlp_allocation-p)/coeff)
            for step in (-2,-1,0,1,2):
                t=list(w);t[i]+=delta+step
                if t[i]>0: consider(t)
    for _,_,_,base,_ in sorted(found.values())[:16]:
        for j in range(len(base)):
            for delta in range(-30,31):
                v=list(base);v[j]+=delta
                if v[j]<1:continue
                for k in range(len(v)):
                    if k==j:continue
                    p=dpn_allocated_parameters(features,output,v)
                    coeff=features+1+output+sum(v)-v[k]
                    u=v.copy();u[k]+=round((mlp_allocation-p)/coeff)
                    if u[k]>0:consider(u)
    if not found:
        raise RuntimeError('No feasible shallow-DPN width proposal')
    rel, _, _, widths, p = min(found.values())
    if rel > tolerance:
        raise RuntimeError(f'DPN allocation tolerance {tolerance} unmet: {rel:.6f}')
    return {'hidden_widths': list(widths), 'output_widths': [int(output)],
            'hidden_blocks': len(widths), 'total_sequential_blocks': len(widths)+1,
            'target_allocated': int(mlp_allocation), 'actual_allocated': p,
            'relative_error': rel, 'mlp_hidden_widths':list(shape)}


def match_maximal_singleton_dpn(features, output, target, tolerance=.01):
    lo, hi = 1, max(100, int((2*target)**.5)*2+10)
    while lo < hi:
        mid = (lo+hi)//2
        if singleton_capacity(features, mid, output) < target:lo=mid+1
        else:hi=mid
    n = min((max(1,lo-1),lo),key=lambda x:(abs(singleton_capacity(features,x,output)-target),x))
    p=singleton_capacity(features,n,output)
    if abs(p-target)/target>tolerance:
        raise RuntimeError('Cannot singleton-match ticket within tolerance')
    return {'hidden_widths':[1]*n,'output_widths':[1]*output,
            'actual_allocated':p,'target_allocated':target,
            'relative_error':abs(p-target)/target}


def staged_dpn_candidates(features, output, target, tolerance=.01):
    """Original six-hidden-block 3+3 compression with/without input bypass.

    Both share the six hidden widths and initial Stage-1 initialization.
    Caller trains both, then picks a validation winner.
    """
    d, O = int(features), int(output)
    def size(w,t,bypass):
        return (3*w*(d+1)+3*w*w+t*(d+3*w+1)
                +3*w*(t+(d if bypass else 0)+1)+3*w*w
                +O*(t+(d if bypass else 0)+3*w+1))
    best=None
    maxw=min(3000,max(2,int((target/6)**.5)*2+3))
    for w in range(1,maxw+1):
        step=d+6*w+O+1
        pair=[]
        for bypass in (False,True):
            at_one=size(w,1,bypass)
            t0=1+round((target-at_one)/step)
            choices=[(abs(size(w,t,bypass)-target),abs(t-(d+3*w)//2),t,size(w,t,bypass))
                     for t in {1,d+3*w-1,t0-2,t0-1,t0,t0+1,t0+2} if 1<=t<d+3*w]
            if not choices:pair=[];break
            pair.append(min(choices))
        if len(pair)!=2:continue
        rank=(max(p[0] for p in pair),sum(p[0] for p in pair),sum(p[1] for p in pair),w)
        if best is None or rank<best[0]:best=(rank,w,pair)
    out={}
    for idx,bypass in enumerate((False,True)):
        name='staged_input_skip' if bypass else 'staged_compressed'
        if best is None or best[2][idx][0]/target>tolerance:
            out[name]={'status':'infeasible','reason':'Six-stage pair cannot match allocation'}
            continue
        _,w, pair=best
        error,_,t,p=pair[idx]
        out[name]={'status':'candidate','hidden_widths':[w]*6,
                   'output_widths':[O],'stage_boundaries':[3],
                   'transition_widths':[t],'raw_input_bypass':bypass,
                   'target_allocated':target,'actual_allocated':p,
                   'relative_error':error/target}
    return out


def match_resnet_to_ticket(features, output, target, tolerance=.01):
    """Search ResMLP width/depth with full BN and linear parameter accounting."""
    best=None
    for depth in (1,2,3,4):
        for activation in ('relu','reglu'):
            for factor in (0.5,1.,2.):
                for d in range(4,512):
                    h=int(d*factor)
                    if h<1:continue
                    gate=2 if activation.endswith('glu') else 1
                    p=((features+1)*d + depth*(2*d+(d+1)*(h*gate)+(h+1)*d)
                       +2*d+(d+1)*output)
                    rank=(abs(p-target),depth,d,activation,factor,p)
                    if best is None or rank<best:best=rank
    _,depth,d,activation,factor,p=best
    if abs(p-target)/target>tolerance:
        raise RuntimeError('ResNet cannot parameter match selected ticket')
    return {'d':d,'n_layers':depth,'d_hidden_factor':factor,
            'activation':activation,'normalization':'batchnorm',
            'actual_allocated':p,'target_allocated':target,
            'relative_error':abs(p-target)/target}
