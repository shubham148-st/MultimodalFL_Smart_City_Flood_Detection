"""
Heterogeneous Federated Learning (HeteroFL) Operations.

This module provides the core functions for model slicing and aggregation,
enabling clients with varying computational capacities to train sub-networks
extracted from a larger global model.
"""

import copy
import torch


def split_model(global_model, rate):
    """
    Slice global model weights down to a sub-network defined by the given rate.

    Extracts a contiguous subset of channels/dimensions from the global parameters
    to fit the computational constraints of a specific client tier. Handles 
    special exemptions for frozen pre-trained backbones and specific modality 
    projection constraints.

    Args:
        global_model (nn.Module): The full-capacity global PyTorch model.
        rate (float): The slicing rate (e.g., 0.5 for half-width, 1.0 for full).

    Returns:
        dict: A state dictionary containing the sliced model parameters.
    """
    if rate == 1.0:
        return copy.deepcopy(global_model.state_dict())

    global_params = global_model.state_dict()
    local_params = {}

    for name, param in global_params.items():
        # The entire ResNet feature extractor is frozen and shared, do not slice ANY of its params (weights, biases, or BN stats)
        if "cnn." in name:
            local_params[name] = param.clone()
            continue

        if 'weight' in name or 'bias' in name:

            # ── Fixed-input layers (don't slice input dim) ──
            if "time_mlp" in name:
                local_params[name] = param.clone()
                continue

            if "water_net.head" in name:
                if 'weight' in name:
                    in_dim = int(param.shape[1] * rate)
                    local_params[name] = param[:, :in_dim].clone()
                else:
                    local_params[name] = param.clone()
                continue

            if "water_proj" in name:
                if 'weight' in name:
                    out_dim = int(param.shape[0] * rate)
                    local_params[name] = param[:out_dim, :].clone()
                else:
                    out_dim = int(param.shape[0] * rate)
                    local_params[name] = param[:out_dim].clone()
                continue

            # spatial_attn.conv: always 2 input channels, slice output only
            if "spatial_attn.conv" in name:
                if 'weight' in name:
                    # shape [1, 2, k, k] — output is always 1 channel
                    local_params[name] = param.clone()
                else:
                    local_params[name] = param.clone()
                continue

            # img_proj: input always == 512 from fixed ResNet
            if "img_proj" in name:
                if 'weight' in name:
                    out_dim = int(param.shape[0] * rate)
                    local_params[name] = param[:out_dim, :].clone()
                else:
                    out_dim = int(param.shape[0] * rate)
                    local_params[name] = param[:out_dim].clone()
                continue

            # Conv Weights (4-D)
            if len(param.shape) == 4:
                out_ch = int(param.shape[0] * rate)
                in_ch  = int(param.shape[1] * rate)
                if param.shape[1] == 1:
                    in_ch = 1
                local_params[name] = param[:out_ch, :in_ch].clone()

            # Linear Weights (2-D)
            elif len(param.shape) == 2:
                out_dim = int(param.shape[0] * rate)
                in_dim  = int(param.shape[1] * rate)

                if "rain_net.proj.weight" in name:
                    in_dim = 1
                if "water_net.proj.weight" in name:
                    in_dim = 20
                # fc_hidden: input is fusion_dim (scales with rate), output is fusion_hid
                if "fc_hidden.weight" in name:
                    out_dim = int(param.shape[0] * rate)
                    in_dim  = int(param.shape[1] * rate)
                # classifier now takes fusion_hid input (scales), output always 1
                if "classifier.weight" in name:
                    out_dim = 1

                local_params[name] = param[:out_dim, :in_dim].clone()

            # Biases (1-D)
            elif len(param.shape) == 1:
                out_dim = int(param.shape[0] * rate)
                if "classifier.bias" in name:
                    out_dim = 1
                local_params[name] = param[:out_dim].clone()

            # Scalar parameters (0-D) — e.g., temperature
            elif len(param.shape) == 0:
                local_params[name] = param.clone()

        # BN running stats
        elif 'running_mean' in name or 'running_var' in name:
            out_dim = int(param.shape[0] * rate)
            local_params[name] = param[:out_dim].clone()
        else:
            local_params[name] = param.clone()

    return local_params



def aggregate_models(global_model, local_updates, client_rates):
    """
    Perform federated aggregation of sub-network updates into the global model.

    Since clients train sub-networks of varying sizes, this function tracks
    the frequency of updates per parameter index and performs an element-wise 
    average only across the clients that actually trained that specific parameter.

    Args:
        global_model (nn.Module): The current global model to be updated.
        local_updates (list of dict): List of local state_dicts from clients.
        client_rates (list of float): The corresponding model slice rates for each client.

    Returns:
        nn.Module: The updated global model.
    """
    global_state = global_model.state_dict()
    weight_sum = {k: torch.zeros_like(v, dtype=torch.float) for k, v in global_state.items()}
    count = {k: torch.zeros_like(v, dtype=torch.float) for k, v in global_state.items()}

    for local_state, rate in zip(local_updates, client_rates):
        for name, param in local_state.items():
            if name not in weight_sum:
                continue
            pf = param.float()

            if len(param.shape) == 4:
                o, i = param.shape[0], param.shape[1]
                weight_sum[name][:o, :i] += pf
                count[name][:o, :i] += 1
            elif len(param.shape) == 2:
                o, i = param.shape[0], param.shape[1]
                weight_sum[name][:o, :i] += pf
                count[name][:o, :i] += 1
            elif len(param.shape) == 1:
                o = param.shape[0]
                weight_sum[name][:o] += pf
                count[name][:o] += 1
            else:
                weight_sum[name] += pf
                count[name] += 1

    for name in global_state:
        mask = count[name] > 0
        if mask.any():
            averaged = weight_sum[name][mask] / count[name][mask]
            global_state[name][mask] = averaged.type(global_state[name].dtype)

    global_model.load_state_dict(global_state)
    return global_model
