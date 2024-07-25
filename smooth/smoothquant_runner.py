import functools
from typing import List, Optional, Union

import torch
from loguru import logger
from torch import nn
from tqdm import tqdm

from utils import (
    clear_memory,
    get_layers_to_quantize,
    get_lm_head_to_quantize,
    get_model_inputs,
    get_module_outputs,
    get_named_linears,
    set_op_by_name,
    compute_quant_errors,
)
from smoothquant_grid import SmoothQuantizer, get_static_activation_scales
from quantizer import Quantizer

@torch.inference_mode()
def run_smooth_quantize(
    model: nn.Module,
    calib_dataloader: List[torch.Tensor],
    *,
    seq_len: int,
    exclude_layers: Optional[List[str]] = None,
    device: Union[str, torch.device] = "cuda",
    disable_greedy_search: bool = False,
):
    if exclude_layers is None:
        exclude_layers = []
    layers = get_layers_to_quantize(model)
    dtype = next(iter(model.parameters())).dtype
    n_samples = len(calib_dataloader)

    all_inps = torch.zeros(
        (n_samples, seq_len, model.config.hidden_size), dtype=dtype, device=device
    )
    is_seq_first = False

    layer_kwargs = get_model_inputs(model, calib_dataloader, all_inps=all_inps, device=device)

    error_list = []
    mean_scale_list = []
    for layer_idx, layer in enumerate(
        (pbar := tqdm(layers, desc="Running SmoothQuant..."))
    ):
        layer = layer.to(device)
        sq_quantizer = SmoothQuantizer(
            layer=layer, disable_greedy_search=disable_greedy_search, is_seq_first=is_seq_first
        )

        linears_to_quantize = get_named_linears(layer, exclude_layers)

        # Step1: Run SmoothQuant
        input_feat, all_outs = sq_quantizer.get_input_feat(
            linears_to_quantize, all_inps, layer_kwargs
        )
        clear_memory()

        sq_quantizer.search_and_apply_scale(input_feat, layer_kwargs)
        clear_memory()
        layer.cpu()
        layers[layer_idx] = layer
        torch.cuda.empty_cache()

    #     # Step2: Get smoothed-activation per-tensor scales for on-the-fly quantization at runtime
    #     layer = layer.to(device)
    #     mean_scale, decoder_layer_scales = get_static_activation_scales(
    #         8, layer, linears_to_quantize, all_inps, layer_kwargs
    #     )
    #     mean_scale_list.append(mean_scale)

    #     # Step3: Get weight channel-wise quantized scales
    #     for name, m in linears_to_quantize.items():
    #         assert isinstance(m, nn.Linear)
    #         m.quantizer = Quantizer(
    #             n_bit=8, per_tensor=False, group_size=-1, symm_q=True, zeropoint=False
    #         )
    #         w8a8_qlinear = W8A8QuantLinear.from_linear(
    #             m,
    #             name=name,
    #             input_scale=decoder_layer_scales[f"{name}_input"],
    #             operators=OperatorsPluginRegistry.get("cuda").create(),
    #         )
    #         set_op_by_name(layer, name, w8a8_qlinear)

    #     # Step4: Stat activation mean_scale and quantized layer error
    #     layer = layer.to(device)

    #     err_batch = min(32, n_samples)
    #     if sq_quantizer.is_seq_first:
    #         q_output = get_module_outputs(
    #             layer, all_inps[:, :err_batch], kwargs=layer_kwargs, is_seq_first=True
    #         )
    #         max_abs_error, max_rel_error = compute_quant_errors(
    #             all_outs[:, :err_batch], q_output, block_size=4
    #         )
    #     else:
    #         q_output = get_module_outputs(layer, all_inps[:err_batch], kwargs=layer_kwargs)
    #         max_abs_error, max_rel_error = compute_quant_errors(
    #             all_outs[:err_batch], q_output, block_size=4
    #         )
    #     error_list.append((max_abs_error, max_rel_error))

    #     pbar.set_description(
    #             f"Quant FwdErr: Abs={max_abs_error:.3f}, Rel={max_rel_error:.3%}, "
    #             f"for layer_idx {layer_idx}"
    #         )

    #     layer.cpu()
    #     layers[layer_idx] = layer
    #     torch.cuda.empty_cache()

    #     # Swap for next layer inputs
    #     all_inps.data.copy_(all_outs.data)

    # mean_abs_error = 0
    # mean_rel_error = 0
    # for abs_error, rel_error in error_list:
    #     mean_abs_error += abs_error
    #     mean_rel_error += rel_error
    # mean_abs_error /= len(error_list)
    # mean_rel_error /= len(error_list)
    # mean_scale = sum(mean_scale_list) / len(mean_scale_list)
    # logger.info(
    #     f"QLayerFwdErr(mean): "
    #     f"Abs={mean_abs_error:.3f}, "
    #     f"Rel={mean_rel_error:.3%}, "
    #     f"Activation scales(mean): {mean_scale:.3f}"
    # )
    # return mean_abs_error, mean_rel_error, mean_scale
