import contextlib
import copy
import gc
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from transformers.activations import GELUActivation, NewGELUActivation, PytorchGELUTanh
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm


allowed_norms = [nn.LayerNorm, LlamaRMSNorm, Qwen2RMSNorm]
allowed_gelu_fns = Union[nn.GELU, NewGELUActivation, PytorchGELUTanh, GELUActivation]


def clear_memory(weight=None):
    if weight is not None:
        del weight
    gc.collect()
    torch.cuda.empty_cache()


def compute_quant_errors(all_outs, q_output, block_size=4, epsilon=1e-8):
    """
    Compute the maximum absolute error and maximum relative error between the
    given outputs and quantized outputs.

    Parameters:
    - all_outs (torch.Tensor): The tensor containing all the outputs.
    - q_output (torch.Tensor): The tensor containing the quantized outputs.
    - block_size (int): The size of each block to process the outputs. Default is 4.
    - epsilon (float): A small value added to the denominator to avoid division by zero.
    Default is 1e-8.

    Returns:
    - max_abs_error (float): The maximum absolute error between the outputs and quantized outputs.
    - max_rel_error (float): The maximum relative error between the outputs and quantized outputs.
    """
    max_abs_error = 0.0
    max_rel_error = 0.0
    num_samples = all_outs.size(0)
    max_error_indices = None

    for i in range(0, num_samples, block_size):
        end = min(i + block_size, num_samples)
        all_outs_block = all_outs[i:end].cuda()
        q_output_block = q_output[i:end].cuda()

        abs_errors = (all_outs_block - q_output_block).abs()

        block_max_abs_error, block_max_indices = abs_errors.view(end - i, -1).max(dim=1)
        block_overall_max, block_max_sample = block_max_abs_error.max(dim=0)
        block_overall_max = block_overall_max.item()

        if block_overall_max > max_abs_error:
            max_abs_error = block_overall_max
            max_error_indices = (
                i + block_max_sample.item(),
                *np.unravel_index(block_max_indices[block_max_sample].item(), all_outs.shape[1:]),
            )

    # Compute relative error for the element with max absolute error
    if max_error_indices is not None:
        actual_value = all_outs[max_error_indices].abs().item()
        max_rel_error = max_abs_error / (actual_value + epsilon)

    return max_abs_error, max_rel_error


def get_named_linears(module: nn.Module, exclude_layers: List[str]) -> Dict[str, nn.Linear]:
    return {
        name: m
        for name, m in module.named_modules()
        if isinstance(m, nn.Linear) and all(exclude_name != name for exclude_name in exclude_layers)
    }


def get_op_name(module: nn.Module, op: nn.Module):
    for name, m in module.named_modules():
        if m is op:
            return name
    raise ValueError(f"Cannot find {op} in {module}")


def get_op_module(module: nn.Module, op_name: str):
    # get the op by its name relative to the module
    for name, m in module.named_modules():
        if name == op_name:
            return m
    raise ValueError(f"Cannot find op {op_name} in module {module}")


def set_op_by_name(layer, name, new_module):
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


def get_num_heads(model: nn.Module):
    if hasattr(model.config, "num_key_value_heads"):
        num_kv_heads = model.config.num_key_value_heads
    elif hasattr(model.config, "multi_query_group_num"):  # glm
        num_kv_heads = model.config.multi_query_group_num
    else:
        num_kv_heads = model.config.num_attention_heads

    num_attn_heads = model.config.num_attention_heads
    return num_kv_heads, num_attn_heads


def get_layers_to_quantize(model: nn.Module):
    model_name = model.__class__.__name__

    if model_name in [
        "LlamaForCausalLM",
        "MistralForCausalLM",
        "MixtralForCausalLM",
        "Qwen2ForCausalLM",
        "BaichuanForCausalLM",
        "BaiChuanForCausalLM",
        "DeepseekV2ForCausalLM",
        "Qwen2MoeForCausalLM",
        "GemmaForCausalLM",
        "Gemma2ForCausalLM",
    ]:
        layers = model.model.layers
    # "TELECHAT" for TeleChat-52B
    elif model_name in ["QWenLMHeadModel", "TelechatForCausalLM", "TELECHAT"]:
        layers = model.transformer.h
    elif model_name == "ChatGLMForConditionalGeneration":
        layers = model.transformer.encoder.layers
    else:
        raise NotImplementedError(f"Unknown model type: {model_name}")

    return layers


def get_lm_head_to_quantize(model: nn.Module, lm_head_name: str) -> Tuple[nn.Module, str]:
    model_name = model.__class__.__name__
    if model_name == "ChatGLMForConditionalGeneration":
        # Verify the correctness of the 'lm_head' name. This name will be
        # recorded in the configuration for future reference and usage in crossing.
        assert getattr(model.transformer, lm_head_name, None) is not None
        return nn.Sequential(
            model.transformer.encoder.final_layernorm, model.transformer.output_layer
        ), "transformer.output_layer"
    else:
        raise NotImplementedError(f"Unknown model type: {model_name} for lm_head quantize!")


@torch.inference_mode()
def get_module_outputs(
    module: nn.Module,
    inp: torch.Tensor,
    batch_size: int = 4,
    kwargs: Optional[Dict] = None,
    is_seq_first: bool = False,
    output_device: torch.device = torch.device("cuda"),
):
    if kwargs is None:
        kwargs = {}

    outputs = []
    first_output = True

    kwargs = copy.deepcopy(kwargs)
    for key in kwargs:
        if isinstance(kwargs[key], torch.Tensor):
            kwargs[key] = kwargs[key].to(inp.device)
    n_samples = inp.size(1) if is_seq_first else inp.size(0)

    if kwargs.get("attention_mask", None) is not None and kwargs["attention_mask"].ndim == 4:
        dim = 1 if is_seq_first else 0
        kwargs["attention_mask"] = kwargs["attention_mask"].repeat_interleave(batch_size, dim=dim)

    keys_to_remove = [key for key in kwargs]
    for key in keys_to_remove:
        kwargs.pop(key)

    for i in range(0, n_samples, batch_size):
        if is_seq_first:
            mini_batch = inp[:, i : i + batch_size]
            if (
                kwargs.get("attention_mask", None) is not None
                and kwargs["attention_mask"].ndim == 4
            ):
                kwargs["attention_mask"] = kwargs["attention_mask"][:, : mini_batch.size(1)]
        else:
            mini_batch = inp[i : i + batch_size]
            if (
                kwargs.get("attention_mask", None) is not None
                and kwargs["attention_mask"].ndim == 4
            ):
                kwargs["attention_mask"] = kwargs["attention_mask"][: mini_batch.size(0)]
        output = module(mini_batch, **kwargs)
        if isinstance(output, tuple):
            output = output[0]

        # Append or directly concatenate
        if first_output:
            # If we know the output shape and dtype, we can pre-allocate output tensor
            if is_seq_first:
                outputs = torch.empty(
                    (
                        output.shape[0],
                        n_samples,
                    )
                    + output.shape[2:],
                    dtype=output.dtype,
                    device=output_device,
                )
            else:
                outputs = torch.empty(
                    (n_samples,) + output.shape[1:], dtype=output.dtype, device=output_device
                )
            first_output = False
        if is_seq_first:
            outputs[:, i : i + batch_size] = output.to(output_device)
        else:
            outputs[i : i + batch_size] = output.to(output_device)
        del output
        torch.cuda.empty_cache()

    return outputs


def get_model_inputs(
    model: nn.Module,
    calib_dataloader: List[torch.Tensor],
    *,
    all_inps: torch.Tensor,
    device: Union[str, torch.device],
):
    layers = get_layers_to_quantize(model)

    model_name = model.__class__.__name__

    # Move the first layer to GPU
    if model_name in {
        "LlamaForCausalLM",
        "MistralForCausalLM",
        "MixtralForCausalLM",
        "Qwen2ForCausalLM",
        "BaichuanForCausalLM",
        "BaiChuanForCausalLM",
        "DeepseekV2ForCausalLM",
        "Qwen2MoeForCausalLM",
        "GemmaForCausalLM",
        "Gemma2ForCausalLM",
    }:
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.norm = model.model.norm.to(device)
    elif model_name in {"QWenLMHeadModel"}:
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.ln_f = model.transformer.ln_f.to(device)
        model.transformer.rotary_emb = model.transformer.rotary_emb.to(device)
        if getattr(model.transformer, "registered_causal_mask", None) is not None:
            model.transformer.registered_causal_mask = model.transformer.registered_causal_mask.to(
                device
            )
    elif model_name in {"TelechatForCausalLM"}:
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
        model.transformer.ln_f = model.transformer.ln_f.to(device)
    elif model_name in {"TELECHAT"}:  # TeleChat-52B
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.ln_f = model.transformer.ln_f.to(device)
    elif model_name == "ChatGLMForConditionalGeneration":
        model.transformer.embedding = model.transformer.embedding.to(device)
        model.transformer.rotary_pos_emb = model.transformer.rotary_pos_emb.to(device)
    else:
        raise NotImplementedError(f"Unknown model type: {model.__class__.__name__}")

    layers[0] = layers[0].to(device)

    layer_kwargs = {}
    i = 0

    def _cache_model_kwargs(_, inp, kwargs, output):
        nonlocal i
        if model_name == "ChatGLMForConditionalGeneration":
            inp, attn_mask, rope = inp
            kwargs["attention_mask"] = attn_mask
            kwargs["rotary_pos_emb"] = rope
        if isinstance(inp, tuple):
            inp = inp[0]
        all_inps[i : i + 1].data.copy_(inp)
        layer_kwargs.update(kwargs)
        i += 1
        # stop inference at the first layer
        raise ValueError

    handle = layers[0].register_forward_hook(_cache_model_kwargs, with_kwargs=True)
    for batch in calib_dataloader:
        with contextlib.suppress(ValueError):
            model(batch[0].to(device))

    handle.remove()

    # Move things back to the CPU (but not the first layer,
    # since we'll just move it back to GPU immediately below)
    if model_name in {
        "LlamaForCausalLM",
        "MistralForCausalLM",
        "MixtralForCausalLM",
        "Qwen2ForCausalLM",
        "BaichuanForCausalLM",
        "BaiChuanForCausalLM",
        "DeepseekV2ForCausalLM",
        "Qwen2MoeForCausalLM",
        "GemmaForCausalLM",
        "Gemma2ForCausalLM",
    }:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif model.__class__.__name__ in {"QWenLMHeadModel"}:
        model.transformer.wte = model.transformer.wte.cpu()
        model.transformer.ln_f = model.transformer.ln_f.cpu()
        model.transformer.rotary_emb = model.transformer.rotary_emb.cpu()
        if model.transformer.registered_causal_mask is not None:
            model.transformer.registered_causal_mask = (
                model.transformer.registered_causal_mask.cpu()
            )
    elif model.__class__.__name__ in {"TelechatForCausalLM"}:
        model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
        model.transformer.ln_f = model.transformer.ln_f.cpu()
    elif model_name in {"TELECHAT"}:  # TeleChat-52B
        model.transformer.wte = model.transformer.wte.cpu()
        model.transformer.ln_f = model.transformer.ln_f.cpu()
    elif model.__class__.__name__ == "ChatGLMForConditionalGeneration":
        model.transformer.embedding = model.transformer.embedding.to(device)
        model.transformer.rotary_pos_emb = model.transformer.rotary_pos_emb.to(device)
    else:
        raise NotImplementedError(f"Unknown model type: {model.__class__.__name__}")

    clear_memory()
    return layer_kwargs


@torch.inference_mode()
def scale_fc_fc(fc1: nn.Linear, fc2: nn.Linear, scales: torch.Tensor):
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)

    scales = scales.to(fc1.weight.device)

    fc1.weight.data[-scales.size(0) :].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.data.div_(scales.view(-1))

    fc2.weight.data.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


@torch.inference_mode()
def scale_fc_fcs(fc1: nn.Linear, fcs: List[nn.Linear], scales: torch.Tensor):
    assert isinstance(fcs, list)

    scales = scales.to(fc1.weight.device)

    # NOTE(xingyu): for qkv packed weights, just scale v
    fc1.weight.data[-scales.size(0) :].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.data[-scales.size(0) :].div_(scales.view(-1))

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.inference_mode()
def scale_ln_fcs(ln: nn.Module, fcs: Union[nn.Linear, List[nn.Linear]], scales: torch.Tensor):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)

    ln.weight.data.div_(scales)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.data.div_(scales)

    for fc in fcs:
        fc.weight.data.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.inference_mode()
def scale_gelu_fc(gelu: allowed_gelu_fns, fc: nn.Linear, scales: torch.Tensor):
    assert isinstance(gelu, allowed_gelu_fns)
    assert isinstance(fc, nn.Linear)

    fc.weight.data.mul_(scales.view(1, -1).to(fc.weight.device))

    for p in fc.parameters():
        assert torch.isnan(p).sum() == 0
