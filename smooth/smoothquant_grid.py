import functools
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from torch import nn


from quantizer import Quantizer
from utils import (
    allowed_norms,
    clear_memory,
    get_module_outputs,
    get_op_module,
    get_op_name,
    scale_fc_fc,
    scale_fc_fcs,
    scale_ln_fcs,
)

INT8_MAX = torch.iinfo(torch.int8).max
INT8_MIN = torch.iinfo(torch.int8).min


def _blockwise_abs_max(inp, scale, block_size=64):
    hidden_dim = inp.shape[-1]
    inp = inp.view(-1, hidden_dim)  # (token, hidden_dim)
    num_blocks = (inp.numel() - 1) // (block_size * hidden_dim) + 1

    max_val = torch.tensor(0.0, dtype=torch.float32, device=inp.device)
    for i in range(num_blocks):
        start_index = i * block_size
        end_index = min(start_index + block_size, inp.size(0))

        inp_block = inp[start_index:end_index, :]
        block_max = (inp_block / scale).abs().max()
        max_val = torch.max(max_val, block_max)

    return max_val


class SmoothQuantizer(Quantizer):
    def __init__(
        self,
        *,
        mse: bool = False,
        norm=2.4,
        grid=100,
        max_shrink=0.8,
        layer: nn.Module,
        disable_greedy_search: bool = False,
        is_seq_first: bool = False,
    ):
        super().__init__(
            n_bit=8,
            symm_q=True,
            group_size=-1,
            zeropoint=False,
            mse=mse,
            norm=norm,
            grid=grid,
            max_shrink=max_shrink,
        )
        self._layer = layer
        self._layer_name = layer.__class__.__name__
        self._disable_greedy_search = disable_greedy_search
        self.is_seq_first = is_seq_first

    @property
    def layer(self):
        return self._layer

    @property
    def layer_name(self):
        return self._layer_name

    def get_input_feat(
        self, named_linears: Dict[str, nn.Linear], all_inps: torch.Tensor, layer_kwargs: Dict
    ):
        input_feat = defaultdict(list)

        # Firstly, get input features of all linear layers.
        def _cache_input_hook(m, x, y, name):
            x = x[0]
            x = x.detach().cpu()
            input_feat[name].append(x)

        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(_cache_input_hook, name=name)
                )
            )
        if self.layer_name == "MixtralDecoderLayer":
            handles.append(
                self._layer.block_sparse_moe.register_forward_hook(
                    functools.partial(_cache_input_hook, name="block_sparse_moe")
                )
            )

        # trigger forward_hook
        all_outs = get_module_outputs(
            self._layer, all_inps, kwargs=layer_kwargs, is_seq_first=self.is_seq_first
        )

        for h in handles:
            h.remove()

        if self.is_seq_first:
            input_feat: Dict[str, torch.Tensor] = {
                name: torch.cat(input_feat[name], dim=1) for name in input_feat
            }
        else:
            input_feat: Dict[str, torch.Tensor] = {
                name: torch.cat(input_feat[name], dim=0) for name in input_feat
            }

        return input_feat, all_outs

    def get_layers_for_reverse_scaling(
        self, input_feat: Dict[str, torch.Tensor], layer_kwargs: Dict[str, Any]
    ):
        if self.layer_name in ["LlamaDecoderLayer", "Qwen2DecoderLayer"]:
            return _get_llama_layers_for_reverse_scaling(self.layer, input_feat, layer_kwargs)
        elif self.layer_name == "QWenBlock":  # qwen1
            return _get_qwen1_layers_for_reverse_scaling(self.layer, input_feat, layer_kwargs)
        elif self.layer_name == "BaichuanLayer":
            return _get_baichuan1_layers_for_reverse_scaling(self.layer, input_feat, layer_kwargs)
        elif self.layer_name == "GLMBlock":
            return _get_glm3_layers_for_reverse_scaling(self.layer, input_feat, layer_kwargs)
        elif self.layer_name == "MixtralDecoderLayer":
            return _get_mixtral_layers_for_reverse_scaling(self.layer, input_feat, layer_kwargs)
        else:
            raise NotImplementedError(f"Unknown model type: {self.layer_name}")

    def search_and_apply_scale(self, input_feat, layer_kwargs):
        module_config: List[Dict] = self.get_layers_for_reverse_scaling(input_feat, layer_kwargs)

        for mod_cfg in module_config:
            self._search_best_scale_and_apply(**mod_cfg)

    @torch.inference_mode()
    def _search_best_scale_and_apply(
        self,
        prev_op: nn.Module,
        linears: List[nn.Linear],
        inp: torch.Tensor,
        module2inspect: Optional[nn.Module] = None,
        kwargs: Optional[Dict] = None,
    ):
        if kwargs is None:
            kwargs = {}
        if module2inspect is None:
            assert len(linears) == 1
            module2inspect = linears[0]

        if "use_cache" in kwargs:
            kwargs.pop("use_cache")

        if torch.cuda.device_count() > 1:
            module2inspect.to("cuda:1")
        else:
            module2inspect.to("cuda")
        device = next(module2inspect.parameters()).device

        # [Step 1]: Compute per-channel max/mean of weights
        weight_abs = torch.cat([fc.weight.data for fc in linears], dim=0).abs()  # c_out, c_in
        weight_scale = weight_abs.amax(dim=0).clamp(min=1e-5)
        del weight_abs
        clear_memory()

        # [Step 2]: Compute per-channel max of activations
        act_scale = inp.to(device).abs().view(-1, inp.shape[-1]).amax(dim=0)
        clear_memory()

        # [Step 3]: Compute module fp16 output as the baseline
        fp16_output = get_module_outputs(
            module2inspect, inp.to(device), kwargs=kwargs, is_seq_first=self.is_seq_first, output_device=device
        )

        # [Step 4]: Search for the best scale factor
        best_scales = self._compute_best_scale(
            inp,
            weight_scale,
            act_scale,
            module2inspect,
            linears,
            fp16_output,
            device,
            kwargs,
        )
        linear_names = [get_op_name(self._layer, m) for m in linears]
        if (
            self.layer_name == "MixtralDecoderLayer"
            and get_op_name(self._layer, prev_op) == "post_attention_layernorm"
        ):
            # Add gate as its prev_op layernorm will be scaled.
            linear_names.append("block_sparse_moe.gate")

        self._apply_scale(
            self._layer, [(get_op_name(self._layer, prev_op), linear_names, best_scales)]
        )

    @torch.inference_mode()
    def _compute_best_scale(
        self,
        inp: torch.Tensor,
        w_scale: torch.Tensor,
        act_scale: torch.Tensor,
        module2inspect: nn.Module,
        linears: List[nn.Linear],
        fp16_output: torch.Tensor,
        device: torch.device,
        kwargs,
    ):
        """Compute quantized loss and search for the best scale factor

        L(s) = || INT8_GEMM((W * s), (s^-1 * X), s_w, s_X) - W * X||
        It's irrelevant to the bias term since it's not smoothed and quantized.
        """
        n_grid = 10
        history = []
        best_ratio = -1
        best_scales = None
        best_error = float("inf")

        org_state = {k: v.cpu() for k, v in module2inspect.state_dict().items()}

        act_scale = act_scale.view(-1).to(device)
        w_scale = w_scale.view(-1).to(device)

        if self._disable_greedy_search:
            # following SmoothQuant to set ratio=0.5 if not search
            ratio = 0.5
            best_scales = (act_scale.pow(ratio) / w_scale.pow(1 - ratio)).clamp(min=1e-5)
            return best_scales.cpu()

        tile_size = 16
        infer_dtype = inp.dtype
        start = time.time()
        # for ratio in range(n_grid):
        ratio_list = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.92,0.94,0.96,0.98]
        # for ratio in range(n_grid):
        for ratio in ratio_list:
            module2inspect.to(device)
            # NOTE(xingyu): Initiate search within the range 0.5 to 1 due to prior
            # observations indicating enhanced end-to-end performance within this interval.
            # ratio = ratio / (2 * n_grid) + 0.5
            # ratio = ratio / n_grid

            sq_scales = (act_scale.pow(ratio) / w_scale.pow(1 - ratio)).clamp(min=1e-5)
            sq_scales_view = sq_scales.view(1, -1).to(device)

            # smoothed input scale for int8 quantization
            # input_scale = _blockwise_abs_max(inp.to(device), sq_scales_view) / INT8_MAX
            clear_memory()

            quant_hooks = []
            for fc in linears:
                fc.weight.data.mul_(sq_scales_view)
                w_quantizer = Quantizer(
                    n_bit=8, per_tensor=False, group_size=-1, symm_q=True, zeropoint=False
                )
                fc.weight.data = w_quantizer.quantize(fc.weight.data)
                # fc.output_scale = w_quantizer.scales[:, 0] * input_scale
                fc.output_scale = w_quantizer.scales[:, 0] 

                def int8_quantize(m, inp):
                    inp = inp[0]
                    input_scale = inp.abs().max(dim=-1).values.unsqueeze(-1) / INT8_MAX
                    qint8_inp = (
                        torch.round(
                            (inp / sq_scales_view)  # smooth
                            / input_scale  # quant
                        )
                        .clamp(INT8_MIN, INT8_MAX)
                        .float() * input_scale  # mock int8 gemm using float to prevent overflow
                    )
                    if m.bias is not None:
                        m.bias.data = m.bias.data.float()
                    return qint8_inp

                def int8_dequantize(m, inp, out):
                    if m.bias is not None:
                        out -= m.bias
                        return (
                            out * m.output_scale  # dequant
                            + m.bias
                        ).to(dtype=infer_dtype)
                    else:
                        return (out * m.output_scale).to(dtype=infer_dtype)

                quant_hooks.append(fc.register_forward_pre_hook(int8_quantize))
                quant_hooks.append(fc.register_forward_hook(int8_dequantize))

            q_output = get_module_outputs(
                module2inspect, inp.to(device), kwargs=kwargs, is_seq_first=self.is_seq_first, output_device=device
            )

            clear_memory(inp)

            for hook in quant_hooks:
                hook.remove()

            # Compute L1 error
            loss = max(
                ((fp16_output[i : i + tile_size] - q_output[i : i + tile_size]).abs().max().item())
                for i in range(0, fp16_output.shape[0], tile_size)
            )

            history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scales = sq_scales.clone()

            # convert weights back to original dtype as we convert to
            # fp32 for int8 gemm simulation before
            module2inspect.to(dtype=infer_dtype)
            module2inspect.load_state_dict(org_state)
            clear_memory()

        if best_ratio == -1:
            raise ValueError("No best ratio found")

        elapse = time.time() - start
        assert torch.isnan(best_scales).sum() == 0, f"best_scales: {best_scales}"
        logger.info(f"{list(module2inspect.state_dict().keys())[0]}, Best scale:{best_ratio}")
        return best_scales.cpu()

    @staticmethod
    def _apply_scale(
        layer: nn.Module,
        scales_list: List[Tuple[str, List[str], torch.Tensor]],
    ):
        for prev_op_name, linear_names, scales in scales_list:
            prev_op = get_op_module(layer, prev_op_name)
            linears = [get_op_module(layer, name) for name in linear_names]

            prev_op.cuda()
            for linear in linears:
                linear.cuda()
            scales.cuda()

            if isinstance(prev_op, nn.Linear) and isinstance(linears[0], nn.Linear):
                scale_fc_fcs(prev_op, linears, scales)

            elif isinstance(prev_op, nn.Linear):
                assert len(linears) == 1
                scale_fc_fc(prev_op, linears[0], scales)

            elif (
                any(isinstance(prev_op, t) for t in allowed_norms)
                or "rmsnorm" in str(prev_op.__class__).lower()
            ):
                scale_ln_fcs(prev_op, linears, scales)

            else:
                raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")

            prev_op.cpu()
            for linear in linears:
                linear.cpu()
            scales.cpu()


@torch.inference_mode()
def get_static_activation_scales(
    n_bit: int,
    layer: nn.Module,
    linears_to_quantize: Dict[str, nn.Linear],
    all_inps: torch.Tensor,
    kwargs: Optional[Dict] = None,
):
    act_dict = defaultdict(dict)

    def stat_io_hook(_, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        if name not in act_dict or "input" not in act_dict[name]:
            act_dict[name]["input"] = x.abs().max().item()
        else:
            act_dict[name]["input"] = max(act_dict[name]["input"], x.abs().max().item())
        if isinstance(y, tuple):
            y = y[0]
        if name not in act_dict or "output" not in act_dict[name]:
            act_dict[name]["output"] = y.abs().max().item()
        else:
            act_dict[name]["output"] = max(act_dict[name]["output"], y.abs().max().item())

    hooks = []
    for name, m in linears_to_quantize.items():
        hooks.append(m.register_forward_hook(functools.partial(stat_io_hook, name=name)))

    mean_scale = []
    for i in range(len(all_inps)):
        layer(all_inps[i : i + 1], **kwargs)
        mean_scale.append(np.mean([v["input"] for v in act_dict.values()]))
    mean_scale = np.mean(mean_scale)

    for hook in hooks:
        hook.remove()

    scale = 2 ** (n_bit - 1) - 1
    scale_dict = {f"{key}_input": value["input"] / scale for key, value in act_dict.items()}

    return mean_scale, scale_dict


def _get_llama_layers_for_reverse_scaling(
    layer,
    input_feat: Dict[str, torch.Tensor],
    layer_kwargs: Dict[str, Any],
):
    # attention out
    # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
    if layer.self_attn.v_proj.weight.shape == layer.self_attn.o_proj.weight.shape:
        layers = [
            dict(
                prev_op=layer.self_attn.v_proj,
                linears=[layer.self_attn.o_proj],
                inp=input_feat["self_attn.o_proj"],
            )
        ]
    else:
        layers = []

    layers.append(
        # attention input
        dict(
            prev_op=layer.input_layernorm,
            linears=[
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ],
            inp=input_feat["self_attn.q_proj"],
            module2inspect=layer.self_attn,
            kwargs=layer_kwargs,
        )
    )
    # linear 2
    layers.append(
        dict(
            prev_op=layer.mlp.up_proj,
            linears=[layer.mlp.down_proj],
            inp=input_feat["mlp.down_proj"],
        )
    )

    # linear 1
    layers.append(
        dict(
            prev_op=layer.post_attention_layernorm,
            linears=[layer.mlp.gate_proj, layer.mlp.up_proj],
            inp=input_feat["mlp.gate_proj"],
            module2inspect=layer.mlp,
        )
    )

    return layers


def _get_qwen1_layers_for_reverse_scaling(
    layer,
    input_feat: Dict[str, torch.Tensor],
    layer_kwargs: Dict[str, Any],
):
    return [
        dict(
            prev_op=layer.attn.c_attn,
            linears=[layer.attn.c_proj],
            inp=input_feat["attn.c_proj"],
        ),
        dict(
            prev_op=layer.ln_1,
            linears=[layer.attn.c_attn],
            inp=input_feat["attn.c_attn"],
            module2inspect=layer.attn,
            kwargs=layer_kwargs,
        ),
        dict(
            prev_op=layer.mlp.w1,
            linears=[layer.mlp.c_proj],
            inp=input_feat["mlp.c_proj"],
        ),
        dict(
            prev_op=layer.ln_2,
            linears=[layer.mlp.w2, layer.mlp.w1],
            inp=input_feat["mlp.w2"],
            module2inspect=layer.mlp,
        ),
    ]


def _get_baichuan1_layers_for_reverse_scaling(
    layer,
    input_feat: Dict[str, torch.Tensor],
    layer_kwargs: Dict[str, Any],
):
    return [
        dict(
            prev_op=layer.self_attn.W_pack,
            linears=[layer.self_attn.o_proj],
            inp=input_feat["self_attn.o_proj"],
        ),
        dict(
            prev_op=layer.input_layernorm,
            linears=[layer.self_attn.W_pack],
            inp=input_feat["self_attn.W_pack"],
            module2inspect=layer.self_attn,
            kwargs=layer_kwargs,
        ),
        dict(
            prev_op=layer.mlp.up_proj,
            linears=[layer.mlp.down_proj],
            inp=input_feat["mlp.down_proj"],
        ),
        dict(
            prev_op=layer.post_attention_layernorm,
            linears=[layer.mlp.gate_proj, layer.mlp.up_proj],
            inp=input_feat["mlp.gate_proj"],
            module2inspect=layer.mlp,
        ),
    ]


def _get_glm3_layers_for_reverse_scaling(
    layer,
    input_feat: Dict[str, torch.Tensor],
    layer_kwargs: Dict[str, Any],
):
    return [
        # NOTE(xingyu): MQA is not supported!
        dict(
            prev_op=layer.input_layernorm,
            linears=[layer.self_attention.query_key_value],
            inp=input_feat["self_attention.query_key_value"],
            module2inspect=layer.self_attention,
            kwargs=layer_kwargs,
        ),
        dict(
            prev_op=layer.mlp.dense_h_to_4h,
            linears=[layer.mlp.dense_4h_to_h],
            inp=input_feat["mlp.dense_4h_to_h"],
        ),
        dict(
            prev_op=layer.post_attention_layernorm,
            linears=[layer.mlp.dense_h_to_4h],
            inp=input_feat["mlp.dense_h_to_4h"],
            module2inspect=layer.mlp,
        ),
    ]


def _get_mixtral_layers_for_reverse_scaling(
    layer,
    input_feat: Dict[str, torch.Tensor],
    layer_kwargs: Dict[str, Any],
):
    layers = [
        # attention input
        dict(
            prev_op=layer.input_layernorm,
            linears=[
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ],
            inp=input_feat["self_attn.q_proj"],
            module2inspect=layer.self_attn,
            kwargs=layer_kwargs,
        )
    ]
    # MoE has multiple MLPs called experts
    # MLP: w2(silu(w1(x)) * w3(x))
    num_experts = layer.block_sparse_moe.num_experts
    for i in range(num_experts):
        expert = layer.block_sparse_moe.experts[i]

        layers.append(
            dict(
                prev_op=expert.w3,
                linears=[expert.w2],
                inp=input_feat[f"block_sparse_moe.experts.{i}.w2"],
            )
        )

    layers.append(
        dict(
            prev_op=layer.post_attention_layernorm,
            linears=[
                weight
                for i in range(num_experts)
                for weight in (
                    layer.block_sparse_moe.experts[i].w3,
                    layer.block_sparse_moe.experts[i].w1,
                )
            ],
            inp=input_feat["block_sparse_moe"],
            module2inspect=layer.block_sparse_moe,
        )
    )

    return layers
