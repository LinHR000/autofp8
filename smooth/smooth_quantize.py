import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from loguru import logger
from transformers import AutoTokenizer, HfArgumentParser,AutoModelForCausalLM

from smoothquant_runner import run_smooth_quantize
from auto_fp8.datautils import get_loaders

@dataclass
class SmoothQuantArguments:
    model_name_or_path: str = field(
        default = '/mnt/data/project/skyllm/shared/test/Qwen2-72B-Instruct',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    save_dir: str = field(
        default = '/mnt/data/project/skyllm/shared/test/Qwen2-72B-Instruc_smooth_grid',
        metadata={"help": "Path to save quantized model"},
    )
    dtype: str = field(
        default="bfloat16",
        metadata={"help": "dtype for the pretrained model inference"},
    )
    safetensors: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to save quantized model in safetensors format"},
    )
    disable_greedy_search: Optional[bool] = field(
        default=False, metadata={"help": "Whether to disable greedy search for SmoothQuant"}
    )
    seed: Optional[int] = field(
        default=42, metadata={"help": "Seed for sampling the calibration data"}
    )
    calib_set: Optional[str] = field(
        default="c4",
        metadata={
            "help": "Path to calibration data or dataset identifier from huggingface.co/dataset"
        },
    )
    download_mode: Optional[str] = field(
        default="reuse_dataset_if_exists",
        metadata={
            "help": "Download mode for the calibration data, "
            '"reuse_dataset_if_exists","reuse_cache_if_exists", "force_redownload".'
        },
    )
    loading_script: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the script for loading the calibration data"},
    )
    n_samples: Optional[int] = field(
        default=128, metadata={"help": "Number of samples for calibration"}
    )
    seq_len: Optional[int] = field(
        default=2048, metadata={"help": "Sequence length for calibration"}
    )
    exclude_layers: Optional[List[str]] = field(
        default_factory=lambda: [],
        metadata={"help": "Layer name in transformers_layers to exclude from quantization."},
    )
    quant_lm_head: Optional[bool] = field(
        default=False, metadata={"help": "Whether to quantize the lm_head layer."}
    )
    lm_head_name: Optional[str] = field(
        default="lm_head", metadata={"help": "Name of the lm_head layer."}
    )

    def __post_init__(self):
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.dtype = getattr(torch, self.dtype)

def save_model(model: nn.Module, save_dir: Path, safetensors: bool = True):
    if safetensors:
        # Some `generation_config` set the temperature > 0, but also set `do_sample` to False.
        # This is contradictory according to HuggingFace's standards.
        # As a workaround, we override `do_sample` and set it to `True`.
        model.generation_config.do_sample = True
        model.save_pretrained(save_dir)

    else:
        torch.save(model.state_dict(), save_dir / "model.pt")


def main():
    parser = HfArgumentParser(SmoothQuantArguments)
    args = parser.parse_args_into_dataclasses()[0]

    assert not args.quant_lm_head, "lm_head quant is not supported for smooth_quantize!"
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype = args.dtype,device_map='cpu',trust_remote_code=True)

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=False, trust_remote_code=True
    )

    logger.info(f"Loading calibration set {args.calib_set}...")
    dataloader, testenc = get_loaders('wikitext2',
                                        seed=42,
                                        model=args.model_name_or_path,
                                        nsamples=128,
                                        seqlen=2048)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    quant_config = {
        "quant_type": 'weight_activation_int8',
        "bits": 8,
        "packing_dtype": "int8",
        # for weight activation quantization, we always use per-channel weight quantization
        # for efficiency and precision trade-off.
        "group_size": -1,
        "symm_q": True,
    }
    tick = time.time()
    # quant_abs_err, quant_rel_err, mean_scale = run_smooth_quantize(
    run_smooth_quantize(
        model,
        dataloader,
        seq_len=args.seq_len,
        exclude_layers=args.exclude_layers,
        disable_greedy_search=args.disable_greedy_search,
    )
    logger.info("Quantization takes {:.2f} minutes".format((time.time() - tick) / 60))
    # quant_config |= {
    #     "exclude_layers": args.exclude_layers,
    #     "quant_abs_error": quant_abs_err,
    #     "quant_rel_error": quant_rel_err,
    #     "activation_mean_scale": mean_scale,
    #     "load_format": "safetensors" if args.safetensors else "pt",
    # }
    logger.info(f"Save quantized model to {save_dir}...")
    save_model(model, save_dir, args.safetensors)

    logger.info("Done.")


if __name__ == "__main__":
    main()
