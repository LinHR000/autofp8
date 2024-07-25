import torch
import os
from datasets import load_dataset
from transformers import AutoTokenizer,Qwen2ForCausalLM,Qwen2MoeForCausalLM

from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig
from auto_fp8.eval_ppl import llama_eval_v2
from auto_fp8.datautils import get_loaders



# pretrained_model_dir = "/mnt/data/project/skyllm/shared/test/Qwen2-72B-Instruct"
# pretrained_model_dir = "/mnt/data/project/skyllm/shared/test/Qwen2-72B-Instruct_merge_qkv"
# pretrained_model_dir = '/mnt/data/linhaoran/models/Qwen2-72B-Instruct-smooth-qkv_gate_up'
# pretrained_model_dir = '/mnt/data/linhaoran/models/Qwen2-72B-Instruct-smooth-qkv_gate_up_down'
pretrained_model_dir = '/mnt/data/linhaoran/models/Qwen2-72B-Instruct-smooth-qkv_o_gate_up_down'
# pretrained_model_dir = '/mnt/data/linhaoran/models/Qwen2-72B-Instruct-smooth-qkv_o_gate_up_down_max'
# pretrained_model_dir = '/mnt/data/linhaoran/models/Qwen2-72B-Instruct-smooth-qkv_o_gate_up_down'
quantized_model_dir = "/mnt/data/linhaoran/models/Qwen2-72B-Instruc_smooth_grid-FP8"
pretrained_model_dir = '/mnt/data/linhaoran/models/Qwen2-72B-Instruc_smooth_grid_resave'

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
model_name = 'qwen2_72b'
seqlen = 2048
nsamples = 128
traindataset_path = os.path.join('cache/', f'traindataset-{model_name}-{seqlen}-{nsamples}.cache')
testenc_path = os.path.join('cache/', f'testenc-{model_name}-{seqlen}-{nsamples}.cache')
if not os.path.isfile(traindataset_path):
    traindataset, testenc = get_loaders(
        'wikitext2',
        seed=42,
        model=pretrained_model_dir,
        nsamples=128,
        seqlen=2048)
    torch.save(traindataset, traindataset_path)
    torch.save(testenc, testenc_path)
traindataset = torch.load(traindataset_path)
testenc = torch.load(testenc_path)

examples = []
for i in range(128):
    examples.append(traindataset[i][0])
# examples = tokenizer(examples, padding=True, truncation=True, return_tensors="pt").to("cuda")
examples = torch.cat(examples).to('cuda')

quantize_config = BaseQuantizeConfig(
    # quant_method="fp8",
    quant_method="int8",
    activation_scheme="static",
    # ignore_patterns=["re:.*lm_head", "re:.*gate", "re:.*o_proj"],
    ignore_patterns=["re:.*lm_head", "re:.*gate"],
    # output_quant_targets = ['qkv_proj','gate_up_proj']
    output_quant_targets = []
)

model = AutoFP8ForCausalLM.from_pretrained(
    pretrained_model_dir, quantize_config=quantize_config
)
model.quantize(examples)
llama_eval_v2(model.model,testenc)

# model.save_quantized(quantized_model_dir)
