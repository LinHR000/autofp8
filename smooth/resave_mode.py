from transformers import AutoTokenizer, HfArgumentParser,AutoModelForCausalLM,AutoConfig
import torch
import os
from auto_fp8.models.modeling_qwen2 import Qwen2ForCausalLMMergeGemm
model_path1 = '/mnt/data/project/skyllm/shared/test/Qwen2-72B-Instruct_merge_qkv'
model_path2 = '/mnt/data/project/skyllm/shared/test/Qwen2-72B-Instruc_smooth_grid'
save_path = '/mnt/data/linhaoran/models/Qwen2-72B-Instruc_smooth_grid_resave'

def load_bin(model_path):
    from safetensors.torch import load_file
    config = AutoConfig.from_pretrained(model_path,trust_remote_code=True)
    dir_list = os.listdir(model_path)
    state_dict = {}
    for file in dir_list:
        if file.endswith('safetensors'):
            state_dict.update(load_file(os.path.join(model_path, file)))
        elif file.endswith('.bin'):
            state_dict.update(torch.load(os.path.join(model_path, file),map_location='cpu'))

    return state_dict
    
state_dict = load_bin(model_path2)
model = Qwen2ForCausalLMMergeGemm.from_pretrained(model_path1,device_map='auto',trust_remote_code=True,torch_dtype=torch.bfloat16)
model.load_state_dict(state_dict, strict=False)
model.save_pretrained(save_path)
tokenizer = AutoTokenizer.from_pretrained(model_path1,trust_remote_code=True)
tokenizer.save_pretrained(save_path)