import torch
from auto_fp8.models.modeling_qwen2 import Qwen2ForCausalLM
from transformers import AutoTokenizer,AutoConfig,AutoModelForCausalLM


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
    patten_q = ""
    patten_gate = ''
    for key in state_dict.keys():
        if ".0" in key and "q_proj" in key and "weight" in key:
            patten_q = key
        if ".0" in key and "gate_proj" in key:
            patten_gate = key

    for i in range(config.num_hidden_layers):
        new_name = patten_q.replace("0",str(i))
        new_name_save = new_name.replace("q_proj","qkv_proj")
        new_value = torch.cat([state_dict[new_name],state_dict[new_name.replace("q_proj","k_proj")],state_dict[new_name.replace('q_proj','v_proj')]])
        state_dict[new_name_save] = new_value
        state_dict.pop(new_name)
        state_dict.pop(new_name.replace("q_proj","k_proj"))
        state_dict.pop(new_name.replace('q_proj','v_proj'))


        new_name = patten_q.replace("0",str(i))
        new_name = new_name.replace(".weight",".bias")
        new_name_save = new_name.replace("q_proj","qkv_proj")
        new_value = torch.cat([state_dict[new_name],state_dict[new_name.replace("q_proj","k_proj")],state_dict[new_name.replace('q_proj','v_proj')]])
        state_dict[new_name_save] = new_value
        state_dict.pop(new_name)
        state_dict.pop(new_name.replace("q_proj","k_proj"))
        state_dict.pop(new_name.replace('q_proj','v_proj'))


        new_name = patten_gate.replace("0",str(i))
        new_name_save = new_name.replace("gate_proj","gate_up_proj")
        new_value = torch.cat([state_dict[new_name],state_dict[new_name.replace("gate_proj","up_proj")]])
        state_dict[new_name_save] = new_value
        state_dict.pop(new_name)
        state_dict.pop(new_name.replace("gate_proj","up_proj"))
    return state_dict

model_path = "/mnt/data/project/skyllm/shared/test/Qwen2-72B-Instruct"
save_path = "/mnt/data/project/skyllm/shared/test/Qwen2-72B-Instruct_t"
config = AutoConfig.from_pretrained(model_path,trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
model = Qwen2ForCausalLM(config).to(config.torch_dtype).cpu()
state_dict = load_bin(model_path)
model.load_state_dict(state_dict, strict=False)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)


