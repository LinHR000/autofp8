import os
import torch
import argparse
from transformers import AutoTokenizer,AutoConfig,AutoModelForCausalLM
from auto_fp8.eval_ppl import llama_eval,llama_eval_v2
from auto_fp8.datautils import get_loaders
from auto_fp8.logger import logger
from get_act_scales import get_act_scales
from smooth import smooth_lm
from auto_fp8.models.modeling_qwen2 import Qwen2ForCausalLMMergeGemm


parser = argparse.ArgumentParser()
# =============================模型输入输出参数=============================================================================================================
parser.add_argument("--model_path", type=str, default='/mnt/data/project/skyllm/shared/test/Qwen2-72B-Instruct', help="model name of model path") 
parser.add_argument("--save_path", default="/mnt/data/linhaoran/models/Qwen2-72B-Instruct-smooth-qkv_o_gate_up_down", type=str, help="direction of logging file")
parser.add_argument("--smooth_part", default=['qkv_proj','gate_up_proj','down_proj',"o_proj"], type=list, help="direction of logging file")
parser.add_argument("--alpha", default=0.5, type=float, help="direction of logging file")
parser.add_argument("--calib_dataset",type=str,default="wikitext2",
    choices=["wikitext2", "ptb", "c4", "mix","pile"],
    help="Where to extract calibration data from.",
)
parser.add_argument("--nsamples", type=int, default=512, help="Number of calibration data samples.")
parser.add_argument("--seqlen", type=int, default=2048, help="Number of calibration data samples.")
parser.add_argument("--seed", type=int, default=42, help="batch size.")
# =============================模型训练超参参数=============================================================================================================
args = parser.parse_args()

def load_bin(state_dict,num_hidden_layers):
    patten_q = ""
    patten_gate = ''
    for key in state_dict.keys():
        if ".0" in key and "q_proj" in key and "weight" in key:
            patten_q = key
        if ".0" in key and "gate_proj" in key:
            patten_gate = key

    for i in range(num_hidden_layers):
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


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map = 'auto', trust_remote_code=True,torch_dtype=torch.bfloat16)
    traindataset, testenc = get_loaders(args.calib_dataset,
                                        seed=args.seed,
                                        model=args.model_path,
                                        nsamples=args.nsamples,
                                        seqlen=args.seqlen)
    # ppl = llama_eval_v2(model, testenc)

    act_scales = get_act_scales(model, tokenizer, traindataset, args.nsamples, args.seqlen)
    model_new = Qwen2ForCausalLMMergeGemm(model.config).to(model.config.torch_dtype).cpu()
    state_dict = load_bin(model.state_dict(),model.config.num_hidden_layers)
    model_new.load_state_dict(state_dict, strict=False)
    # ppl_smoth = llama_eval_v2(model, testenc)
    # print(f"PPL smooth:{ppl_smoth}")
    # print(f"PPL : {ppl}, PPL after smooth : {ppl_smoth}")
    model_new.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    torch.save(act_scales, os.path.join(args.save_path,f'smooth_act_scales-{args.alpha}.pt'))
