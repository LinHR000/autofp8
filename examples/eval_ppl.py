# eval ppl
import numpy as np
import torch
import torch.nn as nn
from logger import logger
from tqdm import tqdm


@torch.no_grad()
def llama_eval(model, testenc, dev, seqlen=2048):
    logger.info("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    # model.model.embed_positions = model.model.embed_positions.to(dev)
    if hasattr(model.model, "project_out") and model.model.project_out:
        model.model.project_out = model.model.project_out.to(dev)
    if hasattr(model.model, "project_in") and model.model.project_in:
        model.model.project_in = model.model.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None,'position_ids':None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    # model.model.embed_positions = model.model.embed_positions.cpu()
    if hasattr(model.model, "project_out") and model.model.project_out:
        model.model.project_out = model.model.project_out.cpu()
    if hasattr(model.model, "project_in") and model.model.project_in:
        model.model.project_in = model.model.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
            # outs[j] = layer(inps[j].unsqueeze(0), position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    # if model.model.project_out is not None:
        # model.model.project_out = model.model.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        # if model.model.project_out is not None:
            # hidden_states = model.model.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    logger.info(f"PPL:{ppl.item()}")

@torch.no_grad()
def llama_eval_v2(model, testenc, seqlen=2048):
    logger.info("Evaluating ...")
    nsamples = testenc['input_ids'].numel() // seqlen
    # nsamples = 10
    nlls = []
    for i in tqdm(range(nsamples)):
        hidden_states = testenc['input_ids'][:, (i * seqlen) : ((i + 1) * seqlen)]
        output = model(hidden_states)
        lm_logits = output.logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc["input_ids"][:, (i * seqlen) : ((i + 1) * seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    logger.info(f"PPL:{ppl.item()}")
    return ppl
