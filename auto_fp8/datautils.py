import pdb
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import torch
import random
import pickle
import os
import json
from auto_fp8.logger import logger
current_directory = os.getcwd()
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)




def get_pile(nsamples, seed, seqlen, model):
    print("get_pile")
    traindata = load_dataset("json", data_files='/cpfs01/user/chenmengzhao/prompt_quantization/val.jsonl.zst', split="train")

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text'][:1000]), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, None

"""
def get_wikitext2(nsamples, seed, seqlen,model):
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False,trust_remote_code=True)
    print("get_wikitext2")
    # traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    # testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    with open('/mnt/infra/haoran.lin2/datasets/ppl_data/wiki2/train.pkl','rb') as r:
        traindata = pickle.load(r)
    with open('/mnt/infra/haoran.lin2/datasets/ppl_data/wiki2/val.pkl','rb') as r:
        devdata = pickle.load(r)
    with open('/mnt/infra/haoran.lin2/datasets/ppl_data/wiki2/test.pkl','rb') as r:
        testdata = pickle.load(r)

    # tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    a = "\n\n".join(traindata['text'])
    text_len = len(a)
    gap = 1000000
    input_ids = []
    if text_len < gap:
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
        input_ids = trainenc.input_ids.view(-1)
    else:
        # trainenc = tokenizer("\n\n".join(traindata['text'])[:gap], return_tensors='pt')
        count = 0
        print(count)
        while count * gap < text_len:
            tmp = tokenizer("\n\n".join(traindata['text'])[count*gap:gap*(count+1)], return_tensors='pt')
            count+=1
            print(count)
            input_ids.append(tmp.input_ids.view(-1))
    input_ids = torch.cat(input_ids)
    input_ids = input_ids.view(1,-1)

    # print(trainenc.input_ids.shape)
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    devenc = tokenizer("\n\n".join(devdata['text']), return_tensors='pt')

    
    random.seed(seed)
    trainloader = []
    devloader = []
    for _ in range(nsamples):
        i = random.randint(0, input_ids.shape[1]//2 - seqlen - 1)
        j = i + seqlen
        inp = input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    for _ in range(nsamples):
        i = random.randint(input_ids.shape[1]//2, input_ids.shape[1]- seqlen - 1)
        j = i + seqlen
        inp = input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        devloader.append((inp, tar))

    return trainloader, devloader, testenc
"""
def get_wikitext2(nsamples, seed, seqlen, model):
    logger.info("get_wikitext2")
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False,trust_remote_code=True)
    # if os.path.isfile(os.path.join(current_directory, 'datasets/wiki_train.json')):
    train_file = '/mnt/data/linhaoran/AutoFP8/datasets/wiki_train.json'
    test_file = '/mnt/data/linhaoran/AutoFP8/datasets/wiki_test.json'
    if os.path.isfile(train_file):
        with open(train_file,'r') as r:
            traindata = json.load(r)['data']
        with open(test_file,'r') as r:
            testdata = json.load(r)['data']
        trainenc = tokenizer("\n\n".join(traindata), return_tensors='pt')
        testenc = tokenizer("\n\n".join(testdata), return_tensors='pt')
    else:
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model):
    print("get_ptb")
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')


    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model):
    print("get_c4")
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )


    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    return trainloader, valenc 

def get_ptb_new(nsamples, seed, seqlen, model):
    print("get_ptb_new")
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata  = load_dataset('ptb_text_only', 'penn_treebank', split='test')


    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer(" ".join(testdata ["sentence"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_new(nsamples, seed, seqlen, model):
    print("get_c4_new")
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]
    return trainloader, valenc


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='',
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if 'pile' in name:
        return get_pile(nsamples, seed, seqlen, model)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model)  
        return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model)  
        return get_c4(nsamples, seed, seqlen, model)
    if 'mix' in name:
        wiki_train,wiki_val=get_wikitext2(nsamples//3, seed, seqlen, model)
        ptb_train,ptb_val=get_ptb(nsamples//3, seed, seqlen, model)
        c4_train,c4_val=get_c4(nsamples//3, seed, seqlen, model)
        train=wiki_train+ptb_train+c4_train
        val=None
        return train,val

def get_bit_dis_loaders(file_path, model, nsamples):
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False,trust_remote_code=True)
    with open(data_path, 'r') as f:
        lines = f.readlines()
    all_dataset = [json.loads(line.strip()) for line in lines]

    sources, targets = zip(*[(s[0][0], f"{s[0][1]}{tokenizer.eos_token}") for s in all_dataset])

    dataset_size = len(sources)
    max_sample = min(nsamples or dataset_size, dataset_size)
    if max_sample < dataset_size:
        indices = random.sample(range(dataset_size), max_sample)
        sources, targets = [sources[i] for i in indices], [targets[i] for i in indices]
    else:
        sources, targets = sources, targets 
                
    split_num = len(sources) // 5
    sources_train, targets_train = sources[split_num:], targets[split_num:]
    print(f"Using {len(sources_train)} samples to train")

    sources_eval, targets_eval = sources[:split_num], targets[:split_num]
    print(f"Using {len(sources_eval)} samples to evaluation")

    trainloader = []
    devloader = []
    for text in sources_train:
        trainenc = tokenizer(text, return_tensors='pt')
        input_ids = trainenc.input_ids.view(-1)
        tar = input_ids.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    for text in sources_eval:
        trainenc = tokenizer(text, return_tensors='pt')
        input_ids = trainenc.input_ids.view(-1)
        tar = input_ids.clone()
        tar[:, :-1] = -100
        devloader.append((inp, tar))

    return trainloader, devloader
    pass