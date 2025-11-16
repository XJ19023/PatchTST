'''
Qwen2.5-0.5 finetune is OK
'''
import os
import random
import shutil
import sys
import torch
import torch.nn as nn
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          AutoConfig,
                          TrainingArguments,
                          DataCollatorForLanguageModeling,
                          Trainer,
                          )
from tqdm import tqdm
import torch.distributed as dist
from safetensors.torch import save_file

from datasets import load_dataset
import argparse
import torch.nn.functional as F
from torch.utils.data import Dataset
import time
from datetime import datetime
import sys
from globalVar import (increas_iterationCounter,
                       save_tensors,
                       set_save_tensor_enable,
                       get_ratio,
                       reset_counter,
                       set_mse,
                       set_clamp_block_size,
                       set_clamp_type)

def tensor_gpu_memory(tensor, name=None):
    if tensor.is_cuda:
        size_in_bytes = tensor.element_size() * tensor.numel()
        size_in_MB = size_in_bytes / 1024**2
        print(f"{name if name else ''} size: {size_in_MB:.2f} MB")
    else:
        print(f"{name if name else ''} is not on GPU.")
# current date and time
current_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
start_time = time.time()
# ----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="opt-125m")
parser.add_argument("--task", type=str, default="wikitext")
parser.add_argument("--n_samples", type=int, default=None)
parser.add_argument("--clamp_block_size", type=int, default=64)
parser.add_argument("--clamp_type", type=str, default='int6')
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument("--local-rank", default=0, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--wgt_nbit", default=4, type=int)
parser.add_argument("--act_nbit", default=8, type=int)
parser.add_argument("--eval_base", action="store_true")
parser.add_argument("--eval_quant", action="store_true")
parser.add_argument("--eval_clamp", action="store_true")
parser.add_argument("--eval_spark", action="store_true")
parser.add_argument("--layer_wise_delta", action="store_true")
parser.add_argument("--profiling", action="store_true")
parser.add_argument("--save_tensor", action="store_true")
args = parser.parse_args()

if args.layer_wise_delta:
    from quant_delta import quantLinear
else:
    from quant import quantLinear

with open('log/mse.txt', 'w') as f:
    pass

@torch.no_grad()
def evaluate(model, dataset, n_samples=None, comment='Evaluating...'):
    model.eval()
    nlls = []
    length = 2048
    n_samples = n_samples if n_samples else dataset.size(1) // length
    for i in tqdm(range(n_samples), desc=comment):
        batch = dataset[:, (i * length) : ((i + 1) * length)].to(model.device)
        with torch.no_grad():
            lm_logits = model(batch).logits
            if 'search' in comment:
                return None
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = dataset[:, (i * length) : ((i + 1) * length)][:, 1:].to(model.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * length
        nlls.append(neg_log_likelihood)

        _ = increas_iterationCounter()

    return torch.exp(torch.stack(nlls).sum() / (n_samples * length))

args.world_size = 1
args.rank = 0  # global rank
model_path = '/localssd/lbxj/' + args.model_name
# model_path = '/cephfs/juxin/models/' + args.model_name
n_samples = args.n_samples
train_samples = 64 # 64
# set_save_tensor_enable()

tokenizer = AutoTokenizer.from_pretrained(model_path)
# test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
# train_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
# evaluator = Evaluator(test_data, tokenizer, "cuda", n_samples=n_samples)


def get_ptb(seqlen, tokenizer, nsamples=512):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt').input_ids

    import random
    random.seed(0)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return testenc
    # return trainloader, testenc

def get_wikitext2(tokenizer, eval=True):
    if eval:
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    else:
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt").input_ids
    return testenc

def get_c4(seqlen, tokenizer, eval=False):
    if eval:
        valdata = load_dataset(
            "json",
            data_files={"validation": "/cephfs/shared/juxin/dataset/c4/en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
    else:
        valdata = load_dataset(
            "json",
            data_files={"train": "/cephfs/shared/juxin/dataset/c4/en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
            if tmp.input_ids.shape[1] >= seqlen:
                break
        if tmp.input_ids.shape[1] == seqlen:
            # rare case, discovered with Yi tokenizer
            valenc.append(tmp.input_ids)
        else:
            i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    return valenc    


if args.task == 'wikitext':
    test_data = get_wikitext2(tokenizer, True)
    train_data = get_wikitext2(tokenizer, False)
if args.task == 'c4':
    test_data = get_c4(2048, tokenizer, True)
    train_data = get_c4(2048, tokenizer, True)
if args.task == 'ptb':
    test_data = get_ptb(2048, tokenizer)

os.makedirs(f'log/{args.model_name}', exist_ok=True)
with open('log/Running.log', 'a') as f:
    f.writelines(f'Running {args.model_name} at {current_time}\n')

def set_quant_state(model, quant=False, clamp=False, search=False, spark=False):
    for name, module in model.named_modules():
        if isinstance(module, quantLinear):
            module.enable_quant(quant, clamp, search, spark)
config = AutoConfig.from_pretrained(model_path)
config.use_cache = False  # ✅ 显式修改
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # '/cephfs/juxin/QwT-LLM/llm-qwt/log/Qwen2.5-0.5B/trainer_save',
    torch_dtype=torch.bfloat16,
    # device_map="cuda:0",
    device_map="auto",
    config=config
)

if args.eval_base:
    print(f'---eval {args.model_name} base---')
    if args.save_tensor:
        set_save_tensor_enable()
        ppl = evaluate(model, test_data, args.n_samples)
        print(f'base {args.model_name} PPL: {ppl}')
        saved_name = args.model_name.replace("-", "_").replace(".", "_")
        dir=f'/cephfs/shared/juxin/saved_tensor/qwt/{saved_name}_base'
        os.makedirs(dir, exist_ok=True)
        save_tensors(dir=dir)
    else:
        with open(f'log/{args.model_name}/structure_base.txt', 'w') as f:
            f.writelines(f'{type(model)}\n\n{model}')
        ppl = evaluate(model, test_data, args.n_samples)
        print(f'base {args.model_name} PPL: {ppl}')
        with open(f'log/{args.model_name}/ppl_{args.task}.txt', 'a') as f:
            f.writelines(f'base PPL: {ppl}\n')
    # with open('log/qwt_bench.py', 'a') as f:
    #     f.writelines(f']\n\n')
    
def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear) and name != 'lm_head':
        new_layer = quantLinear.set_param(module, name=name, wgt_nbit=args.wgt_nbit, act_nbit=args.act_nbit)
        _set_module(model, name, new_layer)

if args.eval_quant:
    print(f'---eval {args.model_name} quant---')
    with open('log/base_bench.py', 'a') as f:
        rename = args.model_name.replace("-", "_").replace(".", "_")
        f.writelines(f'{args.task}_{rename} = [\n')
    set_quant_state(model, quant=True, clamp=False)
    if args.save_tensor:
        set_save_tensor_enable()
        ppl = evaluate(model, test_data, args.n_samples)
        print(f'W{args.wgt_nbit}A{args.act_nbit} {args.model_name} PPL: {ppl}')
        saved_name = args.model_name.replace("-", "_").replace(".", "_")
        dir=f'/cephfs/shared/juxin/saved_tensor/w4a8/{saved_name}_quant_{args.task}'
        os.makedirs(dir, exist_ok=True)
        save_tensors(dir=dir)
    else:
        with open(f'log/{args.model_name}/structure_quant.txt', 'w') as f:
            f.writelines(f'{type(model)}\n\n{model}')
        ppl = evaluate(model, test_data, args.n_samples)
        print(f'W{args.wgt_nbit}A{args.act_nbit} {args.model_name} PPL: {ppl}')
        with open(f'log/{args.model_name}/ppl_{args.task}.txt', 'a') as f:
            f.writelines(f'W{args.wgt_nbit}A{args.act_nbit} PPL: {ppl}\n')
    with open('log/base_bench.py', 'a') as f:
        f.writelines(f']\n\n')
if args.eval_clamp:
    print(f'---eval clamp---')
    set_clamp_block_size(args.clamp_block_size)
    set_clamp_type(args.clamp_type)
    # with open(f'log/{args.model_name}/structure_clamp.txt', 'w') as f:
    #     f.writelines(f'\n{type(model)}\n\n{model}')


    clamp_type_list = ['base', 'int5', 'int5 int6']
    clamp_type_list = ['int5 int6']
    set_clamp_type('int5 int6')
    mse_list = [10000, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001, 0.00008, 0.00006, 0.00005, 0.00004, 0.00003, 0.00002, 0.00001]
    mse_list = [0.001]
    for mse in mse_list:
        with open('log/mse.txt', 'a') as f:
            f.writelines(f'\n>>> {args.model_name} at {mse}\n')
        set_mse(mse)
        reset_counter()
        set_quant_state(model, quant=True, clamp=True, search=True)
        _ = evaluate(model, test_data, 10, comment=' Clamp Thresholds Searching...')
        set_quant_state(model, quant=True, clamp=True, search=False)
        ppl = evaluate(model, test_data, args.n_samples, comment="Evaluating...")
        print(f'mse: {mse}, PPL: {ppl}')
        int4, int5, int6, int8, total = get_ratio()
        print(f'clamp ratio: int5 {int5}/{total} ({int5/total:.4f}), int6 {int6}/{total} ({int6/total:.4f})')
        if args.n_samples is None:
            with open(f'log/{args.model_name}/ppl_{args.task}.txt', 'a') as f:
                f.writelines(f'CBS={args.clamp_block_size}, mse={mse:.5f}, ppl: {ppl:.4f} ' \
                             f'clamp ratio: (int4, int5, int6, int8): {int4/total:.4f}, {int5/total:.4f}, {int6/total:.4f}, {int8/total:.4f}, {(int4+int5+int6)/total:.4f}\n')
if args.eval_spark:
    print(f'---eval spark---')
    set_quant_state(model, quant=True, spark=True)
    ppl = evaluate(model, test_data, args.n_samples, comment="Evaluating...")
    print(f'PPL: {ppl}')
    with open(f'log/{args.model_name}/ppl_{args.task}.txt', 'a') as f:
        f.writelines(f'SPARK, ppl: {ppl:.4f}\n')
        
if args.layer_wise_delta:
    print(f'---eval layer_wise_delta---')
    set_clamp_block_size(args.clamp_block_size)
    set_clamp_type(args.clamp_type)

    set_clamp_type('int5 int6')
    mse_list = [0.001]
    for mse in mse_list:
        set_mse(mse)
        reset_counter()
        set_quant_state(model, quant=True, clamp=True, search=True)
        _ = evaluate(model, test_data, 10, comment=' Clamp Thresholds Searching...')

    with open('log/delta_int5.txt', 'w') as f5, open('log/delta_int6.txt', 'w') as f6:
        pass
    delta_int5 = {}
    delta_int6 = {}
    for name, module in model.named_modules():
        if isinstance(module, quantLinear):
            mse_int5_th = module.mse_int5_th
            mse_int6_th = module.mse_int6_th
            name = module.name
            delta_int5[name] = mse_int5_th.int()
            delta_int6[name] = mse_int6_th.int()
            with open('log/delta_int5.txt', 'a') as f5, open('log/delta_int6.txt', 'a') as f6:
                f5.writelines(f'>> delta_int5: {mse_int5_th:.6f}, {name}\n')
                f6.writelines(f'>> delta_int6: {mse_int6_th:.6f}, {name}\n')
    with open(f'log/{args.model_name}/delta_key.py', 'w') as f:
        f.writelines(f"delta_key = [")
        for k in delta_int5.keys():
            f.writelines(f"'{k}',\n")
        f.writelines(f"]")
    save_file(delta_int5, f"log/{args.model_name}/delta_int5.safetensors")
    save_file(delta_int6, f"log/{args.model_name}/delta_int6.safetensors")

# ----------------------------------------------------------
end_time = time.time()
duration = end_time - start_time
hour = duration // 3600
minute = (duration % 3600) // 60
second = duration % 60
print(f'>>> RUNNING TIME: {int(hour)}h-{int(minute)}m-{int(second)}s\n')
with open(f'log/{args.model_name}/ppl_{args.task}.txt', 'a') as f:
    f.writelines(f'>>> RUNNING TIME {args.task}: {int(hour)}h-{int(minute)}m-{int(second)}s  {current_time}\n\n')