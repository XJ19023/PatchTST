import functools
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
# from smoothquant.smooth import smooth_lm
# from smoothquant.fake_quant import quantize_model
import tqdm

from datasets import load_dataset
import argparse
import mycode.quant_utils as quant_utils


parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument(
    "--act_scales_path",
    type=str,
    default="act_scales/llama-2-7b.pt",
)
parser.add_argument("--n_samples", type=int, default=None)
parser.add_argument("--smooth", action="store_true")
parser.add_argument("--quant", action="store_true")
parser.add_argument("--hook", action="store_true")


args = parser.parse_args()
alpha = args.alpha
model_path = f'/localssd/lbxj/{args.model_name}'
# act_scales_path = f'act_scales/{args.model_name}.pt'
n_samples = args.n_samples

def stat_input_hook(module, x, y, name):
    if isinstance(x, tuple):
        x = x[0]
    print(f'{name[21:]:<25} W: {module.weight.shape} A: {x.shape}')

class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=40):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)

        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        n_samples = self.n_samples if self.n_samples else self.dataset.size(1) // 2048
        for i in tqdm.tqdm(range(n_samples), desc="Evaluating..."):
            batch = self.dataset[:, (i * 2048) : ((i + 1) * 2048)].to(model.device)
            with torch.no_grad():
                lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)

        return torch.exp(torch.stack(nlls).sum() / (n_samples * 2048))


tokenizer = AutoTokenizer.from_pretrained(model_path)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
evaluator = Evaluator(dataset, tokenizer, "cuda", n_samples=n_samples)

model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto"
)
'''
if args.hook:
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(
                module.register_forward_hook(functools.partial(stat_input_hook, name=name))
            )
'''

quant_utils.add_quant(model, skip_names=('lm_head',))
qlayers = quant_utils.find_qlayers(model)

for n_bits in [8, 4]:
    int_specs = {'n_bits': n_bits}
    cfg = quant_utils.QuantConfig('int', int_specs, False, None)
    for name in qlayers:
        qlayers[name].name = name
        qlayers[name].set_quant_config(cfg)


    ppl = evaluator.evaluate(model)
    print(f"Perplexity: {ppl}")
    os.makedirs(f'logs/{args.model_name}', exist_ok=True)


    with open(f'logs/{args.model_name}/ppl.txt', 'a') as f:
        f.writelines(f'ppl: {ppl:.6f} {int_specs}\n')
