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
from mx.quant_mx_specs import set_mx_specs
from collections import defaultdict


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
parser.add_argument("--org", action="store_true")
parser.add_argument("--separate", action="store_true")
parser.add_argument("--search", action="store_true")


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
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)


    @torch.no_grad()
    def evaluate(self, model, print_en=True, log_en=False, n_samples=None, info=''):
        model.eval()
        nlls = []
        n_samples = n_samples if n_samples else self.dataset.size(1) // 2048
        for i in tqdm.tqdm(range(n_samples), desc=f"Evaluating {info}..."):
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
        ppl = torch.exp(torch.stack(nlls).sum() / (n_samples * 2048))

        if print_en:
            print(f"Perplexity: {ppl:.8f}")

        if log_en:
            os.makedirs(f'logs/{args.model_name}', exist_ok=True)
            with open(f'logs/{args.model_name}/ppl.txt', 'a') as f:
                f.writelines(f'ppl: {ppl:.8f}, {loggings}\n')
        return ppl


tokenizer = AutoTokenizer.from_pretrained(model_path)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
evaluator = Evaluator(dataset, tokenizer, "cuda")

model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto"
)

os.makedirs(f'logs/{args.model_name}/model_structure', exist_ok=True)
# with open(f'logs/{args.model_name}/model_structure/orginal_model.txt', 'w') as f:
#     f.write(f'>>> Original Model <<<\n')
#     f.write(str(model) + '\n\n')

'''
if args.hook:
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(
                module.register_forward_hook(functools.partial(stat_input_hook, name=name))
            )
'''

if args.org:
    loggings = f'org'
    ppl = evaluator.evaluate(model, log_en=True)

if args.quant:
    quant_utils.add_quant(model, skip_names='lm_head')
    qlayers = quant_utils.find_qlayers(model)

    if args.separate: 
        cal_mse_space = []
        # int quant
        for n_bits in [8, 4]:
            int_specs = {'n_bits': n_bits}
            cal_mse_space.append(quant_utils.QuantConfig('int', int_specs, None))
        # mx quant
        for elem_format in ['int8', 'int4']:
            mx_specs = set_mx_specs(block_size=16, w_elem_format=elem_format, a_elem_format=elem_format)
            cal_mse_space.append(quant_utils.QuantConfig('mx', mx_specs, None))

        for idx, cfg in enumerate(cal_mse_space):
            for name in qlayers:
                qlayers[name].name = name
                qlayers[name].set_quant_config(cfg)

            if idx == 0:
                loggings = f'int8'
            if idx == 1:
                loggings = f'int4'
            if idx == 2:
                loggings = f'BFP8'
            if idx == 3:
                loggings = f'BFP4'
            evaluator.evaluate(model, log_en=True, info=loggings)
    with open(f'logs/{args.model_name}/model_structure/quant_model.txt', 'w') as f:
        f.write(f'>>> Original Model <<<\n')
        f.write(str(model) + '\n\n')
    exit()

    if args.search:
        print('----------Step1: calculate baseline---------------')
        search_samples = 40
        for name in qlayers:
            qlayers[name].name = name
            qlayers[name].step_flag = 1  # 1 for int8 mse as baseline
            qlayers[name].n_samples = search_samples
            cfg = quant_utils.QuantConfig('int', {'n_bits': 8}, None)
            qlayers[name].set_quant_config(cfg)

        ppl_int8 = evaluator.evaluate(model, n_samples=search_samples) # use int8 as baseline
        ppl_th = ppl_int8 * 1.0001
        ppl_th = 0.2303
        print(f'ppl_int8: {ppl_int8:.8f}')
        print(f'ppl_th: {ppl_th:.8f}')
        with open('logs/model_structure/step1.txt', 'w') as f:
            f.write(f'>>> Step1 Model <<< calculate INT8 as baseline\n')
            f.write(str(model) + '\n\n')
        
        print('----------Step2: int8 layers to 4 bits---------------')
        layer_mse = {}
        for name in qlayers:
            layer_mse[name] = qlayers[name].y_mse_quant_mean
        layer_mse_sorted = dict(sorted(layer_mse.items(), key=lambda item: item[1]))

        left, right = 1, len(layer_mse_sorted) + 1 # [left, right)
        while (right > left):
            mid = (right + left) // 2
            print(left, right, mid, len(layer_mse_sorted), end=' ')

            to_4bits = dict(list(layer_mse_sorted.items())[:mid])
            keep_int8 = dict(list(layer_mse_sorted.items())[mid:])
            cfg = quant_utils.QuantConfig('int', {'n_bits': 8}, None)
            for name in keep_int8.keys():
                qlayers[name].step_flag = -1 # keep as int8
                qlayers[name].set_quant_config(cfg)
            mx_specs = set_mx_specs(block_size=16, w_elem_format='int4', a_elem_format='int4')
            cfg = quant_utils.QuantConfig(None, mx_specs, None) # deliver mx_specs
            for name in to_4bits.keys():
                qlayers[name].step_flag = 2 # search for int4 or BFP4
                qlayers[name].set_quant_config(cfg) # deliver mx_specs
            
            _ = evaluator.evaluate(model, n_samples=search_samples, print_en=False) # search for int4 or BFP4
            ppl_tmp = evaluator.evaluate(model, n_samples=search_samples)

            if ppl_tmp < ppl_th:
                left = mid + 1
                if ppl_th == 1000:
                    ppl_th = ppl_tmp
            else:
                right = mid

        left -= 1
        keep_int8 = dict(list(layer_mse_sorted.items())[left:])
        cfg = quant_utils.QuantConfig('int', {'n_bits': 8}, None)
        for name in keep_int8.keys():
            qlayers[name].step_flag = -1 # keep as int8
            qlayers[name].set_quant_config(cfg)
        to_4bits = dict(list(layer_mse_sorted.items())[:left])
        mx_specs = set_mx_specs(block_size=16, w_elem_format='int4', a_elem_format='int4')
        cfg_mx = quant_utils.QuantConfig('mx', mx_specs, None)
        cfg_int = quant_utils.QuantConfig('int', {'n_bits': 4}, None)
        for name in to_4bits.keys():
            qlayers[name].step_flag = -2 # 2 for replaced 4 bits
            if qlayers[name].using_BFP4:
                qlayers[name].set_quant_config(cfg_mx) # deliver mx_specs
            else:
                qlayers[name].set_quant_config(cfg_int) # deliver mx_specs
        
        loggings = f'step2: INT8 to 4 bits, mse_th = {ppl_th}'
        ppl_4bits = evaluator.evaluate(model, log_en=True, print_en=False)
        print(f'ppl_4bits: {ppl_4bits:.8f}')

        with open('logs/model_structure/step2.txt', 'w') as f:
            f.write(f'>>> Step2 Model <<< INT8 to 4 bits\n')
            f.write(str(model) + '\n\n')

        print('----------Step3: left int8 layers to BFP8---------------')
        mse_th = 0.3266
        layer_mse = {}
        for name in qlayers:
            if qlayers[name].step_flag == -1: # frize 4 btis
                layer_mse[name] = qlayers[name].y_mse_quant_mean
        layer_mse_sorted = dict(sorted(layer_mse.items(), key=lambda item: item[1]))

        left, right = 1, len(layer_mse_sorted) + 1
        while (right > left):
            if len(layer_mse_sorted) == 0:
                break
            mid = (right + left) // 2
            print(left, right, mid, len(layer_mse_sorted), end=' ')

            keep_int8 = dict(list(layer_mse_sorted.items())[mid:])
            cfg = quant_utils.QuantConfig('int', {'n_bits': 8}, None)
            for name in keep_int8.keys():
                qlayers[name].step_flag = -1 # keep as int8
                qlayers[name].set_quant_config(cfg)
            to_BFP8 = dict(list(layer_mse_sorted.items())[:mid])
            mx_specs = set_mx_specs(block_size=16, w_elem_format='int8', a_elem_format='int8')
            cfg = quant_utils.QuantConfig('mx', mx_specs, None)
            for name in to_BFP8.keys():
                qlayers[name].step_flag = 3 # 3 for replaced BFP8 
                qlayers[name].set_quant_config(cfg)
            
            mse_tmp =evaluator.evaluate(model, n_samples=search_samples)
            if mse_tmp < mse_th:
                left = mid + 1
                if mse_th == 1000:
                    mse_th = mse_tmp
            else:
                right = mid

        left -= 1
        keep_int8 = dict(list(layer_mse_sorted.items())[left:])
        cfg = quant_utils.QuantConfig('int', {'n_bits': 8}, None)
        for name in keep_int8.keys():
            qlayers[name].step_flag = -1 # keep as int8
            qlayers[name].set_quant_config(cfg)
        to_BFP8 = dict(list(layer_mse_sorted.items())[:left])
        mx_specs = set_mx_specs(block_size=16, w_elem_format='int8', a_elem_format='int8')
        cfg = quant_utils.QuantConfig('mx', mx_specs, None)
        for name in to_BFP8.keys():
            qlayers[name].step_flag = -3 # 3 for replaced BFP8 
            qlayers[name].set_quant_config(cfg)
        
        loggings = f'step3: left INT8 to BFP8, mse_th = {mse_th}'
        mse_BFP8 = evaluator.evaluate(model, log_en=True, print_en=False)
        print(f'mse_BFP8: {mse_BFP8:.8f}')
        with open('logs/model_structure/step3.txt', 'w') as f:
            f.write(f'>>> Step3 Model <<< left INT8 to BFP8\n')
            f.write(str(model) + '\n\n')

        print('----------Step4: enable smooth---------------')
        from mysmoothquant.smooth import smooth_lm
        smooth_factors = defaultdict(list)
        act_scales = torch.load(f'act_scales/{args.model_id}.pt')
        alphas = [i / 10 for i in range(1, 10)]
        for alpha in alphas:
            smooth_lm(model, act_scales, alpha, smooth_factors)

        for name in qlayers:
            qlayers[name].y_mse_smoothquant_mean = [0 for _ in alphas]
            qlayers[name].step_flag = 4  # search smooth

            key = None
            if name.endswith(('W_Q', 'W_K', 'W_V')):
                key = name[:41] + '.W_Q'
            else: 
                key = name
            qlayers[name].smooth_factors = smooth_factors[key]
        evaluator.evaluate(model, n_samples=search_samples)
        
        for name in qlayers:
            quant_mse = qlayers[name].y_mse_quant_mean
            smoothquant_mse = qlayers[name].y_mse_smoothquant_mean
            min_idx, min_val = min(enumerate(smoothquant_mse), key=lambda x: x[1])
            # print(f'{quant_mse:.8f}, {min_val:.8f}, {min_val < quant_mse}')
            if min_val < quant_mse:
                qlayers[name].smooth_factor = qlayers[name].smooth_factors[min_idx]
                qlayers[name].cfg.alpha = (min_idx + 1) / 10

        loggings = 'step4: enable smooth \n'
        mse_smooth =evaluator.evaluate(model, log_en=True, print_en=False)
        print(f'mse_smooth: {mse_smooth:.8f}')
        with open('logs/model_structure/step4.txt', 'w') as f:
            f.write(f'>>> Step4 Model <<< enable smooth\n')
            f.write(str(model) + '\n\n')

        print('----------final quantized model---------------')

