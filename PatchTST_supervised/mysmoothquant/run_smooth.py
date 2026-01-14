'''
搜索内容：
    1. linear层是否需要smooth, alpha值
    2. scale粒度：vector size
    3. INT位宽：4/8 bits
    4. BFP位宽：4/8 bits，block size：16的倍数
'''

import argparse
import functools
import json
import math
import os
import sys
sys.path.append('.')
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

from mysmoothquant.fake_quant import W8A8Linear, quantize_model
from mx import mxLinear
from mx.quant_mx_specs import set_mx_specs
from mycode.globalVar import save_tensors, increas_counter, get_counter, append_activation, save_tensors, reset_activation
import transformers.models.llama.modeling_llama
import mycode.quant_utils as quant_utils
from collections import defaultdict

from data_provider.data_factory import data_provider
from torch.utils.data import DataLoader, Subset


def export_quant_config(save_path):
    qcfg = {}
    for name in qlayers:
        qlayers[name].save_quant_cfg()
        qcfg[name] = qlayers[name].quant_cfg

    with open(save_path, "w") as f:
        json.dump(qcfg, f, indent=2)

def load_quant_config(cfg_path):
    with open(cfg_path) as f:
        qcfg = json.load(f)
    for name in qlayers:
        if qcfg[name]['quant_meth'] == 'int':
            cfg = quant_utils.QuantConfig('int', {'n_bits': qcfg[name]['quant_bits']}, None)
            qlayers[name].set_quant_config(cfg)
        if qcfg[name]['quant_meth'] == 'mx':
            mx_specs = set_mx_specs(block_size=16, w_elem_format=f"int{qcfg[name]['quant_bits']}", a_elem_format=f"int{qcfg[name]['quant_bits']}")
            cfg = quant_utils.QuantConfig('mx', mx_specs, None) # deliver mx_specs
            qlayers[name].set_quant_config(cfg)


def get_data(flag, args):
    data_set, data_loader = data_provider(args, flag)
    return data_set, data_loader

def select_best_index(vec, th):
    A, B, C = torch.tensor(vec[0], device='cuda'), torch.tensor(vec[1], device='cuda'), torch.tensor(vec[2], device='cuda')
    a, b, c = th[0], th[1], th[2]
    # print(A < a)
    # print(B < b)
    # print(C < c)
    mask = (A < a) & (B < b) & (C < c)

    if mask.sum() == 0:
        return None

    mean_vals = (A + B + C) / 3
    mean_vals_masked = mean_vals.clone()
    mean_vals_masked[~mask] = float('inf')

    best_idx = torch.argmin(mean_vals_masked).item()
    if best_idx is not None:
        # print(f'{A[best_idx]:.6f}, {(best_idx + 1) / 10}, {a-A[best_idx]:.6f}, {a/A[best_idx]:.6f}')
        # print(f'{B[best_idx]:.6f}, {(best_idx + 1) / 10}, {b-B[best_idx]:.6f}, {b/B[best_idx]:.6f}')
        # print(f'{C[best_idx]:.6f}, {(best_idx + 1) / 10}, {c-C[best_idx]:.6f}, {c/C[best_idx]:.6f}')
        if a/A[best_idx] < 1.01 or b/B[best_idx] < 1.01 or c/C[best_idx] < 1.01: 
            best_idx = None
    return best_idx

def evaluate(n_samples=None, test_loader=None, log_en=False, print_en=True):
    if test_loader is None:
        _, test_loader = get_data(flag='test', args=args) # default for evaluate
    mse, mae = exp.test(setting, test=1, n_samples=n_samples, test_loader=test_loader) # inference
    torch.cuda.empty_cache()
    if print_en:
        print('MSE: {}, MAE: {}'.format(mse, mae))
    if log_en:
        with open(f"logs/{args.model_id}/result.txt", 'a') as f:
            f.write(f"mse:{mse:.8f}, {loggings}")
            # f.write('\n')
    return mse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Informer, Transformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


    # DLinear
    #parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # Formers 
    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=2, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    parser.add_argument('--w_elem_format', type=str, default='int8', help='gpu')
    parser.add_argument('--a_elem_format', type=str, default='int8', help='gpu')
    parser.add_argument('--block_size', type=int, default=0, help='0 means no quantization')
    parser.add_argument('--acc_bits', type=int, default=0, help='0 means default accumulation bits')
    parser.add_argument('--n_samples', type=int, default=0, help='test samples, 0 means full samples')
    parser.add_argument('--hook', action='store_true', default=False, help='registe hooks')
    parser.add_argument('--smooth', action='store_true', default=False, help='registe hooks')
    parser.add_argument('--alpha', type=float, default=0.5, help='test samples, 0 means full samples')
    parser.add_argument('--quant', action='store_true', default=False, help='registe hooks')
    parser.add_argument('--n_bits', type=int, default=8, help='test samples, 0 means full samples')
    parser.add_argument('--smooth_n_samples', type=int, default=1, help='test samples, 0 means full samples')
    parser.add_argument('--org', action='store_true', default=False, help='registe hooks')
    parser.add_argument('--separate', action='store_true', default=False, help='registe hooks')
    parser.add_argument('--search', action='store_true', default=False, help='registe hooks')
    parser.add_argument('--load_cfg', action='store_true', default=False, help='registe hooks')
    parser.add_argument('--powersmooth', action='store_true', default=False, help='registe hooks')

    args = parser.parse_args()
    torch.cuda.reset_peak_memory_stats()

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)


    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    # print(args)
    loggings = ''


    def stat_input_hook(module, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        if name == 'model.backbone.encoder.layers.0.self_attn.W_Q':
            if get_counter() != 0:
                dir_path = f'logs/{args.model_id}/fp32_tensor/case{get_counter()}'
                os.makedirs(dir_path, exist_ok=True)
                save_tensors(dir=dir_path)
                reset_activation()
            increas_counter()
        append_activation(name, x)
        # print(f'case{get_counter()}_{name}', x.shape, module.weight.shape)
        # with open(f'process_flow.txt', 'a') as f:
        #     f.write(f"{name} {m}\n")

    Exp = Exp_Main
    exp = Exp(args)
    model = exp.model

    import os
    os.makedirs(f'logs/{args.model_id}', exist_ok=True)
    os.makedirs(f'logs/output', exist_ok=True)
    with open(f'logs/{args.model_id}/orginal_model.txt', 'w') as f:
        f.write(f'>>> Original Model <<<\n')
        f.write(str(model) + '\n\n')

    ii = 0
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.model_id))
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                args.model,
                                                                                                args.data,
                                                                                                args.features,
                                                                                                args.seq_len,
                                                                                                args.label_len,
                                                                                                args.pred_len,
                                                                                                args.d_model,
                                                                                                args.n_heads,
                                                                                                args.e_layers,
                                                                                                args.d_layers,
                                                                                                args.d_ff,
                                                                                                args.factor,
                                                                                                args.embed,
                                                                                                args.distil,
                                                                                                args.des, ii)

    print('loading model')
    model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))


    # args.batch_size = 128
    data_set, _ = get_data(flag='test', args=args) # default batch_size for smooth search
    cal_mse_samples = min(32, len(data_set)//args.batch_size)
    num_samples = cal_mse_samples * args.batch_size
    indices = np.random.choice(len(data_set), num_samples, replace=False)
    # 创建一个Subset对象，这样你就只会从数据集里得到这100个样本
    subset = Subset(data_set, indices)
    # 使用这个Subset创建一个新的DataLoader
    dataset_for_search_format = DataLoader(subset, batch_size=args.batch_size, shuffle=False)

    # for idx, item in enumerate(dataset_for_search_format):
    #     print(idx, item[0].shape)
    # print(indices)
    # exit()


    if args.org:

        if args.hook:
            hooks = []
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear) and 'head' not in name and 'W_P' not in name:
                    hooks.append(
                        module.register_forward_hook(functools.partial(stat_input_hook, name=name))
                    )

        loggings = f'org'
        evaluate(log_en=True, n_samples=args.n_samples)

        

    if args.quant:
        quant_utils.add_quant(model)
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            qlayers[name].name = name

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
                    qlayers[name].set_quant_config(cfg)

                if idx == 0:
                    loggings = f'int8'
                if idx == 1:
                    loggings = f'int4'
                if idx == 2:
                    loggings = f'BFP8'
                if idx == 3:
                    loggings = f'BFP4'
                evaluate(log_en=True, n_samples=args.n_samples)

        if args.search:
            if args.load_cfg:
                load_quant_config(f'logs/{args.model_id}/quant_cfg.json')
                # loggings = f'step3: left INT8 to BFP8, load cfg'
                # mse_BFP8 = evaluate(log_en=True, print_en=False)
                # print(f'mse_BFP8: {mse_BFP8:.8f}')
                # with open(f'logs/{args.model_id}/load.txt', 'w') as f:
                #     f.write(f'>>> Step3 Model <<< left INT8 to BFP8\n')
                #     f.write(str(model) + '\n\n')
            else:  
                mse_th_step2, mse_th_step3 = 1000, 1000
                print('----------Step1: calculate baseline---------------')
                for name in qlayers:
                    qlayers[name].step_flag = 1  # 1 for int8 mse as baseline
                    qlayers[name].n_samples = cal_mse_samples
                    cfg = quant_utils.QuantConfig('int', {'n_bits': 8}, None)
                    qlayers[name].set_quant_config(cfg)

                mse_int8 = evaluate(test_loader=dataset_for_search_format) # use int8 as baseline
                mse_th = mse_int8 * 1.0001
                print(f'mse_int8: {mse_int8:.8f}')
                print(f'mse_th: {mse_th_step2:.8f}')
                with open(f'logs/{args.model_id}/step1.txt', 'w') as f:
                    f.write(f'>>> Step1 Model <<< calculate INT8 as baseline\n')
                    f.write(str(model) + '\n\n')
                
                print('----------Step2: int8 layers to 4 bits---------------')
                layer_mse = {}
                for name in qlayers:
                    layer_mse[name] = qlayers[name].y_mse_quant_mean
                layer_mse_sorted = dict(sorted(layer_mse.items(), key=lambda item: item[1]))

                left, right = 1, len(layer_mse_sorted) + 1 # [left, right)
                left, right = 1, len(layer_mse_sorted) - 6 # [left, right)
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
                    
                    _ = evaluate(test_loader=dataset_for_search_format, print_en=False) # search for int4 or BFP4
                    mse_tmp = evaluate(test_loader=dataset_for_search_format)

                    if mse_tmp < mse_th_step2:
                        left = mid + 1
                        if mse_th_step2 == 1000:
                            mse_th_step2 = mse_tmp
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
                
                loggings = f'step2: INT8 to 4 bits, mse_th = {mse_th_step2}'
                mse_4bits = evaluate(log_en=True, print_en=False)
                print(f'mse_4bits: {mse_4bits:.8f}')

                with open(f'logs/{args.model_id}/step2.txt', 'w') as f:
                    f.write(f'>>> Step2 Model <<< INT8 to 4 bits\n')
                    f.write(str(model) + '\n\n')

                print('----------Step3: left int8 layers to BFP8---------------')
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
                    
                    mse_tmp = evaluate(test_loader=dataset_for_search_format)
                    if mse_tmp < mse_th_step3:
                        left = mid + 1
                        if mse_th_step3 == 1000:
                            mse_th_step3 = mse_tmp
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
            
                export_quant_config(f'logs/{args.model_id}/quant_cfg.json')
                loggings = f'step3: left INT8 to BFP8, mse_th = {0}'
                mse_BFP8 = evaluate(log_en=True, print_en=False)
                print(f'mse_BFP8: {mse_BFP8:.8f}')
                with open(f'logs/{args.model_id}/step3.txt', 'w') as f:
                    f.write(f'>>> Step3 Model <<< left INT8 to BFP8\n')
                    f.write(str(model) + '\n\n')

            mse_without_smooth = evaluate(log_en=False, print_en=False, n_samples=4)

            print('----------Step4: enable smooth---------------')
            num_samples = 1
            from mysmoothquant.smooth import smooth_lm
            smooth_factors = defaultdict(list)
            act_scales = torch.load(f'act_scales/{args.model_id}.pt')
            alphas = [i / 10 for i in range(1, 5)]
            # alphas = [i / 10 for i in np.arange(1, 5, 0.5)]
            for alpha in alphas:
                smooth_lm(model, act_scales, alpha, smooth_factors)

            for name in qlayers:
                qlayers[name].y_kl_smoothquant_mean = [0 for _ in alphas]
                qlayers[name].n_samples = num_samples
                qlayers[name].step_flag = 4  # search smooth
                qlayers[name].powersmooth = args.powersmooth

                key = None
                if name.endswith(('W_Q', 'W_K', 'W_V')):
                    key = name[:41] + '.W_Q'
                else: 
                    key = name
                qlayers[name].smooth_factors = smooth_factors[key]

            # args.batch_size = 128
            data_set, _ = get_data(flag='test', args=args) # default batch_size for smooth search
            indices = np.random.choice(len(data_set), num_samples*args.batch_size, replace=False)
            indices = torch.tensor(indices).reshape(num_samples, args.batch_size)
            for indice in indices:
                # 创建一个Subset对象，这样你就只会从数据集里得到这100个样本
                subset = Subset(data_set, indice)
                # 使用这个Subset创建一个新的DataLoader
                dataset_for_enable_smooth = DataLoader(subset, batch_size=args.batch_size, shuffle=False)

                _ = evaluate(test_loader=dataset_for_enable_smooth, print_en=False)
                _ = evaluate(test_loader=dataset_for_enable_smooth, print_en=False)
            
            for name in qlayers:
                if name.endswith(('W_Q', 'W_K', 'W_V')):
                    if name.endswith(('W_Q', )):
                        name_qkv = []
                        quant_kl_th = []
                        smoothquant_kl = []
                    name_qkv.append(name)
                    quant_kl_th.append(qlayers[name].y_kl_quant_mean)
                    smoothquant_kl.append(qlayers[name].y_kl_smoothquant_mean)
                    if name.endswith(('W_V', )):
                        alpha_idx = select_best_index(smoothquant_kl, quant_kl_th)
                        if alpha_idx is not None:
                            for layer_name in name_qkv:
                                qlayers[layer_name].smooth_factor = qlayers[layer_name].smooth_factors[alpha_idx]
                                qlayers[layer_name].cfg.alpha = (alpha_idx + 1) / 10
                else:
                    quant_kl = qlayers[name].y_kl_quant_mean
                    smoothquant_kl = qlayers[name].y_kl_smoothquant_mean
                    min_idx, min_val = min(enumerate(smoothquant_kl), key=lambda x: x[1])
                    if min_val < quant_kl:
                        # print(f'\n{quant_kl:.6f}, {name}')
                        # print(f'{min_val:.6f}, {(min_idx + 1) / 10}, {quant_kl-min_val:.6f}, {quant_kl/min_val:.6f}')
                        if quant_kl > min_val * 1.01:
                            qlayers[name].smooth_factor = qlayers[name].smooth_factors[min_idx]
                            qlayers[name].cfg.alpha = (min_idx + 1) / 10

            mse_with_smooth = evaluate(log_en=False, print_en=False, n_samples=4)
            print(f'mse_with_smooth   : {mse_with_smooth:.8f}')
            print(f'mse_without_smooth: {mse_without_smooth:.8f}')
            if mse_with_smooth >= mse_without_smooth:
                smooth_log = 'smooth disable\n'
            else:
                smooth_log = 'smooth enable \n'

            loggings = f"step4: enable smooth, smooth_mode={'powersmooth' if args.powersmooth else 'smoothquant'}, smooth_judge={smooth_log}"
            mse_smooth = evaluate(log_en=True, print_en=False)
            print(f'mse_smooth: {mse_smooth:.8f}')
            with open(f'logs/{args.model_id}/step4.txt', 'w') as f:
                f.write(f'>>> Step4 Model <<< enable smooth\n')
                f.write(str(model) + '\n\n')

            print('----------final quantized model---------------')
            print("Peak allocated:", torch.cuda.max_memory_allocated() / 1024**3, "GB")

            with open('logs/123.log', 'a') as f:
                f.write(f'{args.model_id}\n')


        
    # dir_path = f'logs/{args.model_id}/fp32_tensor'
    # os.makedirs(dir_path, exist_ok=True)
    # save_tensors(dir=dir_path)