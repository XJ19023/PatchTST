'''
搜索内容：
    1. linear层是否需要smooth, alpha值
    2. scale粒度：vector size
    3. INT位宽：4/8 bits
    4. BFP位宽：4/8 bits，block size：16的倍数
'''

import argparse
import functools
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
from mycode.globalVar import save_tensors, increas_counter, get_counter, append_activation, save_tensors
import transformers.models.llama.modeling_llama
import mycode.quant_utils as quant_utils
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
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
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
    parser.add_argument('--mxquant', action='store_true', default=False, help='registe hooks')
    parser.add_argument('--intquant', action='store_true', default=False, help='registe hooks')
    parser.add_argument('--n_bits', type=int, default=8, help='test samples, 0 means full samples')
    parser.add_argument(
    '--smooth_module',
    type=str,
    default='qkv',
    help='Modules to apply smooth'
)
#     parser.add_argument(
#     '--smooth_module',
#     nargs='+',
#     default=['qkv', 'to_out', 'ff.0', 'ff.3'],
#     help='Modules to apply smooth'
# )
    args = parser.parse_args()

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
            increas_counter()
        append_activation(f'case{get_counter()}_{name}', x)
        # print(f'case{get_counter()}_{name}', x.shape, module.weight.shape)
        # with open(f'process_flow.txt', 'a') as f:
        #     f.write(f"{name} {m}\n")

    Exp = Exp_Main
    exp = Exp(args)
    model = exp.model

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


    smooth_module = []
    smooth_factor = {}
    if args.smooth:
        smooth_module = args.smooth_module.split()
        loggings = f'smooth_module={smooth_module}, alpha={args.alpha}, '
        from mysmoothquant.smooth import smooth_lm
        act_scales = torch.load('act_scales/patchTST.pt')
        smooth_lm(model, act_scales, args.alpha, smooth_module, smooth_factor)
    # smooth_module = []
    
    '''
    if args.mxquant:
        def _set_module(model, submodule_key, module):
            tokens = submodule_key.split('.')
            sub_tokens = tokens[:-1]
            cur_mod = model
            for s in sub_tokens:
                cur_mod = getattr(cur_mod, s)
            setattr(cur_mod, tokens[-1], module)

        mx_specs = set_mx_specs(block_size=args.block_size, w_elem_format=args.w_elem_format, a_elem_format=args.a_elem_format, acc_bits=args.acc_bits)
        loggings += f"mx_specs:{mx_specs['block_size']}, {mx_specs['w_elem_format']}, "
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                new_layer = mxLinear.set_param(module, mx_specs=mx_specs, name=name, smooth_module=smooth_module)
                _set_module(model, name, new_layer)
                
    if args.intquant:
        loggings += f'intquant:{args.n_bits}, '
        model = quantize_model(
            model,
            weight_quant="per_channel",
            act_quant="per_token",
            quantize_bmm_input=False,
            n_bits = args.n_bits,
            smooth_module = smooth_module,
        )
    '''

    quant_utils.add_quant(model)
    qlayers = quant_utils.find_qlayers(model)

    for name in qlayers:
        qlayers[name].name = name
        qlayers[name].quant_meth = 'int'
        qlayers[name].n_bits = 4
        qlayers[name].search = True
        qlayers[name].y_mse_quant_mean = 0
        qlayers[name].y_mse_smoothquant_mean = 0
        qlayers[name].n_samples = 64
        qlayers[name].iters = 0

        key = None
        if 'qkv' in smooth_module and name.endswith(('W_Q', 'W_K', 'W_V')):
            key = name[:41] + '.W_Q'
        if name.endswith(tuple(smooth_module)): 
            key = name
        if key is not None:
            qlayers[name].smooth_factor = smooth_factor[key]

    mse, _ = exp.test(setting, test=1, n_samples=64)
    print(mse)
    exit()

    if args.hook:
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and 'head' not in name and 'W_P' not in name:
                hooks.append(
                    module.register_forward_hook(functools.partial(stat_input_hook, name=name))
                )

    # print(model)
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    mse, mae = exp.test(setting, test=1, n_samples=args.n_samples)
    torch.cuda.empty_cache()

    print('MSE: {}, MAE: {}'.format(mse, mae))
    with open(f"logs/{args.model_id}.txt", 'a') as f:
        f.write(f"mse:{mse:.8}, {loggings}")
        f.write('\n')
        

    # save_tensors(dir=f'save_tensors/org')