import torch
from functools import partial
from dataclasses import dataclass, replace 
from mx.mx_ops import quantize_mx_op
from mx.elemwise_ops import quantize_elemwise_op

from mycode.globalVar import append_activation, append_weight

@dataclass
class QuantConfig:
    quant_meth: str          # int8 / int4 / fp8 / bfp
    quant_specs: dict
    alpha: int # idx

class QuantWrapper(torch.nn.Module):
    '''
        This class is a wrapper for the activation quantization.
        We extract the FP features in the forward pass and quantize the rest using
        the self.quantizer object.
        If a rotation Q is provided, the weight matrix will be rotated,
        a pre-forward hook will be registerd to rotate the activation before quantization.
    '''

    def __init__(self, module:torch.nn.Linear):
        super(QuantWrapper, self).__init__()
        assert isinstance(module, torch.nn.Linear)
        self.weight = module.weight
        self.bias = module.bias

        self.name = False
        self.smooth_factors = None
        self.smooth_factor = None
        self.loss_func = mse # to select quant data format
        self.smooth_loss_func = mse # to select smooth en
        self.n_samples = None
        self.step_flag = None
        self.y_mse_quant_mean = 0 # calculate once, use for quant order
        self.y_kl_quant_mean = 0 # use for smooth
        # self.y_mse_smoothquant_mean = None 
        self.iters = 0
        self.using_BFP4 = False
        self.store_fp32 = True
        self.y_fp32 = None
        self.powersmooth = False


        self.quant_cfg = {'quant_meth': None, 'quant_bits': None, 'step_flag': None}

    def save_quant_cfg(self):
        self.quant_cfg['quant_meth'] = self.cfg.quant_meth
        if self.cfg.quant_meth == 'int':
            self.quant_cfg['quant_bits'] = self.cfg.quant_specs['n_bits']
        if self.cfg.quant_meth == 'mx':
            self.quant_cfg['quant_bits'] = int(self.cfg.quant_specs['w_elem_format'][-1])
        self.quant_cfg['step_flag'] = self.step_flag

    def set_quant_config(self, cfg: QuantConfig):
        self.cfg = replace(cfg)
        # self.x_mse_quant_mean = 0
        # self.x_mse_smoothquant_mean = 0
        # self.w_mse_quant_mean = 0
        # self.w_mse_smoothquant_mean = 0
        self.y_mse_intquant_mean = 0
        self.y_mse_mxquant_mean = 0
        self.iters = 0
        self.using_BFP4 = False
        if self.cfg.quant_meth == 'int':
            self.quant_func = per_token_quant
        if self.cfg.quant_meth == 'mx':
            self.quant_func = mx_quant
        # self._build_quant_params()
    def get_quant_config(self) -> QuantConfig:
        return self.cfg

    def extra_repr(self) -> str:
        str_ = 'quant_meth: '
        if self.cfg.quant_meth == 'int':
            str_ += f"INT{self.cfg.quant_specs['n_bits']}, "
        if self.cfg.quant_meth == 'mx':
            str_ += f"BFP{self.cfg.quant_specs['w_elem_format'][-1]}, "
        str_ += f"setp: {self.step_flag}, "
        str_ += f'smooth: {self.smooth_factor is not None}'
        if self.smooth_factor is not None:
            str_ += f', a: { self.cfg.alpha}'
        return str_

    def forward(self, x):
        if self.step_flag == 1: # calculate int8 mse as baseline
            self.iters += 1
            y_org = torch.functional.F.linear(x, self.weight, self.bias)
            x_quant = per_token_quant(x, {'n_bits': 8})
            w_quant = per_token_quant(self.weight, {'n_bits': 8})
            y_quant = torch.functional.F.linear(x_quant, w_quant, self.bias)
            y_mse_quant = self.loss_func(y_org, y_quant)
            self.y_mse_quant_mean += y_mse_quant
            if self.iters == self.n_samples: 
                self.y_mse_quant_mean /= self.n_samples
                self.step_flag = -1
            return y_org
        
        elif self.step_flag == 2: # replace int8 to 4 bits
            self.iters += 1
            y_org = torch.functional.F.linear(x, self.weight, self.bias)
            x_intquant = per_token_quant(x, {'n_bits': 4})
            w_intquant = per_token_quant(self.weight, {'n_bits': 4})
            y_intquant = torch.functional.F.linear(x_intquant, w_intquant, self.bias)
            y_mse_intquant = self.loss_func(y_org, y_intquant)

            x_mxquant = mx_quant(x, self.cfg.quant_specs)
            w_mxquant = mx_quant(self.weight, self.cfg.quant_specs)
            y_mxquant = torch.functional.F.linear(x_mxquant, w_mxquant, self.bias)
            y_mse_mxquant = self.loss_func(y_org, y_mxquant)
            self.y_mse_intquant_mean += y_mse_intquant
            self.y_mse_mxquant_mean += y_mse_mxquant
            if self.iters == self.n_samples: 
                self.y_mse_intquant_mean /= self.n_samples
                self.y_mse_mxquant_mean /= self.n_samples

                # print(f'{self.y_mse_intquant_mean:.8f}, {self.y_mse_mxquant_mean:.8f}, {self.y_mse_mxquant_mean / self.y_mse_intquant_mean:.2f}')

                if self.y_mse_intquant_mean * 1.2 >= self.y_mse_mxquant_mean:
                # if True:
                    self.using_BFP4 = True
                    self.cfg.quant_meth = 'mx'
                    self.cfg.quant_specs = self.cfg.quant_specs
                    self.quant_func = mx_quant
                else:
                    self.cfg.quant_meth = 'int'
                    self.cfg.quant_specs = {'n_bits': 4}
                    self.quant_func = per_token_quant
                self.step_flag = -2
                
            return y_org
            
        elif self.step_flag == 4:
            if self.store_fp32:
                # append_activation(f'{self.name}_fp32', x)
                # append_weight(f'{self.name}_fp32', self.weight)
                y_fp32 = torch.functional.F.linear(x, self.weight, self.bias)
                torch.save(y_fp32, f'logs/output/{self.name}.pt')
                self.store_fp32 = False
                return y_fp32
            else:
                y_fp32 = torch.load(f'logs/output/{self.name}.pt')
                self.store_fp32 = True
            
            # append_activation(f'{self.name}_quant', x)
            self.iters += 1
            x_quant = self.quant_func(x, self.cfg.quant_specs)
            w_quant = self.quant_func(self.weight, self.cfg.quant_specs)
            y_quant = torch.functional.F.linear(x_quant, w_quant, self.bias)
            y_kl_quant = self.smooth_loss_func(y_fp32, y_quant)
            self.y_kl_quant_mean += y_kl_quant
            if self.iters == self.n_samples: 
                self.y_kl_quant_mean /= self.n_samples

            x_kl_quant = self.smooth_loss_func(x, x_quant)
            w_kl_quant = self.smooth_loss_func(self.weight, w_quant)
            for alpha_idx, smooth_factor in enumerate(self.smooth_factors):
                smooth_factor =  smooth_factor.view(1, -1).to(device=x.device)

                if self.powersmooth:
                    power = torch.log2(smooth_factor + 1e-6).int()
                    power = 2 ** power
                    # residual = smooth_factor / (2.0 ** power)
                    x_smooth = x.div(power)
                    w_smooth = self.weight.mul(power)
                    x_smoothquant = self.quant_func(x_smooth, self.cfg.quant_specs)
                    w_smoothquant = self.quant_func(w_smooth, self.cfg.quant_specs)
                    y_smoothquant = torch.functional.F.linear(x_smoothquant, w_smoothquant, self.bias)

                    y_kl_smoothquant = self.smooth_loss_func(y_fp32, y_smoothquant)

                    # x_kl_smoothquant = self.smooth_loss_func(x_smooth, x_smoothquant)
                    # w_kl_smoothquant = self.smooth_loss_func(w_smooth, w_smoothquant)
                    # print(f'power: {self.name[23:]:<27}, ({x_kl_quant:.8f}, {x_kl_smoothquant:.8f}), ({w_kl_quant:.8f}, {w_kl_smoothquant:.8f}), ({y_kl_quant:.8f}, {y_kl_smoothquant:.8f}), {(alpha_idx+1)/10}, {y_kl_quant > y_kl_smoothquant}, {power.min().item()}, {power.max().item()}') 
                else:
                    x_smooth = x.div(smooth_factor)
                    w_smooth = self.weight.mul(smooth_factor)
                    x_smoothquant = self.quant_func(x_smooth, self.cfg.quant_specs)
                    w_smoothquant = self.quant_func(w_smooth, self.cfg.quant_specs)
                    y_smoothquant = torch.functional.F.linear(x_smoothquant, w_smoothquant, self.bias)

                    y_kl_smoothquant = self.smooth_loss_func(y_fp32, y_smoothquant)

                    # x_kl_smoothquant = self.smooth_loss_func(x_smooth, x_smoothquant)
                    # w_kl_smoothquant = self.smooth_loss_func(w_smooth, w_smoothquant)
                    # print(f'>>>>>: {self.name[23:]:<27}, ({x_kl_quant:.8f}, {x_kl_smoothquant:.8f}), ({w_kl_quant:.8f}, {w_kl_smoothquant:.8f}), ({y_kl_quant:.8f}, {y_kl_smoothquant:.8f}), {(alpha_idx+1)/10}, {y_kl_quant > y_kl_smoothquant}, {smooth_factor.min().item()}, {smooth_factor.max().item()}') 


                self.y_kl_smoothquant_mean[alpha_idx] += y_kl_smoothquant
                if self.iters == self.n_samples: 
                    self.y_kl_smoothquant_mean[alpha_idx] /= self.n_samples
                    self.step_flag = -4
            return y_quant
        
        # elif self.initial_weight:
        #     if self.cfg.smooth_en:
        #         smooth_factor =  self.smooth_factors[self.cfg.alpha_idx].view(1, -1).to(device=x.device)
        #         with torch.no_grad():
        #             self.weight.mul_(self.smooth_factor)

        #     self.weight = self.quant_func(self.weight, self.cfg.quant_specs, True)

        #     self.initial_weight = False
        #     # return torch.zeros(*x.shape[:-1], self.weight.size(0), device=x.device, dtype=x.dtype)
        #     return torch.zeros_like(x[..., :self.weight.size(0)])

        else:
            if self.powersmooth:
                x_smooth = x_quant = None
                w_smooth = w_quant = None
                power = 0
                if self.smooth_factor is not None:
                    smooth_factor =  self.smooth_factor.view(1, -1).to(device=x.device)
                    power = torch.log2(smooth_factor + 1e-6).int()
                    power = 2 ** power
                    # residual = smooth_factor / (2.0 ** power)
                    x_smooth = x.div(power)
                    w_smooth = self.weight.mul(power)

                x_quant = self.quant_func(x_smooth if x_smooth is not None else x, self.cfg.quant_specs)
                w_quant = self.quant_func(w_smooth if w_smooth is not None else self.weight, self.cfg.quant_specs)

                y = torch.functional.F.linear(x_quant, w_quant, self.bias)
            else:
                x_smooth = x_quant = None
                w_smooth = w_quant = None
                if self.smooth_factor is not None:
                    smooth_factor =  self.smooth_factor.view(1, -1).to(device=x.device)
                    x_smooth = x.div(smooth_factor)
                    w_smooth = self.weight.mul(smooth_factor)

                x_quant = self.quant_func(x_smooth if x_smooth is not None else x, self.cfg.quant_specs)
                w_quant = self.quant_func(w_smooth if w_smooth is not None else self.weight, self.cfg.quant_specs)

                y = torch.functional.F.linear(x_quant, w_quant, self.bias)
            return y
        


def add_quant(
    module,
    name='',
    layers=(torch.nn.Linear,),
    skip_names=('head', 'W_P')
):
    if isinstance(module, QuantWrapper):
        return

    for child_name, child in module.named_children():
        full_name = f"{name}.{child_name}" if name else child_name

        # 🚫 跳过指定模块
        if child_name in skip_names:
            continue

        # ✅ 直接替换
        if type(child) in layers:
            setattr(module, child_name, QuantWrapper(child))
        else:
            add_quant(child, full_name, layers, skip_names)

def find_qlayers(module, layers=[QuantWrapper], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlayers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

@torch.no_grad()
def per_token_quant(t, quant_specs=None, inplace=False):
    n_bits = quant_specs['n_bits']
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    # print(t.div(scales).round_())
    if inplace: # for weight
        t = t.div_(scales).round_().mul_(scales)
    else: # for activation
        t = t.div(scales).round().mul(scales)
    return t

@torch.no_grad()
def kl_divergence(x, x_q, bins=2048):
    a = max(x.abs().max(), x_q.abs().max())
    hist_x  = torch.histc(x, bins=bins, min=-a, max=a)
    hist_xq = torch.histc(x_q, bins=bins, min=-a, max=a)

    p = hist_x / hist_x.sum()
    q = hist_xq / hist_xq.sum()

    eps = 1e-8
    kl = torch.sum(p * torch.log((p + eps) / (q + eps)))
    return kl

@torch.no_grad()
def mse(x, x_q):
    return torch.mean((x - x_q) ** 2).item()


@torch.no_grad()
def mx_quant(t, mx_specs):
    bf_in = quantize_elemwise_op(
        t, mx_specs=mx_specs, round=mx_specs["round"]
    )

    # MX quantize everything along input size
    qis_input = quantize_mx_op(
        bf_in,
        mx_specs,
        elem_format=mx_specs['a_elem_format'],
        axes=[-1],
        round=mx_specs["round"],
    )
    return qis_input