import torch
from functools import partial
from dataclasses import dataclass, replace 
from mx.mx_ops import quantize_mx_op
from mx.elemwise_ops import quantize_elemwise_op

@dataclass
class QuantConfig:
    quant_meth: str          # int8 / int4 / fp8 / bfp
    quant_specs: dict
    smooth_en: bool
    alpha_idx: int # idx

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

        self.cfg = None
        self.name = False
        self.smooth_factors = None
        self.initial_weight = True
        self.cal_mse = False
        self.loss_func = mse
        self.n_samples = None
        self.step_flag = None

    def set_quant_config(self, cfg: QuantConfig):
        self.cfg = replace(cfg)
        self.x_mse_quant_mean = 0
        self.x_mse_smoothquant_mean = 0
        self.w_mse_quant_mean = 0
        self.w_mse_smoothquant_mean = 0
        self.y_mse_quant_mean = 0
        self.y_mse_smoothquant_mean = 0
        self.iters = 0
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
        str_ += f'smooth: {self.cfg.smooth_en}, {self.cfg.alpha_idx}'
        return str_

    def forward(self, x):
        if self.cal_mse:
            self.iters += 1
            # if self.name == 'model.backbone.encoder.layers.0.self_attn.W_Q':
            #     print(self.iters, end=' ')
            # quant
            y_org = torch.functional.F.linear(x, self.weight, self.bias)
            x_quant = self.quant_func(x, self.cfg.quant_specs)
            w_quant = self.quant_func(self.weight, self.cfg.quant_specs)
            y_quant = torch.functional.F.linear(x_quant, w_quant, self.bias)
            # x_mse_quant = self.loss_func(x, x_quant)
            # w_mse_quant = self.loss_func(self.weight, w_quant)
            y_mse_quant = self.loss_func(y_org, y_quant)
            # self.x_mse_quant_mean += x_mse_quant
            # self.w_mse_quant_mean += w_mse_quant
            self.y_mse_quant_mean += y_mse_quant
            if self.iters == self.n_samples: 
                # self.x_mse_quant_mean /= self.n_samples
                # self.w_mse_quant_mean /= self.n_samples
                self.y_mse_quant_mean /= self.n_samples
                # print(f'{self.name[23:]:<28}, {self.x_mse_quant_mean:.8f}, {self.w_mse_quant_mean:.8f}, {self.y_mse_quant_mean:.8f}')
                # print(f'runtime: {self.name:<28}, {self.y_mse_quant_mean:.8f}, ')

            if self.cfg.smooth_en:
                smooth_factor =  self.smooth_factors[self.cfg.alpha_idx].view(1, -1).to(device=x.device)
                x_smooth = x.div(smooth_factor)
                w_smooth = self.weight.mul(smooth_factor)
                x_smoothquant = self.quant_func(x_smooth, self.cfg.quant_specs)
                w_smoothquant = self.quant_func(w_smooth, self.cfg.quant_specs)
                y_smoothquant = torch.functional.F.linear(x_smoothquant, w_smoothquant, self.bias)

                x_mse_smoothquant = self.loss_func(x_smooth, x_smoothquant)
                w_mse_smoothquant = self.loss_func(w_smooth, w_smoothquant)
                y_mse_smoothquant = self.loss_func(y_org, y_smoothquant)

                self.x_mse_smoothquant_mean += x_mse_smoothquant
                self.w_mse_smoothquant_mean += w_mse_smoothquant
                self.y_mse_smoothquant_mean += y_mse_smoothquant
                if self.iters == self.n_samples: 
                    self.x_mse_smoothquant_mean /= self.n_samples
                    self.w_mse_smoothquant_mean /= self.n_samples
                    self.y_mse_smoothquant_mean /= self.n_samples
                    self.cal_mse = False
                    if self.y_mse_smoothquant_mean < self.y_mse_quant_mean:
                        self.cfg.smooth_en = True
                    print(f'{self.y_mse_smoothquant_mean:.8f}, ', end='')
                    # print(f'{self.cfg.alpha_idx}: {self.name[23:]:<28}, ({self.x_mse_quant_mean:.8f}, {self.x_mse_smoothquant_mean:.8f}), ({self.w_mse_quant_mean:.8f}, {self.w_mse_smoothquant_mean:.8f}), ({self.y_mse_quant_mean:.8f}, {self.y_mse_smoothquant_mean:.8f}), {self.y_mse_quant_mean > self.y_mse_smoothquant_mean}')
            return y_org
        
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
            x_smooth = x_quant = None
            w_smooth = w_quant = None
            if self.cfg.smooth_en:
                smooth_factor =  self.smooth_factors[self.cfg.alpha_idx].view(1, -1).to(device=x.device)
                x_smooth = x.div(smooth_factor)
                w_smooth = self.weight.mul(smooth_factor)

            x_quant = self.quant_func(x_smooth if x_smooth is not None else x, self.cfg.quant_specs)
            w_quant = self.quant_func(w_smooth if w_smooth is not None else self.weight, self.cfg.quant_specs)

            y = torch.functional.F.linear(x_quant if x_quant is not None else x, \
                                          w_quant if w_quant is not None else self.weight, self.bias)
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