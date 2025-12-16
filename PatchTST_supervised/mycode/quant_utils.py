import torch
from functools import partial

class ActQuantizer(torch.nn.Module):

    '''
        A class for quantizing the activations. We only support (both sym. and asym.) per-token quantization
        for the activations.
    '''

    def __init__(self):
        super(ActQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(1))
        self.register_buffer('zero', torch.zeros(1))
        self.bits = 16

    def free(self):
        self.zero = None
        self.scale = None

    def forward(self, x):
        x_dtype = x.dtype
        if self.bits == 16:
            return x


    # Different from `forward`, this method returns quantized integers, scales (and zeros if asymmetric).
    def quantize(self, x):
        if self.sym:
            return (x, self.scale, self.maxq)

    def configure(self, bits, groupsize=-1, sym=False, clip_ratio=1.0):
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio
        assert self.clip_ratio <= 1 and self.clip_ratio > 0, 'Clip ratio should be in (0, 1]'

    def find_params_per_token_groupwise(self, x):
        init_shape = x.shape
        reshaped_x = x.reshape(-1, x.shape[-2], x.shape[-1] // self.groupsize, self.groupsize)

        xmax = torch.amax(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        xmin = torch.amin(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = xmax / self.maxq
            self.scale[tmp] = 1
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        self.scale = self.scale.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
        self.zero = self.zero.repeat(1, 1, 1, self.groupsize).reshape(init_shape)

    def find_params(self, x):
        if self.bits == 16:
            return

        dev = x.device
        self.maxq = self.maxq.to(dev)

        init_shape = x.shape

        if self.groupsize > 0:
            # group-wise per-token quantization
            self.find_params_per_token_groupwise(x)
            return

        reshaped_x = x.reshape((-1, x.shape[-1]))

        tmp = torch.zeros(reshaped_x.shape[0], device=dev)
        xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clip_ratio
        xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = (xmax / self.maxq).unsqueeze(1).repeat(1, reshaped_x.shape[-1])
            self.scale[tmp] = 1
            self.scale = self.scale.reshape(init_shape)
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

            self.scale = self.scale.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
            self.zero = self.zero.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)

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
        # self.module = module
        self.weight = module.weight
        self.bias = module.bias
        # self.quantizer = ActQuantizer()
        # self.out_quantizer = ActQuantizer()

        self.name = False
        self.smooth_en = False
        self.quant_meth = None
        self.n_bits = None
        self.initial_weight = False
        self.search = False

        self.loss_func = mse
        self.x_mse_quant_mean = 0
        self.x_mse_smoothquant_mean = 0
        self.w_mse_quant_mean = 0
        self.w_mse_smoothquant_mean = 0
        self.y_mse_quant_mean = None
        self.y_mse_smoothquant_mean = None

        self.n_samples = None
        self.iters = None

    def extra_repr(self) -> str:
        str_ = f'quant_meth: {self.quant_meth}, bits: {self.n_bits}, smooth: {self.smooth_factor is not None}'
        return str_

    def forward(self, x):
        if self.search:
            self.iters += 1
            # if self.name == 'model.backbone.encoder.layers.0.self_attn.W_Q':
            #     print(self.iters, end=' ')
            x_smooth = x_quant = x_smoothquant = None
            w_smooth = w_quant = w_smoothquant = None
            # smooth
            self.smooth_factor =  self.smooth_factor.view(1, -1).to(device=x.device)
            x_smooth = x.div(self.smooth_factor)
            w_smooth = self.weight.mul(self.smooth_factor)
            # quant
            x_quant = per_token_quant(x, self.n_bits, inplace=False)
            w_quant = per_token_quant(self.weight, self.n_bits, inplace=False)
            x_smoothquant = per_token_quant(x_smooth, self.n_bits, inplace=False)
            w_smoothquant = per_token_quant(w_smooth, self.n_bits, inplace=False)

            y_org = torch.functional.F.linear(x, self.weight, self.bias)
            y_quant = torch.functional.F.linear(x_quant, w_quant, self.bias)
            y_smoothquant = torch.functional.F.linear(x_smoothquant, w_smoothquant, self.bias)

            x_mse_quant = self.loss_func(x, x_quant)
            x_mse_smoothquant = self.loss_func(x_smooth, x_smoothquant)
            w_mse_quant = self.loss_func(self.weight, w_quant)
            w_mse_smoothquant = self.loss_func(w_smooth, w_smoothquant)
            y_mse_quant = self.loss_func(y_org, y_quant)
            y_mse_smoothquant = self.loss_func(y_org, y_smoothquant)

            self.x_mse_quant_mean += x_mse_quant
            self.x_mse_smoothquant_mean += x_mse_smoothquant
            self.w_mse_quant_mean += w_mse_quant
            self.w_mse_smoothquant_mean += w_mse_smoothquant
            self.y_mse_quant_mean += y_mse_quant
            self.y_mse_smoothquant_mean += y_mse_smoothquant
            if self.iters == self.n_samples: 
                self.x_mse_quant_mean /= self.n_samples
                self.x_mse_smoothquant_mean /= self.n_samples
                self.w_mse_quant_mean /= self.n_samples
                self.w_mse_smoothquant_mean /= self.n_samples
                self.y_mse_quant_mean /= self.n_samples
                self.y_mse_smoothquant_mean /= self.n_samples
                print(f'{self.name[23:]:<28}, ({self.x_mse_quant_mean:.8f}, {self.x_mse_smoothquant_mean:.8f}), ({self.w_mse_quant_mean:.8f}, {self.w_mse_smoothquant_mean:.8f}), ({self.y_mse_quant_mean:.8f}, {self.y_mse_smoothquant_mean:.8f}), {self.y_mse_quant_mean > self.y_mse_smoothquant_mean}')
            return y_org
        
        if self.initial_weight:
            # org_tensor = self.weight.clone()
            if self.smooth_factor is not None:
                self.smooth_factor =  self.smooth_factor.view(1, -1).to(device=x.device)
                with torch.no_grad():
                    self.weight.mul_(self.smooth_factor)

            if self.quant_meth == 'int':
                with torch.no_grad():
                    self.weight = per_token_quant(self.weight, self.n_bits)

            # print(f'{self.extra_repr()}, mse: {torch.mean((self.weight - org_tensor) ** 2).item():.18f}')
            self.initial_weight = False
            # return torch.zeros(*x.shape[:-1], self.weight.size(0), device=x.device, dtype=x.dtype)
            return torch.zeros_like(x[..., :self.weight.size(0)])

        else:
            x_smooth = x_quant = None
            if self.smooth_factor is not None:
                self.smooth_factor =  self.smooth_factor.view(1, -1).to(device=x.device)
                x_smooth = x.div(self.smooth_factor)

            if self.quant_meth == 'int':
                x_quant = per_token_quant(x_smooth if x_smooth is not None else x, self.n_bits, inplace=False)

            y = torch.functional.F.linear(x_quant if x_quant is not None else x, self.weight, self.bias)
            # print(f'{self.extra_repr()}, mse: {torch.mean((x - x_quant) ** 2).item():.18f}')
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
def per_token_quant(t, n_bits=8, inplace=True):
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
