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

        self.smooth_factor = None
        self.quant_meth = None
        self.n_bits = None
        self.initial_weight = False

    def extra_repr(self) -> str:
        str_ = f'quant_meth: {self.quant_meth}, bits: {self.n_bits}'
        return str_

    def forward(self, x):
        if self.initial_weight:
            if self.smooth_factor is not None:
                self.smooth_factor =  self.smooth_factor.view(1, -1).to(device=x.device)
                with torch.no_grad():
                    self.weight.mul_(self.smooth_factor)

            if self.quant_meth == 'int':
                with torch.no_grad():
                    self.weight = per_token_quant(self.weight, self.n_bits)

            self.initial_weight = False
            # return torch.zeros(*x.shape[:-1], self.weight.size(0), device=x.device, dtype=x.dtype)
            return torch.zeros_like(x[..., :self.weight.size(0)])

        else:
            x_smooth = x_quant = None
            if self.smooth_factor is not None:
                self.smooth_factor =  self.smooth_factor.view(1, -1).to(device=x.device)
                x_smooth = x.div(self.smooth_factor)

            if self.quant_meth == 'int':
                x_quant = per_token_quant(x_smooth if x_smooth is not None else x, self.n_bits)

            y = torch.functional.F.linear(x_quant if x_quant is not None else x, self.weight, self.bias)
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
def per_token_quant(t, n_bits=8):
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    # print(t.div(scales).round_())
    t.div_(scales).round_().mul_(scales)
    return t