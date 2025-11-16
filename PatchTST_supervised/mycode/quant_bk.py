from torch.nn.parameter import Parameter, UninitializedParameter
from torch import Tensor, nn
import torch
import math
from functools import partial
from globalVar import (get_iterationCounter,
                       get_save_tensor_enable,
                       append_activation,
                       append_weight,
                       get_data_type,
                       get_clamp_block_size,
                       get_clamp_type,
                       get_profiling_enable)
@torch.no_grad()
def pseudo_quantize_tensor( w, 
                            n_bit=8, 
                            zero_point=True, 
                            q_group_size=-128, 
                            inplace=False, 
                            get_scale_zp=False, 
                            clam_quant_en=False, 
                            name=None,
                            ):

    org_tensor = w
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    # assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=-1, keepdim=True)
        min_val = w.amin(dim=-1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        (
            (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        w_int = torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) # quantized INT8
        if get_save_tensor_enable() and 'act' in name:
            append_activation(f'{name}_{get_iterationCounter()}', w_int)
        if get_save_tensor_enable() and 'wgt' in name:
            append_weight(f'{name}_{get_iterationCounter()}', w_int)
        if clam_quant_en and 'self_attn' in name:
            clamp_block_size = get_clamp_block_size()
            w_int = w_int.reshape(-1, clamp_block_size) # clamp block size
            max = w_int.amax(dim=-1, keepdim=True)
            min = w_int.amin(dim=-1, keepdim=True) 
            present_range = max - min
            # even = (max + min) // 2 # stored tensor
            w_int -= min

            clamp_idx_int1 = present_range <= 1
            clamp_idx_int2 = (present_range <= 3) * (present_range > 1)
            clamp_idx_int3 = (present_range <= 7) * (present_range > 3)
            clamp_idx_int4 = (present_range <= 15) * (present_range > 7)
            clamp_idx_int5 = (present_range <= 31) * (present_range > 15)
            clamp_idx_int6 = (present_range <= 63) * (present_range > 31)
            clamp_idx_int7 = (present_range <= 127) * (present_range > 63)
            clamp_idx_int8 = present_range > 127

            clamp_idx_int1 =clamp_idx_int1.expand(-1, w_int.size(-1)) # scales.squeeze().expand(-1, mse.size(-1))
            clamp_idx_int2 =clamp_idx_int2.expand(-1, w_int.size(-1))
            clamp_idx_int3 =clamp_idx_int3.expand(-1, w_int.size(-1))
            clamp_idx_int4 =clamp_idx_int4.expand(-1, w_int.size(-1))
            clamp_idx_int5 =clamp_idx_int5.expand(-1, w_int.size(-1)) 
            clamp_idx_int6 =clamp_idx_int6.expand(-1, w_int.size(-1))
            clamp_idx_int7 =clamp_idx_int7.expand(-1, w_int.size(-1))
            clamp_idx_int8 =clamp_idx_int8.expand(-1, w_int.size(-1))

            w_int1_clamp = w_int.masked_fill(~clamp_idx_int1, 0)
            w_int2_clamp = w_int.masked_fill(~clamp_idx_int2, 0)
            w_int3_clamp = w_int.masked_fill(~clamp_idx_int3, 0)
            w_int4_clamp = w_int.masked_fill(~clamp_idx_int4, 0)
            w_int5_clamp = w_int.masked_fill(~clamp_idx_int5, 0)
            w_int6_clamp = w_int.masked_fill(~clamp_idx_int6, 0)
            w_int7_clamp = w_int.masked_fill(~clamp_idx_int7, 0)
            w_int8_clamp = w_int.masked_fill(~clamp_idx_int8, 0)

            clamp_type = get_clamp_type()
            if clamp_type == 'base':
                pass
            else:
                # default: int1~4 clamp, int5~6 speculative clamp, int7~8 no clamp
                # decoder: int8(int7, int6, int5), int6(clamp) << 2, int5(clamp, int4) << 1
                w_int1_clamp = (w_int1_clamp // 2) * 2
                w_int2_clamp = (w_int2_clamp // 2) * 2
                w_int3_clamp = (w_int3_clamp // 2) * 2
                w_int4_clamp = (w_int4_clamp // 2) * 2

                mse = torch.mean((w_int5_clamp - (w_int5_clamp // 2) * 2) ** 2, dim=1)
                mse = mse.reshape(org_w_shape[1], -1)
                mse = mse * (scales.squeeze(0))
                mse = mse.reshape(-1, 1)
                mse_int5 = mse[clamp_idx_int5[:,0]]

                mse = torch.mean((w_int6_clamp - (w_int6_clamp // 4) * 4) ** 2, dim=1)
                mse = mse.reshape(org_w_shape[1], -1)
                mse = mse * (scales.squeeze(0))
                mse = mse.reshape(-1, 1)
                mse_int6 = mse[clamp_idx_int6[:,0]]


                w_int6_clamp_idx = ((mse < 0.2).expand(-1, w_int.size(-1))) * clamp_idx_int6
                w_int6_clamp_real = w_int6_clamp.masked_fill(~w_int6_clamp_idx, 0)
                w_int6_clamp_false = w_int6_clamp.masked_fill(w_int6_clamp_idx, 0)

                w_int5_clamp = (w_int5_clamp // 2) * 2 if 'int5' in clamp_type else w_int5_clamp
                w_int6_clamp_real = (w_int6_clamp_real // 4) * 4 if 'int6' in clamp_type else w_int6_clamp_real

                # breakpoint() # 
                with open('log/mse.txt', 'a') as f:
                    # if mse_int5.numel() > 0:
                    #     f.writelines(f'({mse_int5.min().item():.4f}, {mse_int5.mean().item():.4f}, {mse_int5.max().item():.4f})  ')
                    if mse_int6.numel() > 0:
                        f.writelines(f'clamp int6: {w_int6_clamp_idx[:,0].sum():>7} ({w_int6_clamp_idx[:,0].sum()/clamp_idx_int6[:,0].sum():.4f}), '
                                     f'int6: {clamp_idx_int6[:,0].sum():>7} ({clamp_idx_int6[:,0].sum()/mse.size(0):.4f}), total: {mse.size(0):>7} '
                                     f'({mse_int6.min().item():.4f}, {mse_int6.mean().item():.4f}, {mse_int6.max().item():.4f}) '
                                     f'>> {name}\n')

            w_int = w_int1_clamp + w_int2_clamp + w_int3_clamp + w_int4_clamp \
                + w_int5_clamp + w_int6_clamp_real + w_int6_clamp_false + w_int7_clamp + w_int8_clamp + min

            w_int = w_int.reshape(org_w_shape)
            
        w = (w_int - zeros) * scales

    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    # mseLoss = nn.MSELoss()
    # mse = mseLoss(org_tensor, w).item()
    # breakpoint()
    # if 'act' in name:
    #     with open('log/mse.txt', 'a') as f:
    #         f.writelines(f'{name}: {mse:.4f}\n')

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w

class quantLinear(torch.nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 name=None,
                 wgt_nbit=4,
                 act_nbit=8,
                 ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.name = name

        self.quant_en = False
        self.clamp_en = False
        self.spark_en = False

        self.act_quant = partial(pseudo_quantize_tensor, n_bit=act_nbit, name=f'{name}.act')
        self.wgt_quant = partial(pseudo_quantize_tensor, n_bit=wgt_nbit, q_group_size=64, name=f'{name}.wgt') # group-wise quant for weight

        super().__init__(in_features, out_features, bias)

    @staticmethod
    def set_param(module, name, wgt_nbit=4, act_nbit=8
    ):
        assert isinstance(module, torch.nn.Linear)
        new_module = quantLinear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            name=name,
            wgt_nbit=wgt_nbit,
            act_nbit=act_nbit,
        )
        new_module.weight = torch.nn.Parameter(module.weight.data.clone())
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, quant_en={self.quant_en}, clamp_quant={self.clamp_en}'

    def enable_quant(self, quant_en=True, clamp_en=True, spark_en=False):
        self.quant_en = quant_en
        self.clamp_en = clamp_en
        self.spark_en = spark_en

    @torch.no_grad()
    def forward(self, x):
        if self.quant_en:
            q_x = self.act_quant(x, clam_quant_en=self.clamp_en)
            q_w = self.wgt_quant(self.weight)
            y = torch.functional.F.linear(q_x, q_w, self.bias)
        else:
            y = torch.functional.F.linear(x, self.weight, self.bias)
        # with open('log/qwt_bench.py', 'a') as f:
        #     f.writelines(f'[{list(x.squeeze().shape)}, {list(self.weight.shape)}, {list(y.squeeze().shape)}, [], [], 8, 1 ],\n')
        return y
