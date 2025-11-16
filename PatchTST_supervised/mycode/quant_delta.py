from torch.nn.parameter import Parameter, UninitializedParameter
from torch import Tensor, nn
import torch
import math
from functools import partial
from globalVar import (get_iterationCounter,
                       get_save_tensor_enable,
                       append_activation,
                       append_weight,
                       get_mse,
                       get_clamp_block_size,
                       get_clamp_type,
                       increas_int4_int5_int6_int8)
@torch.no_grad()
def pseudo_quantize_tensor( w, 
                            n_bit=8, 
                            zero_point=True, 
                            q_group_size=-128, 
                            inplace=False, 
                            get_scale=False, 
                            clam_quant_en=False, 
                            name=None,
                            percentile=None,
                            profile=False,
                            mse_int5_th=None,
                            mse_int6_th=None,
                            search=False,
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
        # if get_save_tensor_enable() and 'wgt' in name:
        #     append_weight(f'{name}_{get_iterationCounter()}', w_int)
        if clam_quant_en:
        # if clam_quant_en and 'down_proj' not in name:
            clamp_block_size = get_clamp_block_size()
            w_int = w_int.reshape(-1, clamp_block_size) # clamp block size
            max = w_int.amax(dim=-1, keepdim=True)
            min = w_int.amin(dim=-1, keepdim=True) 
            present_range = max - min
            # even = (max + min) // 2 # stored tensor
            w_int -= min

            clamp_idx_int4 = present_range <= 15
            clamp_idx_int5 = (present_range <= 31) * (present_range > 15)
            clamp_idx_int6 = (present_range <= 63) * (present_range > 31)
            clamp_idx_int7 = (present_range <= 127) * (present_range > 63)
            clamp_idx_int8 = present_range > 127

            clamp_idx_int4 =clamp_idx_int4.expand(-1, w_int.size(-1))
            clamp_idx_int5 =clamp_idx_int5.expand(-1, w_int.size(-1)) 
            clamp_idx_int6 =clamp_idx_int6.expand(-1, w_int.size(-1))
            clamp_idx_int7 =clamp_idx_int7.expand(-1, w_int.size(-1))
            clamp_idx_int8 =clamp_idx_int8.expand(-1, w_int.size(-1))

            w_int4_clamp = w_int.masked_fill(~clamp_idx_int4, 0)
            w_int5_clamp = w_int.masked_fill(~clamp_idx_int5, 0)
            w_int6_clamp = w_int.masked_fill(~clamp_idx_int6, 0)
            w_int7_clamp = w_int.masked_fill(~clamp_idx_int7, 0)
            w_int8_clamp = w_int.masked_fill(~clamp_idx_int8, 0)

            clamp_type = get_clamp_type()
            if clamp_type == 'base':
                pass
            else:
                mse = torch.sum(w_int5_clamp - (w_int5_clamp // 2) * 2, dim=1)
                mse = mse.reshape(org_w_shape[1], -1)
                mse = mse.reshape(-1, 1)
                mse_int5 = mse[clamp_idx_int5[:,0]]

                if search:
                    if mse_int5.numel() > 0:
                        mse_int5_th = torch.quantile(mse_int5.float(), percentile[0])
                    else:
                        mse_int5_th = 0

                w_int5_clamp_idx = ((mse < mse_int5_th).expand(-1, w_int.size(-1))) * clamp_idx_int5
                w_int5_clamp_real = w_int5_clamp.masked_fill(~w_int5_clamp_idx, 0)
                w_int5_clamp_false = w_int5_clamp.masked_fill(w_int5_clamp_idx, 0)

                mse = torch.sum(w_int6_clamp - (w_int6_clamp // 4) * 4, dim=1)
                mse = mse.reshape(org_w_shape[1], -1)
                mse = mse.reshape(-1, 1)
                mse_int6 = mse[clamp_idx_int6[:,0]]
                # breakpoint() #
                if search:
                    if mse_int6.numel() > 0:
                        mse_int6_th = torch.quantile(mse_int6.float(), percentile[1])
                    else:
                        mse_int6_th = 0

                w_int6_clamp_idx = ((mse < mse_int6_th).expand(-1, w_int.size(-1))) * clamp_idx_int6
                w_int6_clamp_real = w_int6_clamp.masked_fill(~w_int6_clamp_idx, 0)
                w_int6_clamp_false = w_int6_clamp.masked_fill(w_int6_clamp_idx, 0)

                # w_int5_clamp_real = (w_int5_clamp_real // 2) * 2 if 'int5' in clamp_type else w_int5_clamp_real
                # w_int6_clamp_real = (w_int6_clamp_real // 4) * 4 if 'int6' in clamp_type else w_int6_clamp_real
                tmp = w_int5_clamp_real / 2
                int_part = torch.floor(tmp)
                frac_part = tmp - int_part
                w_int5_clamp_real = int_part + (frac_part > 0.4).to(tmp.dtype)
                w_int5_clamp_real *= 2

                tmp = w_int6_clamp_real / 4
                int_part = torch.floor(tmp)
                frac_part = tmp - int_part
                w_int6_clamp_real = int_part + (frac_part > 0.7).to(tmp.dtype)
                w_int6_clamp_real *= 4

                w_int5_clamp = w_int5_clamp_real + w_int5_clamp_false
                w_int6_clamp = w_int6_clamp_real + w_int6_clamp_false

                # breakpoint() # 
                if profile:
                    increas_int4_int5_int6_int8(clamp_idx_int4[:,0].sum().item(), w_int5_clamp_idx[:,0].sum().item(), w_int6_clamp_idx[:,0].sum().item(), clamp_idx_int8[:,0].sum().item(), mse.size(0))
                    # with open('log/mse.txt', 'a') as f:
                    #     if mse_int6.numel() > 0:
                    #         f.writelines(f'clamp int6: {w_int6_clamp_idx[:,0].sum():>7} ({w_int6_clamp_idx[:,0].sum()/clamp_idx_int6[:,0].sum():.4f}), '
                    #                     f'int6: {clamp_idx_int6[:,0].sum():>7} ({clamp_idx_int6[:,0].sum()/mse.size(0):.4f}), total: {mse.size(0):>7} ')

            w_int = w_int4_clamp + w_int5_clamp + w_int6_clamp + w_int7_clamp + w_int8_clamp + min

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

    if search:
        return w, mse_int5_th, mse_int6_th
    else:
        return w

class quantLinear(torch.nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 name=None,
                #  wgt_nbit=4,
                 act_nbit=8,
                 ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.name = name

        self.search = False
        self.mse_int5_th = 10000 
        self.mse_int6_th = 10000 
        self.quant_en = False 
        self.clamp_en = False
        self.search = False
        self.spark_en = False

        self.act_quant = partial(pseudo_quantize_tensor, n_bit=act_nbit, name=f'{name}.act')

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
            act_nbit=act_nbit,
        )
        weight = pseudo_quantize_tensor(module.weight.data.clone(), n_bit=wgt_nbit, q_group_size=64, name=f'{name}.wgt')
        new_module.weight = torch.nn.Parameter(weight)
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, quant_en={self.quant_en}, clamp_quant={self.clamp_en}'

    def enable_quant(self, quant_en=True, clamp_en=True, search_en=False):
        self.quant_en = quant_en
        self.clamp_en = clamp_en
        self.search = search_en
        if search_en:
            self.mse_int5_th = 10000 
            self.mse_int6_th = 10000

    @torch.no_grad()
    def forward(self, x):
        if self.search:
            mseLoss = nn.MSELoss()
            mse_th = get_mse()
            # mse_th = 0.00001 if 'layers.21.' in self.name else get_mse()
            y_org = torch.functional.F.linear(x, self.weight, self.bias)
            for percentile in [(1,   1), (1,   0.9), (1,   0.8), (1,   0.7), (1,   0.6), (1,   0.5), (1,   0.4), (1,   0.3), (1,   0.2), (1,   0.1),
                               (0.9, 1), (0.9, 0.9), (0.9, 0.8), (0.9, 0.7), (0.9, 0.6), (0.9, 0.5), (0.9, 0.4), (0.9, 0.3), (0.9, 0.2), (0.9, 0.1),
                               (0.8, 1), (0.8, 0.9), (0.8, 0.8), (0.8, 0.7), (0.8, 0.6), (0.8, 0.5), (0.8, 0.4), (0.8, 0.3), (0.8, 0.2), (0.8, 0.1),
                               (0.7, 1), (0.7, 0.9), (0.7, 0.8), (0.7, 0.7), (0.7, 0.6), (0.7, 0.5), (0.7, 0.4), (0.7, 0.3), (0.7, 0.2), (0.7, 0.1),
                               (0.6, 1), (0.6, 0.9), (0.6, 0.8), (0.6, 0.7), (0.6, 0.6), (0.6, 0.5), (0.6, 0.4), (0.6, 0.3), (0.6, 0.2), (0.6, 0.1),
                               (0.5, 1), (0.5, 0.9), (0.5, 0.8), (0.5, 0.7), (0.5, 0.6), (0.5, 0.5), (0.5, 0.4), (0.5, 0.3), (0.5, 0.2), (0.5, 0.1),
                               (0.4, 1), (0.4, 0.9), (0.4, 0.8), (0.4, 0.7), (0.4, 0.6), (0.4, 0.5), (0.4, 0.4), (0.4, 0.3), (0.4, 0.2), (0.4, 0.1),
                               (0.3, 1), (0.3, 0.9), (0.3, 0.8), (0.3, 0.7), (0.3, 0.6), (0.3, 0.5), (0.3, 0.4), (0.3, 0.3), (0.3, 0.2), (0.3, 0.1),
                               (0.2, 1), (0.2, 0.9), (0.2, 0.8), (0.2, 0.7), (0.2, 0.6), (0.2, 0.5), (0.2, 0.4), (0.2, 0.3), (0.2, 0.2), (0.2, 0.1),
                               (0.1, 1), (0.1, 0.9), (0.1, 0.8), (0.1, 0.7), (0.1, 0.6), (0.1, 0.5), (0.1, 0.4), (0.1, 0.3), (0.1, 0.2), (0.1, 0.1)]:
                q_x, mse_int5_th, mse_int6_th = self.act_quant(x, clam_quant_en=self.clamp_en, percentile=percentile, search=True)
                y = torch.functional.F.linear(q_x, self.weight, self.bias)
                mse = mseLoss(y_org, y).item()
                # if mse < 0.0001:
                if mse < mse_th:
                    break
            if self.mse_int5_th > mse_int5_th:
                self.mse_int5_th = mse_int5_th
            if self.mse_int6_th > mse_int6_th:
                self.mse_int6_th = mse_int6_th
            # self.search = False

        q_x = self.act_quant(x, clam_quant_en=self.clamp_en, mse_int5_th=self.mse_int5_th, mse_int6_th=self.mse_int6_th, profile=True)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        
        # with open('log/mse.txt', 'a') as f:
        #     f.writelines(f'{mse:.4f} '
        #                 f'>> {self.name}\n')
        # with open('log/qwt_bench.py', 'a') as f:
        #     f.writelines(f'[{list(x.squeeze().shape)}, {list(self.weight.shape)}, {list(y.squeeze().shape)}, [], [], 8, 1 ],\n')
        return y
