"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""

import torch
import torch.nn.functional as F

from .mx_ops import quantize_mx_op, _shared_exponents, _quantize_elemwise_core
from .elemwise_ops import quantize_elemwise_op
from .specs import apply_mx_specs, get_backwards_mx_specs
from .specs import mx_assert_test
from .matmul_precision import set_matmul_precision

from mycode.globalVar import increas_counter, get_counter, append_activation

f_linear = F.linear
torch_matmul = torch.matmul

def matmul_decompose_general(A, B, acc_bits):
    """
    A: (..., M, K)
    B: (K, N)
    return:
        P: (..., M, K, N)  # all pairwise products
        C: (..., M, N)     # sum along K (matmul result)
    """

    # Expand A → (..., M, K, 1)
    A_exp = A.unsqueeze(-1)

    # Expand B → (1, ..., 1, K, N) 让它能广播
    # 在前面加与 A 相同数量的 batch 维
    expand_dims = A.dim() - 2  # A 的 batch 维数
    B_exp = B.view(*([1] * expand_dims), B.shape[0], B.shape[1])
    # B_exp shape = (..., K, N)

    # Step 1：pairwise 乘法 → (..., M, K, N)
    P = A_exp * B_exp

    # Step 2：对 K 维求和 → (..., M, N)
    # Y = P.sum(dim=-2)
    # 把 N 和 K 交换
    Pt = P.transpose(-1, -2)  # (..., K, N)
    max_exp = _shared_exponents(Pt, axes=[-1])

    Pt = Pt / (2**max_exp)
    mbits = acc_bits + 1  # include sign bit and hidden bit
    max_norm = float(2**(mbits-1) - 1) / 2**(mbits-2)
    Pt = _quantize_elemwise_core(
                Pt, mbits, 0, max_norm, round='floor',
                allow_denorm=True, saturate_normals=True,
                custom_cuda=False)
    Pt = Pt * (2**max_exp)

    ones = torch.ones(Pt.shape[-1], device=Pt.device, dtype=Pt.dtype)  # shape: (N,)
    Y = torch.matmul(Pt, ones)  # (..., K)

    return Y

def matmul_split_B(A, B, acc_bits, chunk_size=2):
    """
    A: (..., M, K)
    B: (K, N)
    将 B 沿最后一维（列 N）分块，逐块计算 A @ B_chunk，避免 OOM。
    返回: (..., M, N)
    """
    K = A.shape[-1]
    assert B.shape[0] == K, "A and B shape mismatch"

    N = B.shape[1]
    outputs = []   # 保存每个块的结果

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        B_chunk = B[:, start:end]        # shape: (K, chunk)

        # 逐块矩阵乘法
        # C_chunk = A @ B_chunk            # shape: (..., M, chunk)
        C_chunk = matmul_decompose_general(A, B_chunk, acc_bits)  # 使用分解方法

        outputs.append(C_chunk)

    # 沿最后一维拼接
    return torch.cat(outputs, dim=-1)

class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias=None,
        mx_specs=None,
        prequantized_weights=False,
        name=None,
    ):
        # element-wise quantize for input
        bf_in = quantize_elemwise_op(
            input, mx_specs=mx_specs, round=mx_specs["round_output"]
        )

        # element-wise quantize for weight and bias
        if not prequantized_weights:
            bf_weight = quantize_elemwise_op(
                weight, mx_specs=mx_specs, round=mx_specs["round_weight"]
            )
        else:
            assert(weight.dtype == torch.bfloat16)
            bf_weight = weight

        if bias is not None:
            ctx.has_bias = True
            if not prequantized_weights:
                bf_bias = quantize_elemwise_op(
                    bias, mx_specs=mx_specs, round=mx_specs["round_weight"]
                )
            else:
                assert(bias.dtype == torch.bfloat16)
                bf_bias = bias
        else:
            ctx.has_bias = False
            bf_bias = None

        if mx_specs["quantize_backprop"]:
            ctx.save_for_backward(bf_in, bf_weight)
        else:
            ctx.save_for_backward(input, weight)

        # MX quantize everything along input size
        qis_input = quantize_mx_op(
            bf_in,
            mx_specs,
            elem_format=mx_specs['a_elem_format'],
            axes=[-1],
            round=mx_specs["round_mx_output"],
        )
        qis_weight = quantize_mx_op(
            bf_weight,
            mx_specs,
            elem_format=mx_specs['w_elem_format'],
            axes=[-1],
            round=mx_specs["round_mx_output"],
        )

        # In case of prequantized weights, the output of quantize_mx_op will return bfloat16 output.
        # while qtzd_i/p can be anything. Thus we match the dtypes here.
        if qis_weight.dtype == torch.bfloat16 and qis_input.dtype != torch.bfloat16:
            qis_weight = qis_weight.to(qis_input.dtype)


        if mx_specs['acc_bits']:
            try:
                output = matmul_decompose_general(qis_input, qis_weight.transpose(0, 1), acc_bits=mx_specs['acc_bits'])
            except:
                try:
                    output = matmul_split_B(qis_input, qis_weight.transpose(0, 1), acc_bits=mx_specs['acc_bits'], chunk_size=8)
                except:
                    raise SystemExit
        else:
            output = qis_input @ qis_weight.transpose(0, 1)

        # print(increas_counter(output.equal(output_org)))
        output = quantize_elemwise_op(
            output, mx_specs=mx_specs, round=mx_specs["round_output"]
        )

        if bias is not None:
            output = output + bf_bias
            output = quantize_elemwise_op(
                output, mx_specs=mx_specs, round=mx_specs["round_output"]
            )

        ctx.mx_specs = get_backwards_mx_specs(mx_specs)
        ctx.prequantized_weights = prequantized_weights
        ctx.name = name
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert(ctx.prequantized_weights == False), \
            "Cannot use prequantized weights when training!"
        # load context
        input, weight = ctx.saved_tensors

        out_dim = weight.shape[0]
        in_dim = weight.shape[1]

        grad_output = quantize_elemwise_op(
            grad_output,
            mx_specs=ctx.mx_specs,
            round=ctx.mx_specs["round_grad_input"],
        )

        #####################################################
        # perform madtile operation for grad_weight, grad_bias
        #####################################################
        # if the input is 2D, quantize everything along examples (batches)
        # if the input is 3D, quantize everything along the first axis
        qex_input = quantize_mx_op(
            input,
            ctx.mx_specs,
            elem_format=ctx.mx_specs['a_elem_format_bp'],
            axes=[-2],
            round=ctx.mx_specs["round_mx_input_grad_weight"],
        )
        qex_grad_output = quantize_mx_op(
            grad_output,
            ctx.mx_specs,
            elem_format=ctx.mx_specs['a_elem_format_bp_ex'],
            axes=[-2],
            round=ctx.mx_specs["round_mx_grad_output_grad_weight"],
        )

        # compute grad_weight [out_features, in_features]
        qex_grad_output = qex_grad_output.reshape(-1, out_dim)
        qex_input = qex_input.reshape(-1, in_dim)

        # Compute grad_weight
        with set_matmul_precision(qex_grad_output, qex_input,
                        ctx.mx_specs['a_elem_format_bp_ex'],
                        ctx.mx_specs['a_elem_format_bp']):
            grad_weight = torch_matmul(qex_grad_output.transpose(0, 1), qex_input)
        
        grad_weight = quantize_elemwise_op(
            grad_weight,
            mx_specs=ctx.mx_specs,
            round=ctx.mx_specs["round_grad_weight"],
        )

        #####################################################
        # perform madtile operation for grad_input
        #####################################################
        # compute grad_input, quantize everything along output size
        qos_weight = quantize_mx_op(
            weight,
            ctx.mx_specs,
            elem_format=ctx.mx_specs['w_elem_format_bp'],
            axes=[0],
            round=ctx.mx_specs["round_mx_weight_grad_input"],
        )
        # grad_output shape is (B, seq, out_dim)
        qos_grad_output = quantize_mx_op(
            grad_output,
            ctx.mx_specs,
            elem_format=ctx.mx_specs['a_elem_format_bp_os'],
            axes=[-1],
            round=ctx.mx_specs["round_mx_grad_output_grad_input"],
        )

        # Compute grad_input
        with set_matmul_precision(qos_grad_output, qos_weight,
                        ctx.mx_specs['a_elem_format_bp_os'],
                        ctx.mx_specs['w_elem_format_bp']):
            grad_input = torch_matmul(qos_grad_output, qos_weight)
        
        grad_input = quantize_elemwise_op(
            grad_input,
            mx_specs=ctx.mx_specs,
            round=ctx.mx_specs["round_grad_input"],
        )

        #####################################################
        # Compute grad_bias
        #####################################################
        if not ctx.has_bias:
            grad_bias = None
        else:
            grad_bias = grad_output.reshape(-1, out_dim).sum(0)
            grad_bias = quantize_elemwise_op(
                grad_bias,
                mx_specs=ctx.mx_specs,
                round=ctx.mx_specs["round_grad_weight"],
            )

        return (grad_input, grad_weight, grad_bias, None, None, None, None)


def linear(
    input,
    weight,
    bias=None,
    mx_specs=None,
    prequantized_weights=False,
    name=None,
):
    mx_assert_test(mx_specs)
    if mx_specs is None:
        return f_linear(input, weight, bias=bias)

    mx_specs = apply_mx_specs(mx_specs)

    return LinearFunction.apply(input, weight, bias, mx_specs, 
                                prequantized_weights, name)


class mxLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        mx_specs=None,
        name=None,
    ):
        mx_assert_test(mx_specs)
        self.mx_none = mx_specs is None

        self.name = name
        self.prequantized_weights = False
        self.mx_specs = apply_mx_specs(mx_specs)
        super().__init__(in_features, out_features, bias)

    @staticmethod
    def set_param(module, mx_specs, name,
    ):
        assert isinstance(module, torch.nn.Linear)
        new_module = mxLinear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            mx_specs=mx_specs,
            name=name,
        )
        new_module.weight = torch.nn.Parameter(module.weight.data.clone())
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def apply_mx_specs(self, mx_specs):
        mx_assert_test(mx_specs)
        self.mx_none = mx_specs is None
        self.mx_specs = apply_mx_specs(mx_specs)

    def append_name(self, postfix):
        self.name += postfix
    
    def prequantize_weights(self):
        # Can't prequantize if not using bfloat weights
        if self.mx_none:
            return
        
        assert self.mx_specs["round"] == 'even', \
            "Bfloat round should be 'even' for prequantizing weights."
        assert torch.cuda.is_bf16_supported(), \
            "Current device does not support bfloat16"
        assert self.mx_specs['bfloat_subnorms'] == True, \
            "Bfloat_subnorms should be set to True for prequantizing weights."
        assert self.mx_specs["bfloat"] == 16, \
            "Only Bfloat16 is supported for prequantizing weights."

        with torch.no_grad():
            self.weight.data = quantize_elemwise_op(
                    self.weight.data,
                    mx_specs=self.mx_specs,
                    round=self.mx_specs["round_weight"]
            ).to(torch.bfloat16)

            if self.bias is not None:
                self.bias.data = quantize_elemwise_op(
                        self.bias.data,
                        mx_specs=self.mx_specs,
                        round=self.mx_specs["round_weight"]
                ).to(torch.bfloat16)

        self.prequantized_weights = True

    def forward(self, inputs):
        # if self.mx_none:
        if self.mx_specs['block_size'] == 0:
            print(self.name, inputs.shape)
            # append_activation(self.name, inputs)
            return super().forward(inputs)
        
        if self.prequantized_weights:
            assert(self.training == False), \
                "Cannot use prequantized weights when training!"

        return linear(
            input=inputs,
            weight=self.weight,
            bias=self.bias,
            mx_specs=self.mx_specs,
            prequantized_weights=self.prequantized_weights,
            name=self.name,
        )
