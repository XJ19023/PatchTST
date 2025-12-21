
from mx.specs import finalize_mx_specs



def set_mx_specs(block_size=16, w_elem_format='int4', a_elem_format='int4', acc_bits=24):
    mx_specs = {
        'w_elem_format': 'int4',
        'a_elem_format': 'int4',
        'block_size': 16,
        'acc_bits': 0,
        'bfloat': 16,
        'scale_bits': 8,
        'custom_cuda': False,
        # For quantization-aware finetuning, do backward pass in FP32
        'quantize_backprop': False,
    }
    mx_specs = finalize_mx_specs(mx_specs)
    mx_specs.update({'block_size' : block_size, 'acc_bits' : acc_bits, 'w_elem_format' : w_elem_format, 'a_elem_format' : a_elem_format})
    return mx_specs
