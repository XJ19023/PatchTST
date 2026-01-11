from safetensors.torch import save_file

iter = 0
def increas_counter():
    global iter
    iter += 1
def get_counter():
    global iter
    return iter


smooth_factor = {}
def append_smooth_factor(k, v):
    global smooth_factor
    smooth_factor[f'{k}'] = v.detach()
def get_smooth_factor():
    global smooth_factor
    return smooth_factor
activations_save = {}
weights_save = {}
def reset_activation():
    global activations_save
    activations_save = {}
def append_activation(k, v):
    global activations_save
    activations_save[f'{k}'] = v.detach()
def append_weight(k, v):
    global weights_save
    weights_save[f'{k}'] = v.detach()
def save_tensors(dir=None):
    global activations_save
    global weights_save
    activations_save = {k: v.contiguous().clone() for k, v in activations_save.items()}
    with open(f'{dir}/activation_key.py', 'w') as f:
        f.writelines(f"act_keys = [")
        for k in activations_save.keys():
            f.writelines(f"'{k}',\n")
        f.writelines(f"]")
    weights_save = {k: v.clone() for k, v in weights_save.items()}
    with open(f'{dir}/weight_key.py', 'w') as f:
        f.writelines(f"wgt_keys = [")
        for k in weights_save.keys():
            f.writelines(f"'{k}',\n")
        f.writelines(f"]")
    save_file(activations_save, f"{dir}/activation.safetensors")
    save_file(weights_save, f"{dir}/weight.safetensors")

save_tensor_enable = False
def set_save_tensor_enable():
    global save_tensor_enable
    save_tensor_enable = True
def get_save_tensor_enable():
    global save_tensor_enable
    return save_tensor_enable

clamped_quant_enable = False
def set_clamped_quant_enable():
    global clamped_quant_enable
    clamped_quant_enable = True
def get_clamped_quant_enable():
    global clamped_quant_enable
    return clamped_quant_enable

profiling_enable = False
def set_profiling_enable():
    global profiling_enable
    profiling_enable = True
def get_profiling_enable():
    global profiling_enable
    return profiling_enable

data_type = None
def set_data_type(dataType):
    global data_type
    data_type = dataType
def get_data_type():
    global data_type
    return data_type

clamp_block_size = None
def set_clamp_block_size(block_size):
    global clamp_block_size
    clamp_block_size = block_size
def get_clamp_block_size():
    global clamp_block_size
    return clamp_block_size

clamp_type = None
def set_clamp_type(CType):
    global clamp_type
    clamp_type = CType
def get_clamp_type():
    global clamp_type
    return clamp_type

mse = None
def set_mse(m):
    global mse
    mse = m
def get_mse():
    global mse
    return mse


int4 = int5 = int6 = int8 = total = 1
def reset_counter():
    global int4, int5, int6, int8, total
    int4 = int5 = int6 = int8 = total = 1
def increas_int4_int5_int6_int8(int4_incr, int5_incr, int6_incr, int8_incr, total_incr):
    global int4, int5, int6, int8, total
    int4 += int4_incr
    int5 += int5_incr
    int6 += int6_incr
    int8 += int8_incr
    total += total_incr
def get_ratio():
    global int4, int5, int6, int8, total
    return int4, int5, int6, int8, total