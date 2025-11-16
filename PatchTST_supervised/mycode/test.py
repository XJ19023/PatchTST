
from safetensors.torch import load_file
state_dict = load_file('log/Qwen3-8B/delta_int5.safetensors')
from delta_key import delta_key
for k in delta_key:
    print(f'{k}, {state_dict[k]}')