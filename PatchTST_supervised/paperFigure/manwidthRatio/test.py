
import torch
from safetensors.torch import load_file

file_path = "data_type_ratio_w8a8_wikitext.safetensors"
loaded = load_file(file_path)
print(loaded)
for k, v in loaded.items():
    print(k, v, v.shape)