'''
    -计算int4 int5_6 int7_8的占比
'''
import argparse
import os
import random
import sys
import numpy as np
from safetensors.torch import load_file
import torch
import math
from safetensors.torch import save_file
from matplotlib import pyplot as plt
import torch.nn as nn

import torch
import matplotlib.pyplot as plt
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np

start_time = time.time()
# ----------------------------------------------------------

# 创建MSE损失函数对象
loss = nn.MSELoss()


from save_tensors.activation_key import act_keys

safetensors_path = f"save_tensors/org/activation.safetensors"
keys = act_keys
state_dict = load_file(safetensors_path)

for k in keys:
    print(k, state_dict[k].shape)