import os
import random
import numpy as np
import torch
import math
from matplotlib import pyplot as plt
import torch.nn.functional as F
import logging
import time
from matplotlib.ticker import PercentFormatter

import time

import pandas as pd
import torch


def main():
    start_time = time.time()
    # ----------------------------------------------------------
    models = [
                'Llama-1.1B',
                'llama-2-7b',
                'Llama-3-8B',
                'Qwen2.5-0.5B',
                'Qwen2.5-1.5B',
                'Qwen2.5-7B',
                'Geomean'
                ]
    hw_arch = ['MANT', 'OliVe', 'SPARK', 'SqzAct']
    labels = ['DRAM', 'Buffer', 'Compute']
    colors = [(70/255, 120/255, 142/255), (120/255, 183/255, 201/255), (187/255, 151/255, 39/255), (50/255, 184/255, 151/255), (199/255, 109/255, 162/255)]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), dpi=300, gridspec_kw={'height_ratios': [1, 1.1]}) # 上下图比例1：1

    # 读 CSV
    df = pd.read_csv("my_res.csv", header=None)
    # 去掉第一行（表头），去掉第一列（Static/Dram/...），去掉最后一列空的逗号
    values = df.iloc[11:12, 1:-1].astype(float).values
    tensor_org = torch.tensor(values, dtype=torch.float32)
    new_row = tensor_org[0:2].sum(dim=0)   # 把前两行相加
    tensor = torch.vstack([new_row, tensor_org[2:]])
    tensor.squeeze_(0)

    print(tensor)

    # 读 CSV（假设文件叫 energy.csv）
    df = pd.read_csv("my_res.csv", header=None)

    # 去掉第一行（表头），去掉第一列（Static/Dram/...），去掉最后一列空的逗号
    values = df.iloc[14:18, 1:-1].astype(float).values

    tensor_org = torch.tensor(values, dtype=torch.float32)
    new_row = tensor_org[0:2].sum(dim=0)   # 把前两行相加
    tensor = torch.vstack([new_row, tensor_org[2:]])
    print(tensor)

    
if __name__ == '__main__':
    main()
