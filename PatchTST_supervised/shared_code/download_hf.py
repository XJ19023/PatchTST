#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/25 14:29
# @Author  : JasonLiu
# @File    : download_hf.py
from huggingface_hub import snapshot_download


model = 'facebook/opt-125m'
repo_id = f"{model}"  # 模型在huggingface上的名称
local_dir = f"/cephfs/shared/juxin/models/opt-125m"  # 本地模型存储的地址
local_dir_use_symlinks = False  # 本地模型使用文件保存，而非blob形式保存
token = " "  # 在hugging face上生成的 access token

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=local_dir_use_symlinks,
    token=token,
    allow_patterns=[
    "*.safetensors",
    "*.bin",
    "*.json",
    "*.txt",
    "*.model",
    "*.md"
    ],
    # ignore_patterns=["*.bin"]  # 可选，如果你想强制忽略 .bin 文件
)

