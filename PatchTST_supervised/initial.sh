#!/bin/bash

rm /root/.profile
# 添加 alias
echo 'alias cdh="cd /cephfs/juxin"' >> /root/.profile
echo 'alias du_sort="du -sh * | sort -hr"' >> /root/.profile
echo 'alias t="tmux"' >> /root/.profile
echo 'alias t_kill="tmux kill-window -t"' >> /root/.profile
echo 'alias ta="tmux a"' >> /root/.profile
echo 'alias cds="cd /cephfs/juxin/smoothquant"' >> /root/.profile
echo 'alias cdb="cd /cephfs/juxin/ANT-Quantization/olive_quantization/bert"' >> /root/.profile
echo 'alias cdq="cd /cephfs/juxin/QwT/QwT-cls-RepQ-ViT"' >> /root/.profile
echo 'alias conda_de="conda deactivate"' >> /root/.profile
echo 'alias conda_llama_codebook="conda activate llama_codebook && cd /cephfs/juxin/llama-codebook/getting-started/finetuning"' >> /root/.profile
echo 'alias conda_smoothquant="conda activate smoothquant && cd /cephfs/juxin/smoothquant"' >> /root/.profile
echo 'alias conda_sqzcomp="conda activate qwen && cd /cephfs/juxin/QwT-LLM/SqzComp/llm-qwt"' >> /root/.profile
echo 'alias conda_speculative="conda activate qwen && cd /cephfs/juxin/QwT-LLM/Speculative/algo"' >> /root/.profile
echo 'alias conda_mant="conda activate mant && cd /cephfs/juxin/QwT-LLM/Speculative/mant_algo"' >> /root/.profile
echo 'alias conda_sim="conda activate sim && cd /cephfs/juxin/QwT-LLM/Speculative/simulator"' >> /root/.profile
echo 'alias conda_ls="conda env list"' >> /root/.profile
echo 'alias test_curl="curl https://scholar.google.com"' >> /root/.profile
echo 'alias fuser="fuser -v /dev/nvidia*"' >> /root/.profile
# echo "set -g mouse on" >> /root/.tmux.conf
echo "bind -n C-k clear-history" >> /root/.tmux.conf


# echo 'export HF_ENDPOINT=https://hf-mirror.com' >> /root/.profile
# echo 'export HF_DATASETS_CACHE="/cephfs/shared/juxin/hf_cache"' >> /root/.profile

# 建立 python 链接
ln -sf /usr/bin/python3 /usr/bin/python

# 安装 nvitop
pip install nvitop -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install Ninja -i https://pypi.tuna.tsinghua.edu.cn/simple
apt install tmux
# pip uninstall evaluate -y
# pip install evaluate==0.3.0 -i https://pypi.tuna.tsinghua.edu.cn/simple


# 安装 smoothquant
# cd smoothquant && pip install -e .

# conda 换源
cat << 'EOF' > /root/.condarc
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud'
auto_activate_base: false
EOF

# pip 换源
mkdir -p /root/.pip
cat << 'EOF' > /root/.pip/pip.conf
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF

cat << 'EOF' > /root/.vimrc
set number

inoremap jk <Esc>
inoremap ( ()<ESC>i
inoremap [ []<ESC>i
inoremap { {}<ESC>i
inoremap " ""<ESC>i
inoremap ' ''<ESC>i
EOF


cat << 'EOF' >> /root/.profile
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/root/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/root/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/root/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

set_proxy() {
    export http_proxy="http://127.0.0.1:7897"
    export https_proxy="http://127.0.0.1:7897"
    echo "✅ Proxy set to 127.0.0.1:7897"
}

unset_proxy() {
    unset http_proxy  #HTTP
    unset https_proxy #HTTPS
    echo "❌ Proxy disabled"
}
EOF

cat << 'EOF' >> /root/.profile
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
EOF

cat << 'EOF' >> /root/.ssh/authorized_keys

ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDKoSwwrUP2hnF9eE9mF5WmNmAiNt/m4K+iJRRyPR3HWYgoBRnUN+Y/4KAShKsw+wBRj1JxBY17Hvs5hSkmPWogbS31l4CfFXFwfPgK57i7UhqINTWk7LEoNDrTpUD6geZmABbOBdcrsiqD11lm0ZZHDhsreS985si+6rnJed8uE3KZZFP8NnMveumbSF85Io+JVIY0iwAVHNCv9+UnXazSIMPGZ6C6B16snlohjY4QFiQaPuKqErMDJlNsxBC3b2fh0lqBPIsqg8e4t6Xnp9XnmkCgBnxLB1gU1cn6PgEImoOWf8XCO7j07/k2Qicv5lawhFNUskpqoDvbEvFP3VXNknXc4gq0ts9EKlVDtWRBw9g48omPkLiD2O578ULa0KNFeZEeXvpXKoLhPIG7pNQsIfB2EA1T42vfufjJWBg4BI3eDSly6/SPkeQvAiEwLFP8aZDPBZ4zf8EUGloHU58Wd0JNKPvR7dsyy8kThTunFZst7Xz/VuRG0cyMRtjGavE= 1183300812@qq.com
EOF



echo -e "\e[36m Please source /root/.profile \e[0m"


# pytorch 安装

