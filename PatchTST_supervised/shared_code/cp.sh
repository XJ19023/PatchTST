#!/bin/bash

start_time=$(date +%s)  # 记录开始时间
# -------------------------------------------------------------------------
cpdir_with_progress() {
    local src_dir="$1"
    local dst_dir="$2"

    if [[ ! -d "$src_dir" ]]; then
        echo "❌ Source directory '$src_dir' does not exist."
        return 1
    fi

    mkdir -p "$dst_dir"

    echo "📂 Copying all files from '$src_dir' to '$dst_dir'..."

    # find "$src_dir" -type f ! -name "*.bin" ! -name "*.tmp" | while read -r src_file; do
    find "$src_dir" -type f | while read -r src_file; do
        # 保留原始文件夹结构
        rel_path="${src_file#$src_dir/}"
        dst_file="$dst_dir/$rel_path"
        dst_subdir=$(dirname "$dst_file")
        mkdir -p "$dst_subdir"

        # 显示 tqdm 进度
        if [[ -f "$src_file" ]]; then
            size=$(stat -c %s "$src_file")
            echo "📦 Copying '$rel_path'  [$(numfmt --to=iec $size)]"
            cat "$src_file" | tqdm --bytes --total "$size" > "$dst_file"
        fi
    done
}

# cpdir_with_progress /cephfs/shared/impact/LLM_models/Qwen2.5/Qwen2.5-0.5B /localssd/lbxj/Qwen2.5-0.5B
# cpdir_with_progress /cephfs/shared/impact/LLM_models/TinyLlama-1.1B-Chat-v1.0 /localssd/lbxj/TinyLlama-1.1B-Chat-v1.0
# cpdir_with_progress /cephfs/shared/impact/LLM_models/llama-2-7b-hf /localssd/lbxj/llama-2-7b-hf
# cpdir_with_progress /cephfs/shared/juxin/models/Llama-2-13b-hf /localssd/lbxj/Llama-2-13b-hf

mkdir -p /localssd/lbxj
echo "Copying opt-125m"
cp -r /cephfs/shared/juxin/models/opt-125m /localssd/lbxj/opt-125m
# echo "Copying opt-350m"
# cp -r /cephfs/shared/juxin/models/opt-350m /localssd/lbxj/opt-350m
# echo "Copying Qwen3-0.6B"
# cp -r /cephfs/shared/juxin/models/Qwen3-0.6B /localssd/lbxj/Qwen3-0.6B
# echo "Copying llama-2-7b-hf"
# cp -r /cephfs/shared/impact/LLM_models/llama-2-7b-hf /localssd/lbxj/llama-2-7b-hf
# echo "Copying Meta-Llama-3-8B"
# cp -r /cephfs/shared/fangchao/models/Meta-Llama-3-8B /localssd/lbxj/Meta-Llama-3-8B
# echo "Copying Qwen2.5-7B"
# cp -r /cephfs/shared/impact/LLM_models/Qwen2.5/Qwen2.5-7B /localssd/lbxj/Qwen2.5-7B
# echo "Copying Qwen3-8B"
# cp -r /cephfs/shared/impact/LLM_models/Qwen3-8B /localssd/lbxj/Qwen3-8B
# echo "Copying Mistral-7B-v0.1"
# cp -r /cephfs/shared/juxin/models/Mistral-7B-v0.1 /localssd/lbxj/Mistral-7B-v0.1
# echo "Copying Llama-2-13b-hf"
# cp -r /cephfs/shared/juxin/models/Llama-2-13b-hf /localssd/lbxj/Llama-2-13b-hf
# echo "Copying Qwen3-14B"
# cp -r /cephfs/shared/impact/LLM_models/Qwen3/Qwen3-14B /localssd/lbxj/Qwen3-14B

# cp /cephfs/shared/juxin/dataset/val.jsonl.zst /localssd/lbxj/
# echo "Copying TinyLlama-1.1B-Chat-v1.0"
# cp -r /cephfs/shared/impact/LLM_models/TinyLlama-1.1B-Chat-v1.0 /localssd/lbxj/TinyLlama-1.1B-Chat-v1.0
# echo "Copying Qwen2.5-0.5B"
# cp -r /cephfs/shared/impact/LLM_models/Qwen2.5/Qwen2.5-0.5B /localssd/lbxj/Qwen2.5-0.5B
# echo "Copying Qwen2.5-1.5B"
# cp -r /cephfs/shared/impact/LLM_models/Qwen2.5/Qwen2.5-1.5B /localssd/lbxj/Qwen2.5-1.5B
# echo "Copying llama-7b-hf"
# cp -r /cephfs/shared/impact/LLM_models/Llama-1/llama-7b-hf /localssd/lbxj/llama-7b-hf
# echo "Copying llama-30b-hf"
# cp -r /cephfs/shared/impact/LLM_models/Llama-1/llama-30b-hf /localssd/lbxj/llama-30b-hf
# echo "Copying llama-13b-hf"
# cp -r /cephfs/shared/impact/LLM_models/Llama-1/llama-13b-hf /localssd/lbxj/llama-13b-hf
# -------------------------------------------------------------------------
end_time=$(date +%s)  # 记录结束时间
duration=$((end_time - start_time))
# 格式化输出为小时-分钟-秒
printf "RUNNING TIME: %02dh-%02dm-%02ds\n\n" $((duration / 3600)) $(((duration % 3600) / 60)) $((duration % 60))


