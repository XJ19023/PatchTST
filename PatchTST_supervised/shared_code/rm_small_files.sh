#!/bin/bash
start_time=$(date +%s)  # 记录开始时间
# -------------------------------------------------------------------------
TARGET_DIR="${1:-/root/miniconda3/pkgs}"
keep=(
  "pytorch-2.5.0-py3.10_cuda12.1_cudnn9.1.0_0"
  "pytorch-2.5.0-py3.10_cuda12.1_cudnn9.1.0_0.tar.bz2"
  "mkl-2023.1.0-h213fc3f_46344"
  "libcublas-12.1.0.26-0.tar.bz2"
  "libcusparse-12.0.2.55-0"
  "torchtriton-3.1.0-py310.tar.bz2"
  "libnpp-12.0.2.50-0"
  "libcufft-11.0.2.4-0"
)

# 构造排除参数
exclude_args=()
for name in "${keep[@]}"; do
  exclude_args+=(! -name "$name")
done


# 进入目标目录
pushd "$TARGET_DIR" > /dev/null || { echo "Error: $TARGET_DIR not exist"; exit 1; }

# 删除不在 keep 列表中的文件和目录
find . -mindepth 1 -maxdepth 1 "${exclude_args[@]}" -exec rm -rfv {} +
du -sh *
# 返回原目录
popd > /dev/null
# -------------------------------------------------------------------------
end_time=$(date +%s)  # 记录结束时间
duration=$((end_time - start_time))
# 格式化输出为小时-分钟-秒
printf "RUNNING TIME: %02dh-%02dm-%02ds\n\n" $((duration / 3600)) $(((duration % 3600) / 60)) $((duration % 60))
