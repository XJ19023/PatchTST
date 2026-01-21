
import re

log_path = "/cephfs/juxin/ANT-Quantization/olive_quantization/bert/log/bert-base/cola/123.log"  # 替换为你的文件路径

with open(log_path, "r") as f:
    log_content = f.read()

# 提取所有 matthews_correlation 的值
matches = re.findall(r"matthews_correlation['\"]?:\s*([0-9.]+)", log_content)

# 转为 float 类型
values = list(map(float, matches))

print( values)

