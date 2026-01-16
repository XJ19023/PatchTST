#!/bin/bash

find . -type f -name "result.txt" -path "*" | while read -r file; do
    total_lines=$(wc -l < "$file")

    # 文件行数小于2时跳过
    if [ "$total_lines" -lt 2 ]; then
        continue
    fi

    # 计算要删除的行号：倒数第二行
    delete_line=$((total_lines - 1))

    # 使用 sed 原地删除
    sed -i "${delete_line}d" "$file"

    echo "Processed: $file"
done
