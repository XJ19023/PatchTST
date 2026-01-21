import os

# 生成文件路径
files_to_search = []
for i in ['weather', 'traffic', 'electricity', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
    for j in ['96', '192', '336', '720']:
        file_path = f"logs/{i}_336_{j}/result.txt"
        files_to_search.append(file_path)

# ========== 需要搜索的关键词 ==========
TARGET_STRING = "org"   # 你要匹配的字符串，可自行修改

# ========== 合并输出文件 ==========
MERGED_FILE = "logs/merged_results.txt"


def search_and_merge(files, target):
    with open(MERGED_FILE, "w", encoding="utf-8") as merged_f:
        for file in files:
            if not os.path.exists(file):
                print(f"[Warning] file not exist: {file}")
                continue

            # print(f"\n========== Searching in {file} ==========")
            merged_f.write(f"\n========== {file} ==========\n")

            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, start=1):
                    # 写入合并文件
                    merged_f.write(line)

                    # 搜索匹配字符串
                    if target in line:
                        print(f"{file:<35} | Line {line_num}: {line.strip()}")

    # print(f"\n所有文件已合并保存至: {MERGED_FILE}")


if __name__ == "__main__":
    search_and_merge(files_to_search, TARGET_STRING)
