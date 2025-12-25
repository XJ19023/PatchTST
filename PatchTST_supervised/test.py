lst = [5, 3, 8, 2, 4]

min_idx, min_val = min(enumerate(lst), key=lambda x: x[1])

print(min_val, min_idx)   # 2 3
