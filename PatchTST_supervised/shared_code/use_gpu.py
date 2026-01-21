import torch
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="GPU Matrix Multiplication")
    parser.add_argument('--M', type=int, default=4, help='Number of rows in matrix A')
    parser.add_argument('--K', type=int, default=4, help='Number of columns in A / rows in B')
    parser.add_argument('--N', type=int, default=4, help='Number of columns in matrix B')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化矩阵
    A = torch.randn(args.M, args.K, device=device)
    # B = torch.randn(args.K, args.N, device=device)

    # # 执行矩阵乘法
    print("Start dummy computation loop. Press Ctrl+C to stop.")
    try:
        while True:
            # C = torch.matmul(A, B)
            C = A * 1
            A = C.detach()
            # time.sleep(1)
    except KeyboardInterrupt:
        print("Stopped by user.")

if __name__ == '__main__':
    main()
