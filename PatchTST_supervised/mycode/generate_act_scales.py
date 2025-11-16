import sys

# for p in sys.path:
#     print(p)
# exit()

import torch
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import argparse

from calibration import get_act_scales


def build_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    kwargs = {"torch_dtype": torch.float16, "device_map": "sequential"}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="/localssd/lbxj/TinyLlama-1.1B-Chat-v1.0", help="model name"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="act_scales/TinyLlama-1.1B-Chat-v1.0.pt",
        help="where to save the act scales",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/cephfs/shared/juxin/dataset/val.jsonl.zst",
        help="location of the calibration dataset, we use the validation set of the Pile dataset",
    )
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--seq_len", type=int, default=512)
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    
    model_path = '/localssd/lbxj/' + args.model_name
    model, tokenizer = build_model_and_tokenizer(model_path)

    if not os.path.exists(args.dataset_path):
        print(f"Cannot find the dataset at {args.dataset_path}")
        print("Please download the Pile dataset and put the validation set at the path")
        print(
            "You can download the validation dataset of the Pile at https://huggingface.co/datasets/mit-han-lab/pile-val-backup/resolve/main/val.jsonl.zst"
        )
        raise FileNotFoundError

    act_scales = get_act_scales(
        model, tokenizer, args.dataset_path, args.num_samples, args.seq_len
    )

    output_path = f'act_scales/{args.model_name}.pt'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(act_scales, output_path)


if __name__ == "__main__":
    main()
