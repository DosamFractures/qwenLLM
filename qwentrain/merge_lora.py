from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def merge(args: argparse.Namespace) -> None:
    dtype = parse_dtype(args.dtype)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )

    peft_model = PeftModel.from_pretrained(base_model, args.adapter_path)
    merged_model = peft_model.merge_and_unload()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print(f"Merged model saved to: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    parser.add_argument("--base-model-path", type=str, default="model/Qwen3.5-2B")
    parser.add_argument("--adapter-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/merged-model")
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    merge(args)


if __name__ == "__main__":
    main()
