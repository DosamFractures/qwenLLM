from __future__ import annotations

"""QLoRA 微调入口。

本文件负责：
1. 读取 `data/*.json` 数据；
2. 将样本转换为聊天模板文本；
3. 使用 PEFT + QLoRA 训练适配器；
4. 输出 adapter 权重到 `outputs/.../adapter`。
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def load_records(data_path: str) -> list[dict[str, str]]:
    """加载数据集并标准化为 {text, class} 格式。

    兼容两种字段命名：
    - 新格式：`class`
    - 旧格式：`output`
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")

    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"No json files found in: {data_path}")

    records: list[dict[str, str]] = []
    for file in files:
        data = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{file} is not a JSON array.")
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"{file}[{idx}] is not an object.")
            if "text" not in item:
                raise ValueError(f"{file}[{idx}] must contain text.")
            if "class" in item:
                class_value = item["class"]
            elif "output" in item:
                # 兼容历史数据字段。
                class_value = item["output"]
            else:
                raise ValueError(f"{file}[{idx}] must contain class (or legacy output).")
            records.append({"text": str(item["text"]), "class": str(class_value)})
    return records


def apply_template(tokenizer: Any, text: str, class_label: str, enable_thinking: bool) -> str:
    """把单条样本映射成可训练的对话文本。"""
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "user", "content": text},
            {"role": "assistant", "content": class_label},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # 兼容旧 tokenizer（无 enable_thinking 参数）。
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
    return f"User: {text}\nAssistant: {class_label}"


def parse_target_modules(modules: str) -> list[str]:
    """解析 LoRA 目标层列表。"""
    return [x.strip() for x in modules.split(",") if x.strip()]


def train(args: argparse.Namespace) -> None:
    """执行 QLoRA 训练主流程。"""
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.load_in_4bit and not torch.cuda.is_available():
        raise RuntimeError("QLoRA 4-bit requires CUDA GPU with bitsandbytes.")

    records = load_records(args.data_path)
    if len(records) < 2 and args.validation_split > 0:
        raise ValueError("Need at least 2 records if validation_split > 0.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token

    raw_dataset = Dataset.from_list(records)
    if args.validation_split > 0:
        ds_split = raw_dataset.train_test_split(test_size=args.validation_split, seed=args.seed)
        train_raw = ds_split["train"]
        eval_raw = ds_split["test"]
    else:
        train_raw = raw_dataset
        eval_raw = None

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        texts = [
            apply_template(tokenizer, t, o, enable_thinking=args.enable_thinking_in_template)
            for t, o in zip(batch["text"], batch["class"])
        ]
        tokenized = tokenizer(texts, truncation=True, max_length=args.max_length, padding=False)
        tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]
        return tokenized

    train_dataset = train_raw.map(
        tokenize_batch,
        batched=True,
        remove_columns=train_raw.column_names,
        desc="Tokenizing train dataset",
    )
    eval_dataset = None
    if eval_raw is not None:
        eval_dataset = eval_raw.map(
            tokenize_batch,
            batched=True,
            remove_columns=eval_raw.column_names,
            desc="Tokenizing eval dataset",
        )

    quant_config = None
    if args.load_in_4bit:
        quant_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=quant_dtype,
        )

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.config.use_cache = False

    if args.load_in_4bit:
        # 量化训练前的标准准备步骤。
        model = prepare_model_for_kbit_training(model)
    elif torch.cuda.is_available():
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=parse_target_modules(args.target_modules),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir / "adapter"

    bf16_enabled = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16_enabled = torch.cuda.is_available() and not bf16_enabled

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        lr_scheduler_type=args.lr_scheduler_type,
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        per_device_eval_batch_size=args.per_device_eval_batch_size if eval_dataset is not None else None,
        optim="paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        bf16=bf16_enabled,
        fp16=fp16_enabled,
        report_to="none",
        gradient_checkpointing=True,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Training completed. Adapter saved to: {adapter_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    """构建微调脚本参数。"""
    parser = argparse.ArgumentParser(description="QLoRA fine-tune for Qwen3 and compatible CausalLM.")
    parser.add_argument("--model-path", type=str, default="model/Qwen3.5-2B")
    parser.add_argument("--data-path", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/qwen3.5-2b-qlora")

    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable 4-bit loading. QLoRA path is enabled by default.",
    )
    parser.add_argument("--enable-thinking-in-template", action="store_true")

    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine")

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    return parser


def main() -> None:
    """程序入口。"""
    parser = build_arg_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

