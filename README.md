# QwenTrain

[中文文档](./README_CN.md)

Simple local project for:
- terminal chat with `Qwen3-1.7B` (and easy model switching),
- QLoRA fine-tuning from JSON dataset files in `data/`.

## 1) Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 2) Dataset Format

Put one or more `.json` files under `data/`.
Each file must be a JSON array and each item must include `text` and `class`.

Example:

```json
[
  {"text": "This is a test message", "class": "1"},
  {"text": "Another test message", "class": "0"}
]
```

## 3) Run Terminal Chat (Local Model)

Default model path is `model/Qwen3-1.7B`.

```bash
python -m qwentrain.chat_cli --model-path model/Qwen3-1.7B --local-files-only
```

Optional CUDA 4-bit loading:

```bash
python -m qwentrain.chat_cli --model-path model/Qwen3-1.7B --local-files-only --load-in-4bit
```

Useful chat commands:
- `/model <alias|path>` switch model on the fly
- `/aliases` show aliases from `configs/models.json`
- `/reset` clear conversation
- `/think on|off` toggle thinking mode in prompt template
- `/show_think on|off` show or hide `<think>` content
- `/exit` quit

## 4) QLoRA Fine-Tuning

Run QLoRA training from `data/`:

```bash
python -m qwentrain.train_qlora \
  --model-path model/Qwen3-1.7B \
  --data-path data \
  --output-dir outputs/qwen3-1.7b-qlora \
  --local-files-only
```

The LoRA adapter will be saved to:

`outputs/qwen3-1.7b-qlora/adapter`

Notes:
- `--load-in-4bit` is enabled by default (QLoRA path).
- If you need full precision LoRA fallback, add `--no-load-in-4bit`.

## 5) Merge Adapter into Base Model (Optional)

```bash
python -m qwentrain.merge_lora \
  --base-model-path model/Qwen3-1.7B \
  --adapter-path outputs/qwen3-1.7b-qlora/adapter \
  --output-dir outputs/merged-qwen3-1.7b \
  --local-files-only
```
