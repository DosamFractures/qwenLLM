# QwenTrain

[中文文档](./README_CN.md)

Simple local project for:
- terminal chat with local Qwen models (easy model switching),
- QLoRA fine-tuning from JSON dataset files in `data/`.

## 1) Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Notes:
- `--load-in-4bit` needs CUDA GPU and bitsandbytes.
- On CPU/MPS, start chat without `--load-in-4bit`.

## 2) Dataset Format

Put one or more `.json` files under `data/`.
Each file must be a JSON array and each item must include `text` and `output`.

Example:

```json
[
  {"text": "This is a test message", "output": "1"},
  {"text": "Another test message", "output": "0"}
]
```

## 3) Run Terminal Chat (Local Model)

Recommended:

```bash
python -m qwentrain.chat_cli -model qwen3.5-2b --local-files-only
```

Run with explicit path:

```bash
python -m qwentrain.chat_cli -model model/Qwen3.5-2B --local-files-only
```

Optional CUDA 4-bit loading:

```bash
python -m qwentrain.chat_cli -model qwen3.5-2b --local-files-only --load-in-4bit
```

Useful chat commands:
- `/model <alias|path>` switch model on the fly
- `/aliases` show aliases from `configs/models.json`
- `/reset` clear conversation
- `/think on|off` toggle thinking mode in prompt template
- `/show_think on|off` show or hide `<think>` content
- `/exit` quit

Model aliases are defined in `configs/models.json`, for example:
- `qwen3-1.7b -> model/Qwen3-1.7B`
- `qwen3.5-2b -> model/Qwen3.5-2B`

## 4) Run in PyCharm

Preferred run mode:
- Run as module: `qwentrain.chat_cli` (same behavior as `python -m qwentrain.chat_cli`)

If running script file directly:
- Script path: `qwentrain/chat_cli.py`
- Parameters: `-model qwen3.5-2b --local-files-only`
- Working directory: project root (`.../Qwentrain`)

If working directory is not project root, use absolute model path:
- `-model /Users/beefnoodle/IdeaProjects/Qwentrain/model/Qwen3.5-2B`

## 5) QLoRA Fine-Tuning

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

## 6) Merge Adapter into Base Model (Optional)

```bash
python -m qwentrain.merge_lora \
  --base-model-path model/Qwen3-1.7B \
  --adapter-path outputs/qwen3-1.7b-qlora/adapter \
  --output-dir outputs/merged-qwen3-1.7b \
  --local-files-only
```

