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

Default model used by scripts is `Qwen3.5-2B`.

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
  --model-path model/Qwen3.5-2B \
  --data-path data \
  --output-dir outputs/qwen3.5-2b-qlora \
  --local-files-only
```

The LoRA adapter will be saved to:

`outputs/qwen3.5-2b-qlora/adapter`

Notes:
- `--load-in-4bit` is enabled by default (QLoRA path).
- If you need full precision LoRA fallback, add `--no-load-in-4bit`.

## 6) Merge Adapter into Base Model (Optional)

```bash
python -m qwentrain.merge_lora \
  --base-model-path model/Qwen3.5-2B \
  --adapter-path outputs/qwen3.5-2b-qlora/adapter \
  --output-dir outputs/merged-qwen3.5-2b \
  --local-files-only
```

## 7) Offline Deployment (whl + Docker)

If your target server is offline, prepare wheel packages in an online environment first:

```bash
./prepare_offline_wheels.sh
```

By default, it uses `python:3.11-slim` and generates wheels for `linux/amd64`.  
If your offline server is ARM, set platform explicitly:

```bash
./prepare_offline_wheels.sh --docker-platform linux/arm64
```

This generates:
- `wheelhouse/` (offline wheel files)
- `requirements.offline.lock.txt` (locked dependency list)

Build offline image (default mode, installs from local wheels only):

```bash
./build_docker_image.sh qwentrain:offline
```

Online fallback build (debug only):

```bash
./build_docker_image.sh qwentrain:online --online
```

Export image for transfer:

```bash
docker save -o qwentrain_offline.tar qwentrain:offline
```

Load and run on offline server:

```bash
docker load -i qwentrain_offline.tar
docker run --rm -it -v /path/to/model:/opt/qwentrain/model qwentrain:offline
```

### ARM64 Recommended Flow (Verified)

If your target offline server is `arm64`, use:

```bash
./prepare_offline_wheels.sh --docker-platform linux/arm64
docker build --platform linux/arm64 --build-arg USE_OFFLINE_WHEELS=1 -t qwentrain:offline-arm64 -f Dockerfile .
docker save -o qwentrain_offline_arm64.tar qwentrain:offline-arm64
```

On the offline `arm64` server:

```bash
docker load -i qwentrain_offline_arm64.tar
docker run --rm -it -v /path/to/model:/opt/qwentrain/model qwentrain:offline-arm64
```
