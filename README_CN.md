# QwenTrain 中文说明

这是一个本地项目，用于：
- 在终端运行本地 Qwen 模型，并支持快捷切换模型；
- 基于 `data/` 目录中的 JSON 数据进行 QLoRA 微调。

[English README](./README.md)

## 1）环境准备

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

说明：
- `--load-in-4bit` 需要 CUDA GPU 和 bitsandbytes；
- 在 CPU/MPS 环境下，请不要加 `--load-in-4bit`。

## 2）数据集格式

将一个或多个 `.json` 文件放到 `data/` 目录下。  
每个文件都必须是 JSON 数组，每条样本必须包含 `text` 和 `output` 字段。

示例：

```json
[
  {"text": "这是一条测试消息", "output": "1"},
  {"text": "这是另一条测试消息", "output": "0"}
]
```

## 3）启动终端对话（本地模型）

推荐方式：

```bash
python -m qwentrain.chat_cli -model qwen3.5-2b --local-files-only
```

也可以传模型路径：

```bash
python -m qwentrain.chat_cli -model model/Qwen3.5-2B --local-files-only
```

如果你使用 CUDA，也可以启用 4bit 加载：

```bash
python -m qwentrain.chat_cli -model qwen3.5-2b --local-files-only --load-in-4bit
```

常用对话命令：
- `/model <alias|path>` 动态切换模型
- `/aliases` 查看 `configs/models.json` 中配置的模型别名
- `/reset` 清空当前会话历史
- `/think on|off` 开关 thinking 模式（通过 prompt 模板控制）
- `/show_think on|off` 显示或隐藏 `<think>` 内容
- `/exit` 退出程序

模型别名配置在 `configs/models.json`，例如：
- `qwen3-1.7b -> model/Qwen3-1.7B`
- `qwen3.5-2b -> model/Qwen3.5-2B`

## 4）在 PyCharm 中运行

推荐运行方式：
- 以模块方式运行：`qwentrain.chat_cli`（与 `python -m qwentrain.chat_cli` 行为一致）

如果直接运行脚本文件：
- Script path：`qwentrain/chat_cli.py`
- Parameters：`-model qwen3.5-2b --local-files-only`
- Working directory：项目根目录（`.../Qwentrain`）

如果 Working directory 不是项目根目录，请使用绝对路径：
- `-model /Users/beefnoodle/IdeaProjects/Qwentrain/model/Qwen3.5-2B`

## 5）执行 QLoRA 微调

从 `data/` 目录读取数据并训练：

```bash
python -m qwentrain.train_qlora \
  --model-path model/Qwen3-1.7B \
  --data-path data \
  --output-dir outputs/qwen3-1.7b-qlora \
  --local-files-only
```

训练后的 LoRA 适配器默认保存在：

`outputs/qwen3-1.7b-qlora/adapter`

说明：
- `--load-in-4bit` 默认开启（QLoRA 路径）；
- 如果要改为非 4bit 的 LoRA 训练，可加 `--no-load-in-4bit`。

## 6）将 LoRA 合并回底座模型（可选）

```bash
python -m qwentrain.merge_lora \
  --base-model-path model/Qwen3-1.7B \
  --adapter-path outputs/qwen3-1.7b-qlora/adapter \
  --output-dir outputs/merged-qwen3-1.7b \
  --local-files-only
```

