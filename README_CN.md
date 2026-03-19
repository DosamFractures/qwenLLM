# QwenTrain 中文说明

这是一个本地项目，用于：
- 在终端运行 `Qwen3-1.7B` 进行对话，并支持快捷切换模型；
- 基于 `data/` 目录中的 JSON 数据进行 QLoRA 微调。

[English README](./README.md)

## 1）环境准备

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 2）数据集格式

将一个或多个 `.json` 文件放到 `data/` 目录下。  
每个文件都必须是 JSON 数组，每条样本必须包含 `text` 和 `class` 字段。

示例：

```json
[
  {"text": "这是一条测试消息", "class": "1"},
  {"text": "这是另一条测试消息", "class": "0"}
]
```

## 3）启动终端对话（本地模型）

默认模型路径为 `model/Qwen3-1.7B`。

```bash
python -m qwentrain.chat_cli --model-path model/Qwen3-1.7B --local-files-only
```

如果你使用 CUDA，也可以启用 4bit 加载：

```bash
python -m qwentrain.chat_cli --model-path model/Qwen3-1.7B --local-files-only --load-in-4bit
```

常用对话命令：
- `/model <alias|path>` 动态切换模型
- `/aliases` 查看 `configs/models.json` 中配置的模型别名
- `/reset` 清空当前会话历史
- `/think on|off` 开关 thinking 模式（通过 prompt 模板控制）
- `/show_think on|off` 显示或隐藏 `<think>` 内容
- `/exit` 退出程序

## 4）执行 QLoRA 微调

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

## 5）将 LoRA 合并回底座模型（可选）

```bash
python -m qwentrain.merge_lora \
  --base-model-path model/Qwen3-1.7B \
  --adapter-path outputs/qwen3-1.7b-qlora/adapter \
  --output-dir outputs/merged-qwen3-1.7b \
  --local-files-only
```
