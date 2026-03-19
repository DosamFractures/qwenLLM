from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def detect_device(preferred: str) -> str:
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype_name: str, device: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32

    # auto mode
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def load_model_aliases(config_path: str | None) -> dict[str, str]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("models config must be a JSON object of alias:path.")
    aliases: dict[str, str] = {}
    for alias, model_path in data.items():
        aliases[str(alias)] = str(model_path)
    return aliases


def is_qwen_like_tokenizer(tokenizer: Any) -> bool:
    return hasattr(tokenizer, "apply_chat_template")


def extract_thinking_and_answer(raw_text: str) -> tuple[str | None, str]:
    text = raw_text.strip()
    text = text.replace("<|im_end|>", "").strip()
    text = text.replace("<|endoftext|>", "").strip()
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant", 1)[-1].strip()

    think_match = re.search(r"<think>\s*(.*?)\s*</think>\s*(.*)", text, flags=re.S)
    if think_match:
        thinking = think_match.group(1).strip()
        answer = think_match.group(2).strip()
        return thinking or None, answer
    return None, text


class ChatSession:
    def __init__(
        self,
        model_path: str,
        model_aliases: dict[str, str],
        device: str,
        dtype_name: str,
        load_in_4bit: bool,
        trust_remote_code: bool,
        local_files_only: bool,
        enable_thinking: bool,
        show_thinking: bool,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> None:
        self.model_aliases = model_aliases
        self.device = device
        self.dtype_name = dtype_name
        self.load_in_4bit = load_in_4bit
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        self.enable_thinking = enable_thinking
        self.show_thinking = show_thinking
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.history: list[dict[str, str]] = []
        self.model_path = ""

        self.tokenizer = None
        self.model = None
        self.model_dtype = None

        self.reload_model(model_path, clear_history=True)

    def resolve_model_path(self, model_or_alias: str) -> str:
        return self.model_aliases.get(model_or_alias, model_or_alias)

    def reload_model(self, model_or_alias: str, clear_history: bool = True) -> None:
        from transformers import BitsAndBytesConfig

        resolved = self.resolve_model_path(model_or_alias)
        path = Path(resolved)
        if not path.exists() and resolved not in self.model_aliases.values():
            raise FileNotFoundError(f"Model path not found: {resolved}")

        dtype = resolve_dtype(self.dtype_name, self.device)
        if self.device == "cpu" and dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            resolved,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.bos_token

        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }

        if self.load_in_4bit:
            if self.device != "cuda":
                raise RuntimeError("4-bit loading needs CUDA GPU and bitsandbytes.")
            quant_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=quant_dtype,
            )
            model_kwargs["device_map"] = "auto"
        elif self.device == "cuda":
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(resolved, **model_kwargs)
        if not self.load_in_4bit and self.device in {"cpu", "mps"}:
            self.model.to(self.device)

        self.model.eval()
        self.model_path = resolved
        self.model_dtype = dtype

        if clear_history:
            self.history = []

    def _build_prompt(self, messages: list[dict[str, str]]) -> str:
        if is_qwen_like_tokenizer(self.tokenizer):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

        turns = []
        for message in messages:
            role = message["role"].capitalize()
            turns.append(f"{role}: {message['content']}")
        turns.append("Assistant:")
        return "\n".join(turns)

    def _get_inputs_on_model_device(self, text: str) -> dict[str, torch.Tensor]:
        model_inputs = self.tokenizer([text], return_tensors="pt")
        if hasattr(self.model, "device"):
            target_device = self.model.device
            return {k: v.to(target_device) for k, v in model_inputs.items()}
        return model_inputs

    def ask(self, user_text: str) -> tuple[str | None, str]:
        self.history.append({"role": "user", "content": user_text})
        prompt = self._build_prompt(self.history)
        model_inputs = self._get_inputs_on_model_device(prompt)
        prompt_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][prompt_len:]
        raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
        thinking, answer = extract_thinking_and_answer(raw_output)
        self.history.append({"role": "assistant", "content": answer})
        return thinking, answer


def run_chat(args: argparse.Namespace) -> None:
    aliases = load_model_aliases(args.models_config)
    session = ChatSession(
        model_path=args.model,
        model_aliases=aliases,
        device=detect_device(args.device),
        dtype_name=args.dtype,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        enable_thinking=args.enable_thinking,
        show_thinking=args.show_thinking,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("=" * 72)
    print("QwenTrain Chat")
    print(f"Model: {session.model_path}")
    print(f"Device: {session.device} | dtype: {session.model_dtype} | 4bit: {session.load_in_4bit}")
    print("Type /help for commands.")
    print("=" * 72)

    while True:
        try:
            text = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not text:
            continue

        if text.startswith("/"):
            cmd, *rest = text.split(maxsplit=1)
            arg = rest[0] if rest else ""

            if cmd in {"/exit", "/quit"}:
                print("Bye.")
                break

            if cmd == "/help":
                print("Commands:")
                print("  /help                 show commands")
                print("  /exit                 quit")
                print("  /reset                clear chat history")
                print("  /model <alias|path>   switch model")
                print("  /think on|off         enable/disable thinking mode in prompt")
                print("  /show_think on|off    show/hide <think> output")
                print("  /max_new_tokens <n>   set max output tokens")
                print("  /temp <float>         set temperature")
                print("  /top_p <float>        set top-p")
                print("  /aliases              print model aliases from config")
                continue

            if cmd == "/reset":
                session.history = []
                print("History cleared.")
                continue

            if cmd == "/aliases":
                if not session.model_aliases:
                    print("No aliases configured.")
                else:
                    for alias, model_path in session.model_aliases.items():
                        print(f"{alias}: {model_path}")
                continue

            if cmd == "/model":
                if not arg:
                    print("Usage: /model <alias|path>")
                    continue
                session.reload_model(arg, clear_history=True)
                print(f"Switched model to: {session.model_path}")
                continue

            if cmd == "/think":
                if arg not in {"on", "off"}:
                    print("Usage: /think on|off")
                    continue
                session.enable_thinking = arg == "on"
                print(f"enable_thinking={session.enable_thinking}")
                continue

            if cmd == "/show_think":
                if arg not in {"on", "off"}:
                    print("Usage: /show_think on|off")
                    continue
                session.show_thinking = arg == "on"
                print(f"show_thinking={session.show_thinking}")
                continue

            if cmd == "/max_new_tokens":
                session.max_new_tokens = int(arg)
                print(f"max_new_tokens={session.max_new_tokens}")
                continue

            if cmd == "/temp":
                session.temperature = float(arg)
                print(f"temperature={session.temperature}")
                continue

            if cmd == "/top_p":
                session.top_p = float(arg)
                print(f"top_p={session.top_p}")
                continue

            print(f"Unknown command: {cmd}")
            continue

        thinking, answer = session.ask(text)
        if thinking and session.show_thinking:
            print(f"\nThink>\n{thinking}")
        print(f"\nAssistant> {answer}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Terminal chat for local Qwen/LLM models.")
    parser.add_argument(
        "-model",
        "--model",
        "--model-path",
        dest="model",
        type=str,
        default="model/Qwen3.5-2B",
        help="Target model path or alias to run.",
    )
    parser.add_argument("--models-config", type=str, default="configs/models.json")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--load-in-4bit", action="store_true", help="Requires CUDA + bitsandbytes.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--show-thinking", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_chat(args)


if __name__ == "__main__":
    main()
