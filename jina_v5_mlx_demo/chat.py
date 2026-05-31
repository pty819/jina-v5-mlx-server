import gc
import threading
from pathlib import Path

import mlx.core as mx

from jina_v5_mlx_demo.idle_evictor import IdleEvictor


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHAT_MODEL_DIR = PROJECT_DIR / "models" / "Hy-MT2-1.8B-4bit"
CHAT_MODEL_ID = "mlx-community/Hy-MT2-1.8B-4bit"
DEFAULT_IDLE_SECONDS = 30 * 60


def _message_content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                raise ValueError("Only text chat message content is supported")
        return "".join(parts)
    return str(content)


def _normalize_messages(messages: list[dict]) -> list[dict]:
    normalized = []
    for message in messages:
        normalized.append({
            "role": message["role"],
            "content": _message_content_to_text(message.get("content")),
        })
    return normalized


def _fallback_chat_prompt(messages: list[dict]) -> str:
    role_prefix = {
        "system": "SYSTEM: ",
        "user": "USER: ",
        "assistant": "ASSISTANT: ",
        "tool": "TOOL: ",
    }
    prompt = ""
    for message in messages:
        prompt += f"{role_prefix.get(message['role'], '')}{message['content']}\n"
    return f"{prompt}ASSISTANT:".rstrip()


class MLXChatService:
    def __init__(
        self,
        model_dir: Path = DEFAULT_CHAT_MODEL_DIR,
        *,
        idle_seconds: int = DEFAULT_IDLE_SECONDS,
    ):
        self.model_id = CHAT_MODEL_ID
        self.model_dir = Path(model_dir)
        self._model = None
        self._tokenizer = None
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self._evictor = IdleEvictor(
            evict=self._evict_model,
            idle_seconds=idle_seconds,
        )

    def complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: str | list[str] | None,
    ) -> dict:
        model, tokenizer = self._load()
        self._evictor.touch()
        normalized = _normalize_messages(messages)
        prompt = self._render_prompt(tokenizer, normalized)

        from mlx_lm.generate import generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=temperature, top_p=top_p)
        with self._generate_lock:
            text = generate(
                model,
                tokenizer,
                prompt,
                verbose=False,
                max_tokens=max_tokens,
                sampler=sampler,
            )
            mx.eval(mx.array([0]))

        text, finish_reason = self._apply_stop(text, stop, max_tokens, tokenizer)
        prompt_tokens = self._count_tokens(tokenizer, prompt)
        completion_tokens = self._count_tokens(tokenizer, text)
        return {
            "content": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "finish_reason": finish_reason,
        }

    def close(self):
        self._evictor.stop()
        self._evict_model()

    def _render_prompt(self, tokenizer, messages: list[dict]):
        if getattr(tokenizer, "has_chat_template", False):
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except TypeError:
                return tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                )
        return _fallback_chat_prompt(messages)

    def _count_tokens(self, tokenizer, value) -> int:
        if isinstance(value, list):
            return len(value)
        return len(tokenizer.encode(str(value)))

    def _apply_stop(self, text: str, stop: str | list[str] | None, max_tokens: int, tokenizer):
        stop_values = []
        if isinstance(stop, str):
            stop_values = [stop]
        elif isinstance(stop, list):
            stop_values = stop

        first_stop = None
        for value in stop_values:
            if value and value in text:
                index = text.index(value)
                if first_stop is None or index < first_stop:
                    first_stop = index

        if first_stop is not None:
            return text[:first_stop], "stop"

        completion_tokens = self._count_tokens(tokenizer, text)
        return text, "length" if completion_tokens >= max_tokens else "stop"

    def _evict_model(self):
        with self._load_lock:
            with self._generate_lock:
                if self._model is not None:
                    self._model = None
                    self._tokenizer = None
                    gc.collect()
                    mx.clear_cache()
                    mx.synchronize()

    def _load(self):
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                return self._model, self._tokenizer

            if not self.model_dir.exists():
                raise FileNotFoundError(
                    f"Chat model not found at {self.model_dir}. Run: "
                    "uv run hf download mlx-community/Hy-MT2-1.8B-4bit "
                    "--local-dir models/Hy-MT2-1.8B-4bit"
                )

            from mlx_lm.utils import load

            self._model, self._tokenizer = load(str(self.model_dir))
            self._evictor.start()
            return self._model, self._tokenizer
