import gc
import threading
from pathlib import Path

import mlx.core as mx

from jina_v5_mlx_demo.idle_evictor import IdleEvictor


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHAT_MODEL_DIR = PROJECT_DIR / "models" / "Hy-MT2-1.8B-4bit"
CHAT_MODEL_ID = "mlx-community/Hy-MT2-1.8B-4bit"
DEFAULT_IDLE_SECONDS = 30 * 60
STREAM_CHUNK_TYPE = "content"
STREAM_FINAL_TYPE = "final"


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
        chunks = list(self.stream_complete(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        ))
        text = "".join(chunk["content"] for chunk in chunks if chunk["type"] == STREAM_CHUNK_TYPE)
        final = next(chunk for chunk in reversed(chunks) if chunk["type"] == STREAM_FINAL_TYPE)
        return {
            "content": text,
            "prompt_tokens": final["prompt_tokens"],
            "completion_tokens": final["completion_tokens"],
            "finish_reason": final["finish_reason"],
            "prompt_tps": final["prompt_tps"],
            "generation_tps": final["generation_tps"],
            "peak_memory_gb": final["peak_memory_gb"],
        }

    def stream_complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: str | list[str] | None,
    ):
        model, tokenizer = self._load()
        self._evictor.touch()
        normalized = _normalize_messages(messages)
        prompt = self._render_prompt(tokenizer, normalized)
        prompt_tokens = self._count_tokens(tokenizer, prompt)
        stop_values = _normalize_stop(stop)

        from mlx_lm.generate import stream_generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=temperature, top_p=top_p)
        emitted_until = 0
        text = ""
        final_response = None
        finish_reason = "stop"

        with self._generate_lock:
            for response in stream_generate(
                model,
                tokenizer,
                prompt,
                max_tokens=max_tokens,
                sampler=sampler,
            ):
                final_response = response
                text += response.text
                stop_index = _first_stop_index(text, stop_values)
                if stop_index is not None:
                    if stop_index > emitted_until:
                        yield {
                            "type": STREAM_CHUNK_TYPE,
                            "content": text[emitted_until:stop_index],
                        }
                    emitted_until = stop_index
                    finish_reason = "stop"
                    break

                safe_end = _safe_stream_end(text, emitted_until, stop_values)
                if safe_end > emitted_until:
                    yield {
                        "type": STREAM_CHUNK_TYPE,
                        "content": text[emitted_until:safe_end],
                    }
                    emitted_until = safe_end
                if response.finish_reason is not None:
                    finish_reason = response.finish_reason

            if final_response is None:
                finish_reason = "stop"
                completion_tokens = 0
                prompt_tps = 0.0
                generation_tps = 0.0
                peak_memory = mx.get_peak_memory() / 1e9
            else:
                if emitted_until < len(text):
                    yield {
                        "type": STREAM_CHUNK_TYPE,
                        "content": text[emitted_until:],
                    }
                completion_tokens = final_response.generation_tokens
                prompt_tps = final_response.prompt_tps
                generation_tps = final_response.generation_tps
                peak_memory = final_response.peak_memory
            mx.eval(mx.array([0]))

        yield {
            "type": STREAM_FINAL_TYPE,
            "finish_reason": finish_reason,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "prompt_tps": prompt_tps,
            "generation_tps": generation_tps,
            "peak_memory_gb": peak_memory,
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


def _normalize_stop(stop: str | list[str] | None) -> list[str]:
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, list):
        return [value for value in stop if value]
    return []


def _first_stop_index(text: str, stop_values: list[str]) -> int | None:
    first_stop = None
    for value in stop_values:
        index = text.find(value)
        if index != -1 and (first_stop is None or index < first_stop):
            first_stop = index
    return first_stop


def _safe_stream_end(text: str, emitted_until: int, stop_values: list[str]) -> int:
    if not stop_values:
        return len(text)
    hold = max(len(value) for value in stop_values) - 1
    return max(emitted_until, len(text) - hold)
