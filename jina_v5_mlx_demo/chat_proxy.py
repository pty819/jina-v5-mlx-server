import asyncio
from contextlib import AbstractAsyncContextManager
from typing import Any

import httpx


CHAT_MODEL_ID = "mlx-community/Hy-MT2-1.8B-4bit"
DEFAULT_CHAT_UPSTREAM_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_CHAT_UPSTREAM_MODEL = CHAT_MODEL_ID


class ChatProxyError(RuntimeError):
    def __init__(self, status_code: int, payload: Any):
        super().__init__(str(payload))
        self.status_code = status_code
        self.payload = payload


class ChatProxyClient:
    def __init__(
        self,
        *,
        upstream_base_url: str = DEFAULT_CHAT_UPSTREAM_BASE_URL,
        model_id: str = CHAT_MODEL_ID,
        upstream_model: str | None = DEFAULT_CHAT_UPSTREAM_MODEL,
        api_key: str | None = None,
        timeout_seconds: float = 600.0,
    ):
        self.upstream_base_url = upstream_base_url.rstrip("/")
        self.model_id = model_id
        self.upstream_model = upstream_model
        self.api_key = api_key
        self._active = 0
        self._active_lock = asyncio.Lock()
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds, connect=10.0),
        )

    def queue_state(self) -> dict[str, int]:
        return {
            "queued": 0,
            "active": self._active,
            "unfinished": self._active,
        }

    async def complete(self, payload: dict) -> dict:
        body = self._prepare_payload(payload)
        await self._increment_active()
        try:
            try:
                response = await self._client.post(
                    self._url("/chat/completions"),
                    json=body,
                    headers=self._headers(),
                )
            except httpx.RequestError as error:
                raise ChatProxyError(502, {"error": {"message": str(error), "type": "upstream_unavailable"}}) from error
            await self._raise_for_upstream_error(response)
            return response.json()
        finally:
            await self._decrement_active()

    async def open_stream(self, payload: dict):
        body = self._prepare_payload(payload)
        await self._increment_active()
        stream_context = self._client.stream(
            "POST",
            self._url("/chat/completions"),
            json=body,
            headers=self._headers(),
        )
        try:
            response = await stream_context.__aenter__()
        except httpx.RequestError as error:
            await self._decrement_active()
            raise ChatProxyError(502, {"error": {"message": str(error), "type": "upstream_unavailable"}}) from error
        if response.status_code >= 400:
            try:
                await self._raise_for_upstream_error(response)
            finally:
                await stream_context.__aexit__(None, None, None)
                await self._decrement_active()
        return self._iter_stream(response, stream_context)

    async def close(self):
        await self._client.aclose()

    def _prepare_payload(self, payload: dict) -> dict:
        body = dict(payload)
        if self.upstream_model:
            body["model"] = self.upstream_model
        elif "model" not in body:
            body["model"] = self.model_id
        return body

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json, text/event-stream"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _url(self, path: str) -> str:
        return f"{self.upstream_base_url}{path}"

    async def _iter_stream(
        self,
        response: httpx.Response,
        stream_context: AbstractAsyncContextManager[httpx.Response],
    ):
        try:
            try:
                async for chunk in response.aiter_raw():
                    if chunk:
                        yield chunk
            except httpx.StreamConsumed:
                if response.content:
                    yield response.content
        finally:
            await stream_context.__aexit__(None, None, None)
            await self._decrement_active()

    async def _raise_for_upstream_error(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        await response.aread()
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            payload: Any = response.json()
        else:
            payload = {"error": {"message": response.text}}
        raise ChatProxyError(response.status_code, payload)

    async def _increment_active(self):
        async with self._active_lock:
            self._active += 1

    async def _decrement_active(self):
        async with self._active_lock:
            self._active -= 1
