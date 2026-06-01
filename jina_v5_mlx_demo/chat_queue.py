import asyncio
import contextlib
from dataclasses import dataclass

from jina_v5_mlx_demo.schema import RequestError


@dataclass
class ChatJob:
    messages: list[dict]
    max_tokens: int
    temperature: float
    top_p: float
    stop: str | list[str] | None
    stream_queue: asyncio.Queue | None = None
    loop: asyncio.AbstractEventLoop | None = None
    future: asyncio.Future | None = None
    cancelled: bool = False


class ChatQueue:
    def __init__(self, chat_service, *, inference_gate: asyncio.Lock, max_queue_size: int = 32):
        self.chat_service = chat_service
        self.inference_gate = inference_gate
        self.max_queue_size = max_queue_size
        self._pending: list[ChatJob] = []
        self._condition = asyncio.Condition()
        self._worker_task: asyncio.Task | None = None
        self._active_jobs = 0
        self._stopping = False

    async def start(self):
        if self._worker_task is None:
            self._stopping = False
            self._worker_task = asyncio.create_task(self._run(), name="chat-queue-worker")

    async def stop(self):
        self._stopping = True
        async with self._condition:
            self._condition.notify_all()
        if self._worker_task is not None:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
            self._worker_task = None
        for job in self._pending:
            if job.future and not job.future.done():
                job.future.cancel()
            if job.stream_queue is not None:
                await job.stream_queue.put({"type": "error", "error": "chat queue stopped"})
                await job.stream_queue.put({"type": "done"})
        self._pending.clear()

    async def complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: str | list[str] | None,
    ):
        await self.start()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        job = ChatJob(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            future=future,
        )
        await self._enqueue(job)
        return await future

    async def stream(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: str | list[str] | None,
    ):
        stream = await self.open_stream(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
        async for item in stream:
            yield item

    async def open_stream(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: str | list[str] | None,
    ):
        await self.start()
        queue = asyncio.Queue()
        job = ChatJob(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            stream_queue=queue,
            loop=asyncio.get_running_loop(),
        )
        await self._enqueue(job)
        return self._stream_items(job, queue)

    async def _stream_items(self, job: ChatJob, queue: asyncio.Queue):
        try:
            while True:
                item = await queue.get()
                if item["type"] == "done":
                    break
                if item["type"] == "error":
                    raise RuntimeError(item["error"])
                yield item
        except GeneratorExit:
            job.cancelled = True
            raise
        except asyncio.CancelledError:
            job.cancelled = True
            raise

    async def _enqueue(self, job: ChatJob) -> None:
        async with self._condition:
            if len(self._pending) >= self.max_queue_size:
                raise RequestError("chat queue is full", status=503)
            self._pending.append(job)
            self._condition.notify_all()

    def queue_state(self) -> dict[str, int]:
        queued = len(self._pending)
        active = self._active_jobs
        return {
            "queued": queued,
            "active": active,
            "unfinished": queued + active,
        }

    async def _run(self):
        while not self._stopping:
            job = await self._next_job()
            if job is None:
                continue
            await self._process_job(job)

    async def _next_job(self) -> ChatJob | None:
        async with self._condition:
            while not self._pending and not self._stopping:
                await self._condition.wait()
            if self._stopping:
                return None
            return self._pending.pop(0)

    async def _process_job(self, job: ChatJob):
        self._active_jobs += 1
        try:
            if job.stream_queue is not None:
                await self._process_stream_job(job)
            else:
                async with self.inference_gate:
                    result = await asyncio.to_thread(
                        self.chat_service.complete,
                        job.messages,
                        max_tokens=job.max_tokens,
                        temperature=job.temperature,
                        top_p=job.top_p,
                        stop=job.stop,
                    )
                future = job.future
                if future is not None and not future.done():
                    future.set_result(result)
        except Exception as error:
            future = job.future
            if future is not None and not future.done():
                future.set_exception(error)
            if job.stream_queue is not None:
                await job.stream_queue.put({"type": "error", "error": str(error)})
                await job.stream_queue.put({"type": "done"})
        finally:
            self._active_jobs -= 1

    async def _process_stream_job(self, job: ChatJob) -> None:
        async with self.inference_gate:
            await asyncio.to_thread(self._run_stream_in_thread, job)

    def _run_stream_in_thread(self, job: ChatJob) -> None:
        if job.stream_queue is None or job.loop is None:
            return
        try:
            for chunk in self.chat_service.stream_complete(
                job.messages,
                max_tokens=job.max_tokens,
                temperature=job.temperature,
                top_p=job.top_p,
                stop=job.stop,
            ):
                if job.cancelled:
                    break
                asyncio.run_coroutine_threadsafe(job.stream_queue.put(chunk), job.loop).result()
        except Exception as error:
            asyncio.run_coroutine_threadsafe(
                job.stream_queue.put({"type": "error", "error": str(error)}),
                job.loop,
            ).result()
        finally:
            asyncio.run_coroutine_threadsafe(
                job.stream_queue.put({"type": "done"}),
                job.loop,
            ).result()
