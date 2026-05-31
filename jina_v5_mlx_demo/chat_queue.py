import asyncio
import contextlib
from dataclasses import dataclass


@dataclass
class ChatJob:
    messages: list[dict]
    max_tokens: int
    temperature: float
    top_p: float
    stop: str | list[str] | None
    future: asyncio.Future | None = None


class ChatQueue:
    def __init__(self, chat_service, *, inference_gate: asyncio.Lock):
        self.chat_service = chat_service
        self.inference_gate = inference_gate
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
        async with self._condition:
            self._pending.append(job)
            self._condition.notify_all()
        return await future

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
        finally:
            self._active_jobs -= 1
