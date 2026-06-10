import asyncio
import contextlib
from dataclasses import dataclass


@dataclass
class RerankJob:
    query: str
    documents: list[str]
    top_n: int | None
    return_embeddings: bool
    future: asyncio.Future | None = None


class RerankQueue:
    def __init__(self, rerank_service, *, inference_gate: asyncio.Lock, cache_trimmer=None):
        self.rerank_service = rerank_service
        self.inference_gate = inference_gate
        self.cache_trimmer = cache_trimmer
        self._pending: list[RerankJob] = []
        self._condition = asyncio.Condition()
        self._worker_task: asyncio.Task | None = None
        self._active_jobs = 0
        self._stopping = False

    async def start(self):
        if self._worker_task is None:
            self._stopping = False
            self._worker_task = asyncio.create_task(self._run(), name="rerank-queue-worker")

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

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
        return_embeddings: bool = False,
    ):
        await self.start()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        job = RerankJob(
            query=query,
            documents=documents,
            top_n=top_n,
            return_embeddings=return_embeddings,
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

    async def _next_job(self) -> RerankJob | None:
        async with self._condition:
            while not self._pending and not self._stopping:
                await self._condition.wait()
            if self._stopping:
                return None
            return self._pending.pop(0)

    async def _process_job(self, job: RerankJob):
        self._active_jobs += 1
        try:
            async with self.inference_gate:
                result = await asyncio.to_thread(
                    self.rerank_service.rerank,
                    job.query,
                    job.documents,
                    top_n=job.top_n,
                    return_embeddings=job.return_embeddings,
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
            self._trim_cache_if_idle()

    def _trim_cache_if_idle(self):
        if self.cache_trimmer is not None:
            self.cache_trimmer.trim_if_idle(self.queue_state())
