import asyncio
import contextlib
import time
from dataclasses import dataclass, field


@dataclass
class BatchJob:
    text: str
    task_type: str
    dimensions: int
    max_length: int
    token_count: int
    future: asyncio.Future
    created_at: float = field(default_factory=time.monotonic)

    @property
    def key(self):
        return (self.task_type, self.dimensions, self.max_length)


class DynamicBatcher:
    def __init__(
        self,
        embedding_service,
        *,
        max_batch_size: int = 4,
        batch_timeout_ms: int = 5,
        max_batch_tokens: int = 8192,
        length_tolerance: float = 0.2,
        inference_gate: asyncio.Lock | None = None,
        cache_trimmer=None,
    ):
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if batch_timeout_ms < 0:
            raise ValueError("batch_timeout_ms must be >= 0")
        if max_batch_tokens < 1:
            raise ValueError("max_batch_tokens must be >= 1")
        if length_tolerance < 0:
            raise ValueError("length_tolerance must be >= 0")

        self.embedding_service = embedding_service
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout_ms / 1000
        self.max_batch_tokens = max_batch_tokens
        self.length_tolerance = length_tolerance
        self.inference_gate = inference_gate or asyncio.Lock()
        self.cache_trimmer = cache_trimmer
        self._active_jobs = 0
        self._pending: list[BatchJob] = []
        self._condition = asyncio.Condition()
        self._worker_task: asyncio.Task | None = None
        self._stopping = False

    async def start(self):
        if self._worker_task is None:
            self._stopping = False
            self._worker_task = asyncio.create_task(self._run(), name="dynamic-embedding-batcher")

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
            if not job.future.done():
                job.future.cancel()
        self._pending.clear()

    def queue_state(self) -> dict[str, int]:
        queued = len(self._pending)
        active = self._active_jobs
        return {
            "queued": queued,
            "active": active,
            "unfinished": queued + active,
        }

    async def embed(self, texts: list[str], *, task_type: str, dimensions: int, max_length: int) -> list[list[float]]:
        await self.start()
        loop = asyncio.get_running_loop()
        jobs = []
        prepared = []

        for text in texts:
            token_count = min(self.embedding_service.count_tokens([text], task_type), max_length)
            if token_count > self.max_batch_tokens:
                raise ValueError(
                    f"input token count {token_count} exceeds max_batch_tokens={self.max_batch_tokens}"
                )
            prepared.append((text, token_count))

        async with self._condition:
            for text, token_count in prepared:
                future = loop.create_future()
                job = BatchJob(
                    text=text,
                    task_type=task_type,
                    dimensions=dimensions,
                    max_length=max_length,
                    token_count=token_count,
                    future=future,
                )
                jobs.append(job)
                self._pending.append(job)
            self._condition.notify_all()

        return await asyncio.gather(*(job.future for job in jobs))

    async def _run(self):
        while not self._stopping:
            batch = await self._next_batch()
            if not batch:
                continue
            await self._process_batch(batch)

    async def _next_batch(self) -> list[BatchJob]:
        async with self._condition:
            while not self._pending and not self._stopping:
                await self._condition.wait()

            if self._stopping:
                return []

            if len(self._pending) < self.max_batch_size:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._wait_for_full_queue(), timeout=self.batch_timeout)

            return self._select_batch_locked()

    async def _wait_for_full_queue(self):
        while len(self._pending) < self.max_batch_size and not self._stopping:
            await self._condition.wait()

    def _select_batch_locked(self) -> list[BatchJob]:
        anchor = self._pending[0]
        compatible = [
            job
            for job in self._pending
            if job is anchor or (job.key == anchor.key and self._length_matches(anchor, job))
        ]
        compatible.sort(key=lambda job: (abs(job.token_count - anchor.token_count), job.created_at))

        batch: list[BatchJob] = []
        for job in compatible:
            if len(batch) >= self.max_batch_size:
                break
            if self._would_fit(batch, job):
                batch.append(job)

        selected = set(id(job) for job in batch)
        self._pending = [job for job in self._pending if id(job) not in selected]
        return batch

    def _length_matches(self, anchor: BatchJob, candidate: BatchJob) -> bool:
        lower = anchor.token_count * (1 - self.length_tolerance)
        upper = anchor.token_count * (1 + self.length_tolerance)
        return lower <= candidate.token_count <= upper

    def _would_fit(self, batch: list[BatchJob], candidate: BatchJob) -> bool:
        if not batch:
            return candidate.token_count <= self.max_batch_tokens
        max_tokens = max([job.token_count for job in batch] + [candidate.token_count])
        return max_tokens * (len(batch) + 1) <= self.max_batch_tokens

    async def _process_batch(self, batch: list[BatchJob]):
        self._active_jobs += len(batch)
        try:
            first = batch[0]
            async with self.inference_gate:
                embeddings = await asyncio.to_thread(
                    self.embedding_service.embed,
                    [job.text for job in batch],
                    task_type=first.task_type,
                    dimensions=first.dimensions,
                    max_length=first.max_length,
                )
        except Exception as error:
            for job in batch:
                if not job.future.done():
                    job.future.set_exception(error)
            return
        finally:
            self._active_jobs -= len(batch)
            self._trim_cache_if_idle()

        for job, embedding in zip(batch, embeddings):
            if not job.future.done():
                job.future.set_result(embedding)

    def _trim_cache_if_idle(self):
        if self.cache_trimmer is not None:
            self.cache_trimmer.trim_if_idle(self.queue_state())
