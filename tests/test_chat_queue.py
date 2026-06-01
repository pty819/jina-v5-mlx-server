import asyncio
import threading
import unittest

from jina_v5_mlx_demo.chat_queue import ChatQueue


class RecordingChatService:
    model_id = "fake-chat-model"

    def __init__(self):
        self.calls = []

    def complete(self, messages, *, max_tokens, temperature, top_p, stop):
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": stop,
            }
        )
        return {
            "content": messages[-1]["content"].upper(),
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "finish_reason": "stop",
        }

    def stream_complete(self, messages, *, max_tokens, temperature, top_p, stop):
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": stop,
            }
        )
        yield {"type": "content", "content": messages[-1]["content"]}
        yield {
            "type": "final",
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "finish_reason": "stop",
        }


class BlockingChatService(RecordingChatService):
    def __init__(self, started, release):
        super().__init__()
        self.started = started
        self.release = release

    def complete(self, messages, *, max_tokens, temperature, top_p, stop):
        self.started.set()
        self.release.wait(timeout=1)
        return super().complete(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )


class ChatQueueTest(unittest.IsolatedAsyncioTestCase):
    async def test_returns_result_to_original_request(self):
        service = RecordingChatService()
        queue = ChatQueue(service, inference_gate=asyncio.Lock())
        await queue.start()

        first, second = await asyncio.gather(
            queue.complete(
                [{"role": "user", "content": "alpha"}],
                max_tokens=16,
                temperature=0.0,
                top_p=1.0,
                stop=None,
            ),
            queue.complete(
                [{"role": "user", "content": "beta"}],
                max_tokens=16,
                temperature=0.0,
                top_p=1.0,
                stop=None,
            ),
        )

        self.assertEqual(first["content"], "ALPHA")
        self.assertEqual(second["content"], "BETA")
        await queue.stop()

    async def test_reports_active_while_processing(self):
        started = threading.Event()
        release = threading.Event()
        queue = ChatQueue(
            BlockingChatService(started, release),
            inference_gate=asyncio.Lock(),
        )
        await queue.start()

        task = asyncio.create_task(
            queue.complete(
                [{"role": "user", "content": "alpha"}],
                max_tokens=16,
                temperature=0.0,
                top_p=1.0,
                stop=None,
            )
        )

        await asyncio.to_thread(started.wait, 1)
        self.assertEqual(queue.queue_state()["active"], 1)
        self.assertEqual(queue.queue_state()["unfinished"], 1)

        release.set()
        await task
        self.assertEqual(queue.queue_state()["unfinished"], 0)
        await queue.stop()

    async def test_stream_returns_chunks_to_original_request(self):
        service = RecordingChatService()
        queue = ChatQueue(service, inference_gate=asyncio.Lock())
        await queue.start()

        chunks = []
        async for chunk in queue.stream(
            [{"role": "user", "content": "alpha"}],
            max_tokens=16,
            temperature=0.0,
            top_p=1.0,
            stop=None,
        ):
            chunks.append(chunk)

        self.assertEqual(chunks[0]["content"], "alpha")
        self.assertEqual(chunks[-1]["finish_reason"], "stop")
        await queue.stop()


if __name__ == "__main__":
    unittest.main()
