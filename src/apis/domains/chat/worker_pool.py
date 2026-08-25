from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
from time import perf_counter

from anyio import Event, create_memory_object_stream, create_task_group, to_thread
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from loguru import logger

from src.apis.domains.chat.schemas import ChatRequest, ChatResponse
from src.apis.domains.chat.service import handle_chat
from src.core_ai.nextrip_agent.answer_generation import SupportsAnswerGeneration
from src.core_ai.nextrip_agent.conversation import (
    SupportsConversationContextualization,
)
from src.infra.kb_client import KbClient
from src.infra.current_data_client import CurrentDataClient
from src.infra.chat_store import ChatStore
from src.infra.user_profile_store import UserProfileStore
from src.infra.weather import OpenMeteoWeatherClient
from src.shared.request_context import current_request_id


ChatHandler = Callable[
    [ChatRequest, KbClient, SupportsAnswerGeneration | None],
    ChatResponse,
]


@dataclass
class ChatJob:
    request: ChatRequest
    kb_client: KbClient
    answer_generator: SupportsAnswerGeneration | None
    request_id: str
    queued_at: float = field(default_factory=perf_counter)
    completed: Event = field(default_factory=Event)
    result: ChatResponse | None = None
    error: Exception | None = None


class ChatWorkerPool:
    def __init__(
        self,
        *,
        worker_count: int,
        queue_capacity: int,
        handler: ChatHandler = handle_chat,
        weather_client: OpenMeteoWeatherClient | None = None,
        chat_store: ChatStore | None = None,
        chat_history_limit: int = 8,
        conversation_contextualizer: SupportsConversationContextualization
        | None = None,
        user_profile_store: UserProfileStore | None = None,
        current_data_client: CurrentDataClient | None = None,
    ) -> None:
        if worker_count < 1:
            raise ValueError("worker_count must be positive")
        if queue_capacity < 1:
            raise ValueError("queue_capacity must be positive")
        self.worker_count = worker_count
        self.queue_capacity = queue_capacity
        self._handler = handler
        self._weather_client = weather_client
        self._chat_store = chat_store
        self._chat_history_limit = chat_history_limit
        self._conversation_contextualizer = conversation_contextualizer
        self._user_profile_store = user_profile_store
        self._current_data_client = current_data_client
        self._send_stream: MemoryObjectSendStream[ChatJob] | None = None
        self._active_jobs = 0

    @asynccontextmanager
    async def run(self) -> AsyncIterator["ChatWorkerPool"]:
        if self._send_stream is not None:
            raise RuntimeError("Chat worker pool is already running")

        send_stream, receive_stream = create_memory_object_stream[ChatJob](
            self.queue_capacity
        )
        self._send_stream = send_stream
        try:
            async with create_task_group() as task_group:
                for worker_id in range(1, self.worker_count + 1):
                    task_group.start_soon(
                        self._worker,
                        worker_id,
                        receive_stream.clone(),
                        name=f"chat-worker-{worker_id}",
                    )
                await receive_stream.aclose()
                logger.info(
                    "Chat worker pool started workers={} queue_capacity={}",
                    self.worker_count,
                    self.queue_capacity,
                )
                try:
                    yield self
                finally:
                    await send_stream.aclose()
                    logger.info(
                        "Chat worker pool stopping active_jobs={}",
                        self._active_jobs,
                    )
        finally:
            self._send_stream = None

    async def submit(
        self,
        request: ChatRequest,
        kb_client: KbClient,
        answer_generator: SupportsAnswerGeneration | None,
    ) -> ChatResponse:
        send_stream = self._send_stream
        if send_stream is None:
            raise RuntimeError("Chat worker pool is not running")

        job = ChatJob(
            request=request,
            kb_client=kb_client,
            answer_generator=answer_generator,
            request_id=current_request_id(),
        )
        await send_stream.send(job)
        statistics = send_stream.statistics()
        logger.info(
            "Chat job queued session_id={} queued_jobs={} active_jobs={}",
            request.session_id,
            statistics.current_buffer_used,
            self._active_jobs,
        )
        await job.completed.wait()
        if job.error is not None:
            raise job.error
        if job.result is None:
            raise RuntimeError("Chat worker completed without a result")
        return job.result

    def statistics(self) -> dict[str, int]:
        if self._send_stream is None:
            return {
                "workers": self.worker_count,
                "active_jobs": 0,
                "queued_jobs": 0,
                "queue_capacity": self.queue_capacity,
            }
        stream_statistics = self._send_stream.statistics()
        return {
            "workers": self.worker_count,
            "active_jobs": self._active_jobs,
            "queued_jobs": stream_statistics.current_buffer_used,
            "queue_capacity": self.queue_capacity,
        }

    async def _worker(
        self,
        worker_id: int,
        receive_stream: MemoryObjectReceiveStream[ChatJob],
    ) -> None:
        async with receive_stream:
            async for job in receive_stream:
                self._active_jobs += 1
                queue_wait_ms = int((perf_counter() - job.queued_at) * 1000)
                with logger.contextualize(request_id=job.request_id):
                    logger.info(
                        "Chat worker start worker_id={} session_id={} queue_wait_ms={}",
                        worker_id,
                        job.request.session_id,
                        queue_wait_ms,
                    )
                    try:
                        if (
                            self._weather_client is None
                            and self._chat_store is None
                            and self._current_data_client is None
                        ):
                            handler = partial(
                                self._handler,
                                job.request,
                                job.kb_client,
                                job.answer_generator,
                            )
                        else:
                            handler = partial(
                                self._handler,
                                job.request,
                                job.kb_client,
                                job.answer_generator,
                                weather_client=self._weather_client,
                                chat_store=self._chat_store,
                                chat_history_limit=self._chat_history_limit,
                                conversation_contextualizer=self._conversation_contextualizer,
                                user_profile_store=self._user_profile_store,
                                current_data_client=self._current_data_client,
                            )
                        job.result = await to_thread.run_sync(
                            handler,
                            abandon_on_cancel=False,
                        )
                    except Exception as exc:
                        job.error = exc
                        logger.exception(
                            "Chat worker error worker_id={} session_id={} error_type={}",
                            worker_id,
                            job.request.session_id,
                            exc.__class__.__name__,
                        )
                    finally:
                        self._active_jobs -= 1
                        job.completed.set()
                        logger.info(
                            "Chat worker end worker_id={} session_id={} active_jobs={}",
                            worker_id,
                            job.request.session_id,
                            self._active_jobs,
                        )
