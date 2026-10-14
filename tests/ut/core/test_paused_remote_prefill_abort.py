# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.request import Request, RequestStatus

import vllm_ascend.patch.platform.patch_paused_remote_prefill  # noqa: F401
from vllm_ascend.core.paused_remote_prefill import mark_late_remote_prefill_abort
from vllm_ascend.core.recompute_scheduler import RecomputeScheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
    MooncakeConnectorScheduler,
)
from vllm_ascend.patch.platform.patch_balance_schedule import BalanceScheduler

_REMOTE_PARAMS = {
    "do_remote_prefill": True,
    "do_remote_decode": False,
    "remote_block_ids": ([1, 2, 3],),
    "remote_engine_id": "remote",
    "remote_request_id": "remote_req1",
    "remote_host": "localhost",
    "remote_port": 5000,
}


def _request(request_id: str, kv_transfer_params: dict | None) -> Request:
    extra_args = None if kv_transfer_params is None else {"kv_transfer_params": kv_transfer_params}
    return Request(
        request_id=request_id,
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=8, extra_args=extra_args),
        pooling_params=None,
    )


def _stub_scheduler(cls, pause_state: PauseState):
    scheduler = cls.__new__(cls)
    scheduler.requests = {}
    scheduler.waiting = []
    scheduler.connector = None
    scheduler.log_stats = False
    scheduler._pause_state = pause_state
    scheduler.finished_req_ids = set()
    scheduler.finish_calls = []

    def enqueue_waiting_request(self, request):
        self.waiting.append(request)

    def finish_requests(self, request_ids, finished_status):
        self.finish_calls.append((list(request_ids), finished_status))
        for request_id in request_ids:
            request = self.requests.pop(request_id)
            self.waiting.remove(request)
            request.status = finished_status
            if self.connector is not None:
                self.connector.request_finished(request, ([],))
            self.finished_req_ids.add(request_id)

    def has_requests(self) -> bool:
        return bool(self.requests) or bool(self.finished_req_ids)

    scheduler._enqueue_waiting_request = MethodType(enqueue_waiting_request, scheduler)
    scheduler.finish_requests = MethodType(finish_requests, scheduler)
    scheduler.has_requests = MethodType(has_requests, scheduler)
    return scheduler


def _assert_new_remote_prefill_is_marked(scheduler) -> None:
    remote = _request("remote", dict(_REMOTE_PARAMS))
    plain = _request("plain", None)

    scheduler.add_request(remote)
    scheduler.add_request(plain)

    assert remote.abort_immediately is True
    assert remote.status == RequestStatus.WAITING
    assert remote in scheduler.waiting
    assert plain.abort_immediately is False
    assert plain in scheduler.waiting

    remote.streaming_queue = deque()
    again = _request("remote", dict(_REMOTE_PARAMS))
    scheduler.add_request(again)

    assert remote.abort_immediately is True
    assert again.abort_immediately is False
    assert scheduler.requests["remote"] is remote


def test_mark_helper_cases():
    def make(pause_state, params, finished=False):
        request = SimpleNamespace(
            is_finished=lambda: finished,
            kv_transfer_params=params,
            abort_immediately=False,
        )
        scheduler = SimpleNamespace(_pause_state=pause_state)
        mark_late_remote_prefill_abort(scheduler, request)
        return request

    marked = make(PauseState.PAUSED_NEW, {"do_remote_prefill": True})
    assert marked.abort_immediately is True

    unchanged = [
        make(PauseState.PAUSED_NEW, None),
        make(PauseState.PAUSED_NEW, {"do_remote_prefill": False}),
        make(PauseState.UNPAUSED, {"do_remote_prefill": True}),
        make(PauseState.PAUSED_ALL, {"do_remote_prefill": True}),
        make(PauseState.PAUSED_NEW, {"do_remote_prefill": True}, finished=True),
    ]
    assert all(request.abort_immediately is False for request in unchanged)


def test_balance_scheduler_add_request_marks_only_new_remote_prefill():
    balance = BalanceScheduler.__new__(BalanceScheduler)
    async_sched = AsyncScheduler.__new__(AsyncScheduler)
    assert getattr(balance.add_request.__func__, "_marks_late_remote_prefill_abort", False)
    assert getattr(async_sched.add_request.__func__, "_marks_late_remote_prefill_abort", False)
    assert "add_request" not in BalanceScheduler.__dict__
    assert "add_request" not in AsyncScheduler.__dict__

    scheduler = _stub_scheduler(BalanceScheduler, PauseState.PAUSED_NEW)
    _assert_new_remote_prefill_is_marked(scheduler)
    assert scheduler.finish_calls == []


def test_recompute_scheduler_add_request_marks_only_new_remote_prefill():
    scheduler = _stub_scheduler(RecomputeScheduler, PauseState.PAUSED_NEW)
    scheduler.is_mtp_kv_consumer = False
    _assert_new_remote_prefill_is_marked(scheduler)
    assert scheduler.finish_calls == []


def test_mark_then_existing_abort_enqueues_empty_recv():
    class _KVCacheGroup:
        def __init__(self):
            self.layer_names = ["model.layers.0.self_attn"]
            self.kv_cache_spec = MagicMock()

    class _KVCacheConfig:
        def __init__(self):
            self.kv_cache_groups = [_KVCacheGroup()]
            self.num_blocks = 10

    vllm_config = MagicMock()
    vllm_config.cache_config.block_size = 16
    vllm_config.kv_transfer_config.kv_port = 5000
    vllm_config.parallel_config.prefill_context_parallel_size = 1
    vllm_config.parallel_config.decode_context_parallel_size = 1
    vllm_config.parallel_config.tensor_parallel_size = 2
    vllm_config.parallel_config.data_parallel_size = 1
    vllm_config.parallel_config.data_parallel_rank = 0
    vllm_config.parallel_config.pipeline_parallel_size = 1
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = True
    vllm_config.model_config.hf_config = SimpleNamespace()

    with (
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.init_ascend_config"),
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.get_ascend_config",
            return_value=MagicMock(),
        ),
    ):
        connector = MooncakeConnectorScheduler(vllm_config, "test_engine", _KVCacheConfig())

    scheduler = _stub_scheduler(BalanceScheduler, PauseState.PAUSED_NEW)
    scheduler.connector = connector

    plain = _request("plain", None)
    remote = _request("remote", dict(_REMOTE_PARAMS))
    scheduler.add_request(plain)
    scheduler.add_request(remote)

    assert remote.abort_immediately is True
    assert remote.status == RequestStatus.WAITING
    assert scheduler.finish_calls == []

    scheduler.finish_requests([remote.request_id], RequestStatus.FINISHED_ABORTED)

    assert scheduler.finish_calls == [(["remote"], RequestStatus.FINISHED_ABORTED)]
    assert remote.status == RequestStatus.FINISHED_ABORTED
    assert remote not in scheduler.waiting
    assert "remote" not in scheduler.requests
    assert scheduler.finished_req_ids == {"remote"}
    assert scheduler.has_requests() is True
    assert remote.kv_transfer_params["do_remote_prefill"] is False
    assert connector._reqs_need_recv["remote"][1] == ([],)

    metadata = connector.build_connector_meta(SimpleNamespace())
    assert metadata.requests["remote"].local_block_ids == ([],)
    assert metadata.requests["remote"].num_external_tokens == 0
    assert metadata.requests["remote"].remote_request_id == "remote_req1"
    assert connector._reqs_need_recv == {}

    assert plain.abort_immediately is False
    assert plain.status == RequestStatus.WAITING
    assert plain in scheduler.waiting
    assert scheduler.requests["plain"] is plain
