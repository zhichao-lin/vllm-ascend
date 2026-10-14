# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Mark a remote-prefill request that arrives after Decode has paused new work."""

from vllm.v1.core.sched.interface import PauseState


def mark_late_remote_prefill_abort(scheduler, request) -> None:
    """Set ``abort_immediately`` so EngineCore aborts after ``add_request``.

    Decode under ``PAUSED_NEW`` does not schedule a request that is still in
    ``waiting``. A request that still needs a remote prefill would then never
    reach the receive thread, and the prefiller would hold its blocks until
    the abort timeout. EngineCore already aborts when this flag is set, which
    runs the connector's existing empty-recv path and sends DONE.

    Only a request inserted while the scheduler is in ``PAUSED_NEW`` is marked.
    Ordinary requests, requests whose receive was already arranged, and
    requests that arrive while the scheduler is unpaused stay unchanged.
    """
    if request.is_finished() or scheduler._pause_state != PauseState.PAUSED_NEW:
        return
    params = request.kv_transfer_params
    if params is None or not params.get("do_remote_prefill"):
        return
    request.abort_immediately = True
