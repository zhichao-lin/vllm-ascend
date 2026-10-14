# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Abort a late remote-prefill request admitted while Decode is paused.

``EngineCore.add_request`` already finishes a request whose
``abort_immediately`` flag is set. This patch sets that flag for a request
that is inserted into the scheduler after ``PAUSED_NEW`` is in effect and
still has ``do_remote_prefill``. The scheduler itself does not finish the
request; EngineCore does that once ``add_request`` returns.

The wrap is installed on the original ``Scheduler`` class, not on
``BalanceScheduler``. ``AsyncScheduler`` subclasses the original class at
import time, and ``BalanceScheduler`` does not define ``add_request``, so
both pick up the wrapped method. ``RecomputeScheduler`` overrides
``add_request`` and calls the same helper itself.
"""

from vllm_ascend.core.paused_remote_prefill import mark_late_remote_prefill_abort

_MARK_ATTR = "_marks_late_remote_prefill_abort"


def _scheduler_class_owning_add_request():
    import vllm.v1.core.sched.scheduler as sched_mod

    # After the balance-schedule patch, the module attribute is
    # BalanceScheduler. Walk the MRO so the wrap lands on the class that
    # actually defines add_request.
    for candidate in sched_mod.Scheduler.__mro__:
        if candidate.__name__ == "Scheduler" and "add_request" in candidate.__dict__:
            return candidate
    raise RuntimeError("Could not find Scheduler.add_request to wrap")


def install_paused_remote_prefill_abort() -> None:
    owner = _scheduler_class_owning_add_request()
    original = owner.__dict__["add_request"]
    if getattr(original, _MARK_ATTR, False):
        return

    def add_request(self, request):
        is_new = request.request_id not in self.requests
        original(self, request)
        # Streaming updates hit an id that is already queued. Only a newly
        # inserted, still-unfinished request can be a late remote prefill.
        if is_new and not request.is_finished():
            mark_late_remote_prefill_abort(self, request)

    setattr(add_request, _MARK_ATTR, True)
    owner.add_request = add_request


install_paused_remote_prefill_abort()
