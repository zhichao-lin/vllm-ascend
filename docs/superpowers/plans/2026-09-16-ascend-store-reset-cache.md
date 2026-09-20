# AscendStore reset_cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `AscendStoreConnector.reset_cache()` actually wipe Mooncake (join rank-0 Put/Get queues, `remove_all`, clear `load_specs`) and return real `True`/`False` so verl `reset_prefix_cache(reset_connector=True)` invalidates the pool after sleep/weight update.

**Architecture:** Tag the existing LookupKey ZMQ channel with `LOOKUP_MSG`/`RESET_MSG`. Scheduler `reset_cache()`: `getattr(self, "connector_scheduler", None) is None` → worker, return `None`; `use_layerwise` → `False` and no RPC; else `load_specs.clear()`, `_kv_cache_events = None`, then REQ/REP `RESET_MSG`. Rank-0 `LookupKeyServer` maps RESET to `KVPoolWorker.reset_store()` (unique worker entry: `request_queue.join()` on non-`None` send/recv threads, then `MooncakeBackend.reset()`, `finally` clear `_invalid_block_ids`). `MooncakeBackend.reset()` is `remove_all(force=True)` with TypeError fallback; ignore C++ return codes. Memcache/yuanrong and `use_layerwise=True` return `False`. Do not add a global coordinator; four engines each `remove_all` (idempotent).

**Tech Stack:** Python 3, unittest + pytest runner, ZMQ REQ/REP, Mooncake `MooncakeDistributedStore`, existing `tests/ut/distributed/ascend_store/_mock_deps.py`.

**Spec:** `docs/superpowers/specs/2026-09-15-ascend-store-reset-cache-design.md`

**Branch:** `pd-kvpool-fix` (this checkout). Do not touch `csrc/third_party/catlass`. Commits: author `zhichao <linzhichao2@huawei.com>`, no Cursor / Co-authored-by trailer.

**Depends on:** nothing. verl plan runs after this lands.

**How to run tests:** from repo root

```bash
python -m pytest tests/ut/distributed/ascend_store/ -v
```

**Do not:**

- Call `self.connector_scheduler is None` without `getattr` (worker `__init__` never assigns the attr).
- Wrap `m_store.reset()` in `bool()` (`bool(0)` is False; spec treats C++ `0` as success inside `MooncakeBackend.reset()` only).
- Mark `Backend.reset` `@abstractmethod` in Task 1 (memcache/yuanrong would become uninstantiable until Task 2).
- Call `Thread.join()` on transfer threads.
- Drop `while self.running:` when rewriting `LookupKeyServer.process_request`.
- Wrap `recv_multipart` in `except: continue` (socket `close()` would busy-loop). Catch handler/`send` only; `_handle_frames` already swallows `reset_store` errors.

---

## File map

| File | Responsibility |
| --- | --- |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/base.py` | `Backend.reset() -> bool` (Task 1: `NotImplementedError`; Task 2: `@abstractmethod`) |
| `.../backend/mooncake_backend.py` | Mooncake `reset()`: `remove_all(force=True)`, TypeError fallback, ignore return codes |
| `.../backend/memcache_backend.py` | `reset()` → `False` + ERROR `backend=memcache` |
| `.../backend/yuanrong_backend.py` | `reset()` → `False` + ERROR `backend=yuanrong` |
| `.../pool_scheduler.py` | Protocol constants; `LookupKeyClient.reset()`; tagged `lookup()`; `KVPoolScheduler.reset_store()` |
| `.../pool_worker.py` | `KVPoolWorker.reset_store()`: join queues, `m_store.reset()`, `finally` clear `_invalid_block_ids` |
| `.../ascend_store_connector.py` | `LookupKeyServer` tagged handler + full REP loop; `AscendStoreConnector.reset_cache()` |
| `tests/ut/distributed/ascend_store/test_backend.py` | Backend `reset()` cases |
| `tests/ut/distributed/ascend_store/test_pool_scheduler.py` | Client RESET / tagged LOOKUP / `reset_store()` lazy client + RPC exceptions |
| `tests/ut/distributed/ascend_store/test_pool_worker.py` | `reset_store()` join order, None threads, `finally` clear |
| `tests/ut/distributed/ascend_store/test_ascend_store_connector.py` | `reset_cache()` scheduler/worker/layerwise; LookupKeyServer RESET (incl. join-before-reset via real `reset_store`) |

---

### Task 1: MooncakeBackend.reset()

**Files:**
- Modify: `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/base.py`
- Modify: `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py`
- Test: `tests/ut/distributed/ascend_store/test_backend.py`

Do **not** put `@abstractmethod` on `reset` in this task. Memcache/Yuanrong stay instantiable until Task 2 implements them and then promotes the base method to abstract.

- [ ] **Step 1: Write the failing tests**

Append to `test_backend.py` (keep the `_mock_deps` import already at top of that file):

```python
class TestMooncakeBackendReset(unittest.TestCase):
    def _backend(self, store):
        backend = MooncakeBackend.__new__(MooncakeBackend)
        backend.store = store
        return backend

    def test_store_none_returns_true(self):
        self.assertTrue(self._backend(None).reset())

    def test_remove_all_force_true_ignores_positive_count(self):
        store = MagicMock()
        store.remove_all.return_value = 7
        self.assertTrue(self._backend(store).reset())
        store.remove_all.assert_called_once_with(force=True)

    def test_typeerror_falls_back_without_force(self):
        store = MagicMock()
        store.remove_all.side_effect = [TypeError("no force"), -800]
        self.assertTrue(self._backend(store).reset())
        self.assertEqual(store.remove_all.call_args_list[0].kwargs, {"force": True})
        self.assertEqual(store.remove_all.call_args_list[1].args, ())

    def test_negative_or_none_return_is_still_true(self):
        for ret in (-1, None, 0):
            store = MagicMock()
            store.remove_all.return_value = ret
            self.assertTrue(self._backend(store).reset(), msg=repr(ret))

    def test_exception_returns_false(self):
        store = MagicMock()
        store.remove_all.side_effect = RuntimeError("boom")
        self.assertFalse(self._backend(store).reset())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/ut/distributed/ascend_store/test_backend.py::TestMooncakeBackendReset -v`

Expected: FAIL with `AttributeError: 'MooncakeBackend' object has no attribute 'reset'`.

- [ ] **Step 3: Write minimal implementation**

In `backend/base.py`, add next to the other methods (**not** `@abstractmethod`):

```python
    def reset(self) -> bool:
        """Wipe backend metadata. True if the wipe succeeded or there was nothing to wipe."""
        raise NotImplementedError(f"{type(self).__name__}.reset() is not implemented")
```

In `mooncake_backend.py`, on `MooncakeBackend` after `get()`:

```python
    def reset(self) -> bool:
        if self.store is None:
            return True
        try:
            try:
                ret = self.store.remove_all(force=True)
            except TypeError:
                ret = self.store.remove_all()
            logger.info("MooncakeBackend.reset remove_all returned %r", ret)
            return True
        except Exception:
            logger.exception("MooncakeBackend.reset failed")
            return False
```

Do not branch on `ret`. Do not close or re-`setup()` the store. Do not call `ensure_initialized()` just to wipe.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ut/distributed/ascend_store/test_backend.py::TestMooncakeBackendReset -v`

Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add \
  vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/base.py \
  vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py \
  tests/ut/distributed/ascend_store/test_backend.py
GIT_AUTHOR_NAME='zhichao' GIT_AUTHOR_EMAIL='linzhichao2@huawei.com' \
git commit -m "$(cat <<'EOF'
feat: add MooncakeBackend.reset via remove_all

EOF
)"
```

No Cursor trailer. Do not `git add` `csrc/third_party/catlass`.

---

### Task 2: Memcache / Yuanrong reset False

**Files:**
- Modify: `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/base.py`
- Modify: `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py`
- Modify: `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/yuanrong_backend.py`
- Test: `tests/ut/distributed/ascend_store/test_backend.py`

Log the extra_config backend name (`memcache` / `yuanrong`), not the class name. Spec ERROR: sleep/wake reset_cache currently unsupported for `backend=memcache` / `backend=yuanrong`.

- [ ] **Step 1: Write the failing tests**

```python
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
    MemcacheBackend,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.yuanrong_backend import (
    YuanrongBackend,
)


class TestUnsupportedBackendReset(unittest.TestCase):
    def test_memcache_reset_returns_false(self):
        backend = MemcacheBackend.__new__(MemcacheBackend)
        self.assertFalse(backend.reset())

    def test_yuanrong_reset_returns_false(self):
        backend = YuanrongBackend.__new__(YuanrongBackend)
        self.assertFalse(backend.reset())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/ut/distributed/ascend_store/test_backend.py::TestUnsupportedBackendReset -v`

Expected: FAIL (`NotImplementedError` from base `reset`).

- [ ] **Step 3: Write minimal implementation**

`MemcacheBackend.reset`:

```python
    def reset(self) -> bool:
        logger.error(
            "sleep/wake reset_cache currently unsupported for backend=memcache"
        )
        return False
```

`YuanrongBackend.reset`:

```python
    def reset(self) -> bool:
        logger.error(
            "sleep/wake reset_cache currently unsupported for backend=yuanrong"
        )
        return False
```

Then in `backend/base.py`, replace the Task 1 `NotImplementedError` body with `@abstractmethod` (all three subclasses now implement `reset`):

```python
    @abstractmethod
    def reset(self) -> bool:
        """Wipe backend metadata. True if the wipe succeeded or there was nothing to wipe."""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ut/distributed/ascend_store/test_backend.py::TestUnsupportedBackendReset tests/ut/distributed/ascend_store/test_backend.py::TestMooncakeBackendReset tests/ut/distributed/ascend_store/test_backend.py::TestBackendABC -v`

Expected: PASS. `TestBackendABC.test_cannot_instantiate` still fails to construct `Backend` (`assertRaises(TypeError)` still passes).

- [ ] **Step 5: Commit**

```bash
git add \
  vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/base.py \
  vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py \
  vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/yuanrong_backend.py \
  tests/ut/distributed/ascend_store/test_backend.py
GIT_AUTHOR_NAME='zhichao' GIT_AUTHOR_EMAIL='linzhichao2@huawei.com' \
git commit -m "$(cat <<'EOF'
feat: reject memcache/yuanrong sleep-wake reset

EOF
)"
```

---

### Task 3: LookupKey protocol constants + client reset/lookup tags

**Files:**
- Modify: `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`
- Test: `tests/ut/distributed/ascend_store/test_pool_scheduler.py`

Constants live **only** in `pool_scheduler.py`. `ascend_store_connector.py` will import them in Task 4. Do not copy. Do not let `pool_scheduler` import `ascend_store_connector`.

Extend the **existing** `pool_scheduler` import at the top of `test_pool_scheduler.py` (do not add a second import of `LookupKeyClient`):

```python
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import (
    KVPoolScheduler,
    LOOKUP_MSG,
    LookupKeyClient,
    RESET_MSG,
    RESP_ERR,
    RESP_OK,
    get_zmq_rpc_path_lookup,
)
```

- [ ] **Step 1: Write the failing tests**

In `test_pool_scheduler.py` update `TestLookupKeyClient.test_lookup` expected first frame, and add:

```python
class TestLookupKeyClientReset(unittest.TestCase):
    def _client(self, recv_bytes):
        with (
            patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.make_zmq_socket") as mock_sock,
            patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.zmq"),
        ):
            socket = MagicMock()
            socket.recv.return_value = recv_bytes
            mock_sock.return_value = socket
            config = MagicMock()
            config.parallel_config.data_parallel_rank = 0
            config.kv_transfer_config.kv_connector_extra_config = {}
            client = LookupKeyClient(config)
            client.socket = socket
            return client, socket

    def test_reset_ok(self):
        client, socket = self._client(RESP_OK)
        self.assertTrue(client.reset())
        socket.send.assert_called_once_with(RESET_MSG)

    def test_reset_err_or_garbage_is_false(self):
        for payload in (RESP_ERR, b"", (32).to_bytes(4, "big"), b"nope"):
            client, _ = self._client(payload)
            self.assertFalse(client.reset(), msg=repr(payload))

    def test_reset_exception_is_false(self):
        client, socket = self._client(RESP_OK)
        socket.send.side_effect = RuntimeError("zmq down")
        self.assertFalse(client.reset())
```

Change existing `test_lookup` frames list so index 0 is `LOOKUP_MSG` and the old 4-byte `token_len` is frame 1 (keep the existing `@patch` decorators and encoder mock):

```python
        self.assertEqual(
            frames,
            [
                LOOKUP_MSG,
                (64).to_bytes(4, "big"),
                b"groups",
                (16).to_bytes(4, "big"),
                b"hashes",
            ],
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestLookupKeyClient tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestLookupKeyClientReset -v`

Expected: FAIL (`ImportError` / `AttributeError` for `LOOKUP_MSG` / `reset`).

- [ ] **Step 3: Write minimal implementation**

At module level in `pool_scheduler.py` (above `LookupKeyClient`):

```python
LOOKUP_MSG = b"lookup"
RESET_MSG = b"reset"
RESP_OK = b"\x01"
RESP_ERR = b"\x00"
```

`LookupKeyClient.lookup`: prepend `LOOKUP_MSG` to `all_frames` before the existing `send_multipart(..., copy=False)`. Do not drop `copy=False`.

`LookupKeyClient.reset`:

```python
    def reset(self) -> bool:
        try:
            self.socket.send(RESET_MSG)
            resp = self.socket.recv()
            return bytes(resp) == RESP_OK
        except Exception:
            logger.exception("LookupKeyClient.reset failed")
            return False
```

Do not add a timeout. Do not treat `linger` as a timeout. Equality is `bytes(resp) == RESP_OK`, never `!= RESP_ERR`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestLookupKeyClient tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestLookupKeyClientReset -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add \
  vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py \
  tests/ut/distributed/ascend_store/test_pool_scheduler.py
GIT_AUTHOR_NAME='zhichao' GIT_AUTHOR_EMAIL='linzhichao2@huawei.com' \
git commit -m "$(cat <<'EOF'
feat: tag LookupKey ZMQ with LOOKUP/RESET

EOF
)"
```

---

### Task 4: LookupKeyServer RESET handler + worker reset_store

**Files:**
- Modify: `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py`
- Modify: `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`
- Test: `tests/ut/distributed/ascend_store/test_ascend_store_connector.py`
- Test: `tests/ut/distributed/ascend_store/test_pool_worker.py`

Spec §3 sequence (join → `m_store.reset()` → always clear `_invalid_block_ids` → ACK) runs inside `KVPoolWorker.reset_store()`, which is the unique worker entry. `LookupKeyServer` only calls `reset_store()` and maps `ok is True` to `RESP_OK`. Join is **not** duplicated in the server handler.

Extract request handling into `LookupKeyServer._handle_frames(all_frames) -> bytes` so tests do not start the REP thread. Keep the `while self.running:` loop.

- [ ] **Step 1: Write the failing tests**

In `test_pool_worker.py`, extend `from unittest.mock import MagicMock, patch` to include `call`. `threading` is already imported. Import `KVPoolWorker` inside `_worker` like the rest of this file. Append:

```python
class TestKVPoolWorkerResetStore(unittest.TestCase):
    def _worker(self, send=None, recv=None, reset_ok=True):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker.kv_send_thread = send
        worker.kv_recv_thread = recv
        worker.m_store = MagicMock()
        worker.m_store.reset.return_value = reset_ok
        worker._invalid_block_ids = {1, 2}
        worker._invalid_block_ids_lock = threading.Lock()
        return worker

    def test_joins_non_none_queues_then_resets(self):
        parent = MagicMock()
        send = parent.send
        recv = parent.recv
        store = parent.store
        store.reset.return_value = True
        worker = self._worker(send=send, recv=recv)
        worker.m_store = store
        self.assertTrue(worker.reset_store())
        self.assertEqual(
            parent.mock_calls,
            [
                call.send.request_queue.join(),
                call.recv.request_queue.join(),
                call.store.reset(),
            ],
        )
        send.join.assert_not_called()
        recv.join.assert_not_called()
        self.assertEqual(worker._invalid_block_ids, set())

    def test_none_threads_still_remove_all(self):
        worker = self._worker(send=None, recv=None)
        self.assertTrue(worker.reset_store())
        worker.m_store.reset.assert_called_once()
        self.assertEqual(worker._invalid_block_ids, set())

    def test_clears_invalid_ids_even_when_reset_fails(self):
        worker = self._worker(reset_ok=False)
        self.assertFalse(worker.reset_store())
        self.assertEqual(worker._invalid_block_ids, set())

    def test_clears_invalid_ids_when_join_raises(self):
        send = MagicMock()
        send.request_queue.join.side_effect = RuntimeError("join fail")
        worker = self._worker(send=send)
        self.assertFalse(worker.reset_store())
        worker.m_store.reset.assert_not_called()
        self.assertEqual(worker._invalid_block_ids, set())
```

In `test_ascend_store_connector.py`: add `import threading`; change `from unittest.mock import MagicMock, patch` to `MagicMock, call, patch`; extend the existing connector import with `LookupKeyServer`; then add:

```python
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import (
    LOOKUP_MSG,
    RESET_MSG,
    RESP_ERR,
    RESP_OK,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
```

```python
class TestLookupKeyServerReset(unittest.TestCase):
    def _server(self, reset_ok=True):
        server = LookupKeyServer.__new__(LookupKeyServer)
        server.decoder = MagicMock()
        server.pool_worker = MagicMock()
        server.pool_worker.reset_store.return_value = reset_ok
        server.pool_worker.lookup_scheduler.return_value = 8
        server.decoder.decode.side_effect = [[0], ["aa"]]
        return server

    def test_reset_ok(self):
        server = self._server(True)
        self.assertEqual(server._handle_frames([RESET_MSG]), RESP_OK)

    def test_reset_err(self):
        server = self._server(False)
        self.assertEqual(server._handle_frames([RESET_MSG]), RESP_ERR)

    def test_reset_exception_is_err(self):
        server = self._server()
        server.pool_worker.reset_store.side_effect = RuntimeError("die")
        self.assertEqual(server._handle_frames([RESET_MSG]), RESP_ERR)

    def test_lookup_still_returns_u32(self):
        server = self._server()
        frames = [LOOKUP_MSG, (4).to_bytes(4, "big"), b"g", (0).to_bytes(4, "big"), b"h"]
        self.assertEqual(server._handle_frames(frames), (8).to_bytes(4, "big"))

    def test_unknown_msg_is_err(self):
        server = self._server()
        self.assertEqual(server._handle_frames([(4).to_bytes(4, "big")]), RESP_ERR)

    def test_reset_path_joins_then_backend_reset(self):
        parent = MagicMock()
        send = parent.send
        recv = parent.recv
        store = parent.store
        store.reset.return_value = True
        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker.kv_send_thread = send
        worker.kv_recv_thread = recv
        worker.m_store = store
        worker._invalid_block_ids = {1, 2}
        worker._invalid_block_ids_lock = threading.Lock()
        server = LookupKeyServer.__new__(LookupKeyServer)
        server.decoder = MagicMock()
        server.pool_worker = worker
        self.assertEqual(server._handle_frames([RESET_MSG]), RESP_OK)
        self.assertEqual(
            parent.mock_calls,
            [
                call.send.request_queue.join(),
                call.recv.request_queue.join(),
                call.store.reset(),
            ],
        )
        send.join.assert_not_called()
        recv.join.assert_not_called()
        self.assertEqual(worker._invalid_block_ids, set())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerResetStore tests/ut/distributed/ascend_store/test_ascend_store_connector.py::TestLookupKeyServerReset -v`

Expected: FAIL (missing `reset_store` / `_handle_frames`).

- [ ] **Step 3: Write minimal implementation**

`pool_worker.py` on `KVPoolWorker` (near `wait_for_save`):

```python
    def reset_store(self) -> bool:
        ok = False
        try:
            if self.kv_send_thread is not None:
                self.kv_send_thread.request_queue.join()
            if self.kv_recv_thread is not None:
                self.kv_recv_thread.request_queue.join()
            if self.m_store is None:
                ok = True
            else:
                ok = self.m_store.reset()
        except Exception:
            logger.exception("KVPoolWorker.reset_store failed")
            ok = False
        finally:
            with self._invalid_block_ids_lock:
                self._invalid_block_ids.clear()
        return ok
```

Never call `Thread.join()` on the transfer threads. Never `bool(self.m_store.reset())`. `MooncakeBackend.reset()` already returns `True`/`False`; pass that through. `load_async=False` means `kv_recv_thread is None` → skip recv join.

`ascend_store_connector.py` imports — extend the existing `pool_scheduler` import (do not import `pool_scheduler` from a second place):

```python
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import (
    LOOKUP_MSG,
    RESET_MSG,
    RESP_ERR,
    RESP_OK,
    KVPoolScheduler,
    get_zmq_rpc_path_lookup,
)
```

Inside `LookupKeyServer.__init__`, replace the nested `process_request` loop. Keep `recv_multipart` **outside** the send try/except (`close()` makes recv raise; wrapping recv in a continue-loop would spin). `_handle_frames` already swallows handler errors. Frame indices for LOOKUP shift by +1 versus the old untagged protocol:

```python
        def process_request():
            while self.running:
                all_frames = self.socket.recv_multipart(copy=False)
                try:
                    self.socket.send(self._handle_frames(all_frames))
                except Exception:
                    logger.exception("LookupKeyServer REP send failed")

        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()
```

Add this as a **method of `LookupKeyServer`** (same indent as `close`, not nested in `__init__`):

```python
    def _handle_frames(self, all_frames) -> bytes:
        try:
            msg_type = bytes(all_frames[0])
            if msg_type == RESET_MSG:
                ok = self.pool_worker.reset_store()
                return RESP_OK if ok is True else RESP_ERR
            if msg_type != LOOKUP_MSG:
                logger.warning("LookupKeyServer unknown msg_type=%r", msg_type)
                return RESP_ERR
            token_len = int.from_bytes(all_frames[1], byteorder="big")
            kv_group_ids = self.decoder.decode([all_frames[2]])
            hbm_hit_tokens = int.from_bytes(all_frames[3], byteorder="big")
            hashes_str = self.decoder.decode(all_frames[4:])
            result = self.pool_worker.lookup_scheduler(
                token_len,
                hashes_str,
                kv_group_ids,
                use_layerwise=False,
                hbm_hit_tokens=hbm_hit_tokens,
            )
            return result.to_bytes(4, "big")
        except Exception:
            logger.exception("LookupKeyServer._handle_frames failed")
            return RESP_ERR
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPoolWorkerResetStore tests/ut/distributed/ascend_store/test_ascend_store_connector.py::TestLookupKeyServerReset tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestLookupKeyClient -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add \
  vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py \
  vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py \
  tests/ut/distributed/ascend_store/test_pool_worker.py \
  tests/ut/distributed/ascend_store/test_ascend_store_connector.py
GIT_AUTHOR_NAME='zhichao' GIT_AUTHOR_EMAIL='linzhichao2@huawei.com' \
git commit -m "$(cat <<'EOF'
feat: drain rank-0 queues then RESET Mooncake

EOF
)"
```

---

### Task 5: Scheduler reset_store + connector reset_cache

**Files:**
- Modify: `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`
- Modify: `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py`
- Test: `tests/ut/distributed/ascend_store/test_pool_scheduler.py`
- Test: `tests/ut/distributed/ascend_store/test_ascend_store_connector.py`

Worker vs scheduler: `getattr(self, "connector_scheduler", None) is None`. Worker `__init__` never assigns `connector_scheduler`; tests must cover the missing-attribute case, not only an explicit `None`.

- [ ] **Step 1: Write the failing tests**

In `test_pool_scheduler.py`:

```python
class TestKVPoolSchedulerResetStore(unittest.TestCase):
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_creates_client_when_none(self, mock_client_cls):
        mock_client_cls.return_value.reset.return_value = True
        sched = KVPoolScheduler(make_config(), use_layerwise=False)
        sched.client = None
        self.assertTrue(sched.reset_store())
        mock_client_cls.assert_called_once_with(sched.vllm_config)
        self.assertIs(sched.client, mock_client_cls.return_value)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_false_when_client_reset_false(self, mock_client_cls):
        mock_client_cls.return_value.reset.return_value = False
        sched = KVPoolScheduler(make_config(), use_layerwise=False)
        sched.client = None
        self.assertFalse(sched.reset_store())

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_false_when_client_reset_raises(self, mock_client_cls):
        mock_client_cls.return_value.reset.side_effect = RuntimeError("zmq down")
        sched = KVPoolScheduler(make_config(), use_layerwise=False)
        sched.client = None
        self.assertFalse(sched.reset_store())

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient")
    def test_false_when_client_ctor_raises(self, mock_client_cls):
        mock_client_cls.side_effect = RuntimeError("no socket")
        sched = KVPoolScheduler(make_config(), use_layerwise=False)
        sched.client = None
        self.assertFalse(sched.reset_store())
```

In `test_ascend_store_connector.py`:

```python
class TestAscendStoreConnectorResetCache(unittest.TestCase):
    def _scheduler_connector(self, *, layerwise=False):
        c = AscendStoreConnector.__new__(AscendStoreConnector)
        c.use_layerwise = layerwise
        c.connector_scheduler = MagicMock()
        c.connector_scheduler.load_specs = {"r1": object()}
        c.connector_scheduler.reset_store.return_value = True
        c.connector_scheduler.store_scheduler = MagicMock()
        c._kv_cache_events = object()
        return c

    def test_scheduler_success_clears_load_specs_returns_bool(self):
        c = self._scheduler_connector()
        self.assertTrue(c.reset_cache())
        self.assertEqual(c.connector_scheduler.load_specs, {})
        self.assertIsNone(c._kv_cache_events)
        c.connector_scheduler.reset_store.assert_called_once()

    def test_scheduler_rpc_false_still_cleared_load_specs(self):
        c = self._scheduler_connector()
        c.connector_scheduler.reset_store.return_value = False
        self.assertFalse(c.reset_cache())
        self.assertEqual(c.connector_scheduler.load_specs, {})

    def test_layerwise_returns_false_no_rpc_no_remove_all(self):
        c = self._scheduler_connector(layerwise=True)
        self.assertFalse(c.reset_cache())
        c.connector_scheduler.reset_store.assert_not_called()
        c.connector_scheduler.store_scheduler.remove_all.assert_not_called()
        self.assertEqual(len(c.connector_scheduler.load_specs), 1)

    def test_worker_missing_scheduler_attr_returns_none(self):
        c = AscendStoreConnector.__new__(AscendStoreConnector)
        c.use_layerwise = False
        c.connector_worker = MagicMock()
        self.assertFalse(hasattr(c, "connector_scheduler"))
        self.assertIsNone(c.reset_cache())

    def test_worker_scheduler_none_returns_none(self):
        c = AscendStoreConnector.__new__(AscendStoreConnector)
        c.use_layerwise = False
        c.connector_scheduler = None
        c.connector_worker = MagicMock()
        self.assertIsNone(c.reset_cache())
```

Layerwise returns `False` before `load_specs.clear()`, so the dict stays non-empty. Do not call `reset_store` or `store_scheduler.remove_all`. Worker tests must cover the missing-attribute case (real `__init__` never assigns `connector_scheduler`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestKVPoolSchedulerResetStore tests/ut/distributed/ascend_store/test_ascend_store_connector.py::TestAscendStoreConnectorResetCache -v`

Expected: FAIL (missing methods).

- [ ] **Step 3: Write minimal implementation**

`KVPoolScheduler.reset_store`:

```python
    def reset_store(self) -> bool:
        try:
            if self.client is None:
                self.client = LookupKeyClient(self.vllm_config)
            return self.client.reset()
        except Exception:
            logger.exception("KVPoolScheduler.reset_store failed")
            return False
```

Do not `load_specs.clear()` here.

`AscendStoreConnector.reset_cache`:

```python
    def reset_cache(self) -> bool | None:
        if getattr(self, "connector_scheduler", None) is None:
            return None
        if self.use_layerwise:
            logger.error("layerwise reset_cache is not implemented")
            return False
        self.connector_scheduler.load_specs.clear()
        self._kv_cache_events = None
        return self.connector_scheduler.reset_store()
```

Scheduler path must not return `None`. Do not call `store_scheduler.remove_all` from the connector. Do not `try/except` around `reset_store()` on the connector — `KVPoolScheduler.reset_store` already swallows RPC exceptions so `reset_cache()` does not raise.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestKVPoolSchedulerResetStore tests/ut/distributed/ascend_store/test_ascend_store_connector.py::TestAscendStoreConnectorResetCache -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add \
  vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py \
  vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py \
  tests/ut/distributed/ascend_store/test_pool_scheduler.py \
  tests/ut/distributed/ascend_store/test_ascend_store_connector.py
GIT_AUTHOR_NAME='zhichao' GIT_AUTHOR_EMAIL='linzhichao2@huawei.com' \
git commit -m "$(cat <<'EOF'
feat: implement AscendStoreConnector.reset_cache

EOF
)"
```

---

### Task 6: Full unit-test regression

**Files:** none new.

- [ ] **Step 1: Run the whole ascend_store UT folder**

Run: `python -m pytest tests/ut/distributed/ascend_store/ -v`

Expected: PASS. Existing lookup tests must still pass with tagged LOOKUP frames.

- [ ] **Step 2: Commit only if Step 1 forced extra fixes**

If green and no extra diffs: skip. If you had to patch an overlooked lookup caller, commit that patch with the same author rules.

---

## Self-review (spec coverage)

| Spec requirement | Task |
| --- | --- |
| Mooncake `remove_all(force=True)`, ignore return codes, TypeError fallback | 1 |
| memcache/yuanrong `False` + ERROR `backend=memcache` / `backend=yuanrong` | 2 |
| `@abstractmethod` only after all three backends implement `reset` | 1–2 |
| Constants only in `pool_scheduler.py` | 3 |
| `bytes(resp) == RESP_OK` | 3 |
| Tagged LOOKUP payload unchanged except frame-0 tag | 3–4 |
| Unknown msg / old token_len frame → `RESP_ERR` | 4 |
| `request_queue.join()` not `Thread.join`; join before `m_store.reset()` | 4 |
| Join recv if thread is not None (`load_async=True`); skip if None | 4 |
| Always clear `_invalid_block_ids` (`finally`, including join raise) | 4 |
| LookupKeyServer RESET → unique entry `reset_store()`; `ok is True` → `RESP_OK` | 4 |
| REP `while self.running`; handler/`send` catch; do not swallow `recv` (avoid spin on `close()`) | 4 |
| Lazy `self.client`; RPC exception / ctor exception → `False`, not raised | 5 |
| Layerwise scheduler `False`, no RPC, no `store_scheduler.remove_all` | 5 |
| Worker `reset_cache()` `None` via `getattr(..., None) is None` (missing attr) | 5 |
| `load_specs.clear()` before RPC; still empty when RPC returns False | 5 |
| No verl API, no `store_generation`, no LOOKUP load-plan | out of scope (YAGNI) |
| TP>1 collective drain | explicitly not in this plan (spec chose no) |
