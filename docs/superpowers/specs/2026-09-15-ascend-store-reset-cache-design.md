# AscendStore sleep/wake 后 reset_cache 设计

日期：2026-09-15  
仓库：`vllm-ascend` 分支 `pd-kvpool-fix`  
配套文档：`verl` 仓库 `pd-kvpool-fix` 上的 `docs/superpowers/specs/2026-09-15-kv-pool-sleep-wake-reset-design.md`

## 问题

Colocated RL 的 sleep/wake 会通过 `CaMemAllocator` 重映射 NPU HBM。Mooncake master 里仍保留指向 remap 前物理页的 key。下一步 Prefill 的 `Get` 命中元数据后 DMA 失败，表现为 `HcclBatchGet` / `-800 TRANSFER_FAIL`。

verl 在权重同步之后的 `clear_kv_cache`，以及 hybrid / 显式 `wake_up` 里，已经调用 `reset_prefix_cache(reset_connector=True)`。1P3D colocate_async **不**走 `replica.wake_up()`。对 Ascend store 这条路径目前是空操作：

- `AscendStoreConnector` 没有实现 `reset_cache()`。
- 基类返回 `None`。
- `MultiConnector.reset_cache()` 把 `None` 当成成功（`is not False`）。

上游 vLLM 的 `MooncakeStoreConnector` 已经是目标行为：scheduler 清空 `load_specs`，worker rank 0 对发送队列做 `request_queue.join()`，然后 `store.remove_all(force=True)`；**不解释 C++ 返回码**，只把异常当成失败。

本变更只覆盖 **`backend=mooncake` 且 `use_layerwise=False`**。这是 1P3D 训练脚本的实际配置。`load_async` 默认 `False`（训练脚本如此），但 RESET 路径必须同时正确覆盖 `load_async=True`。

## 目标

verl 的 `clear_kv_cache`，以及 hybrid `wake_up` 里的 `reset_prefix_cache(reset_connector=True)` 之后，Ascend store 必须：

1. 删掉共享 Mooncake master 上的全部 key。
2. 丢掉 scheduler 侧 `load_specs`，避免下一请求按旧 hash 去 `Get`。
3. 向 `MultiConnector` 返回真正的 `True`/`False`（scheduler 角色禁止返回 `None`）。

上一步训练写入的 KV 按设计丢弃。权重已经变了，即使 DMA 成功也不能复用。

`load_specs` 已空 **不是** 成功判据。成功只认 worker rank 0 对 `RESET_MSG` 的 `RESP_OK`（其含义是：本 rank 上非 `None` 的收发队列已 `request_queue.join()`，且 `MooncakeBackend.reset()` 没有抛异常）。RPC 失败时 master 上可能还有旧 key；verl 洞 2 必须失败即停并重试，禁止 `resume_generation`。

## 非目标

- 跨 sleep/wake 或跨权重更新保留 KV。
- 改 vllm-ascend worker sleep 路径里已有的 TransferEngine `unregister_buffer` / `reregister_buffer`。
- 改 `kv_load_failure_policy`，或关掉 `enable_sleep_mode`。
- 为 memcache / yuanrong 实现 `remove_all`。这两个 backend 的 `reset()` 返回 `False`，并打 ERROR：当前不支持 sleep/wake reset。`MultiConnector` 会把这次 `reset_cache()` 判失败。
- `use_layerwise=True`。该模式下不启动 `LookupKeyServer`。scheduler 的 `AscendStoreConnector.reset_cache()` 打 ERROR 并返回 `False`。禁止走 scheduler 侧 store handle 静默 `remove_all`（不 drain worker Put 队列）。
- 增加 `store_generation` 代数。v1 的保证是：先 `request_queue.join()`，再 `remove_all`。
- 不跟随上游 LOOKUP 响应从 4 字节 hit count 扩成 load plan。LOOKUP payload 与响应保持 Ascend 现状。

## 不变量

scheduler 角色的 `reset_cache()` 返回 `True`，当且仅当：

1. `use_layerwise=False` 且 `backend=mooncake`；
2. worker rank 0 已对 `kv_send_thread` / `kv_recv_thread`（非 `None`）执行 `request_queue.join()`；
3. `MooncakeBackend.reset()` 未抛异常（与上游一致：完全不看返回码，正整数、负整数、`None` 只要没抛都算成功）；
4. LookupKey 通道 `bytes(resp) == RESP_OK`。

`load_specs.clear()` 发生在 ZMQ RPC **之前**（与上游相同），用来丢掉 scheduler 侧指向即将被 wipe 的 key 的引用。清完之后若 RPC 失败，`load_specs` 已空但 master 可能仍有旧 key：下一请求会重新 lookup → 命中元数据 → `Get` 打到 remap 前的 PA。因此 `load_specs` 空不能当成成功。

reset 期间该 engine 上不能有新的 lookup / `Get` / `Put`。HTTP 层 drain 全部 P/D 由 verl 洞 1 负责（现码只 drain Prefill）。verl 的 HTTP drain **不等于** 各 TP rank 的 Put/Get 队列 drain。本 spec 仍要求在 `remove_all` 前对本 lookup server 所在进程（rank 0）做 `request_queue.join()`。`reset_cache()` 返回 `False` 时，由 verl 洞 2 失败即停，禁止 `resume_generation`。

`remove_all` 幂等。若同一 engine 在一次 step 里被调用两次（hybrid `wake_up` 再加 `clear_kv_cache`），两次都走完整路径，不要当成 bug。verl 对四个 engine 并行 `clear_kv_cache` 可以接受，本变更不增加“全局只 reset 一次”的协调器。

本仓库假定 verl 对 **每个 P/D engine 的 scheduler** 都调用 `reset_prefix_cache(reset_connector=True)`。现码 `ServerAdapter` 已按 PD 角色绑定各 engine TP0 的 HTTP server，IPC 后对该 handle `clear_kv_cache`；`vLLMPDReplica._server_handle` 虽是 Prefill，但 Adapter 不用那个 handle。本仓库不补全局协调器，也不改 verl。

配套 spec 只修 verl 现码仍在的 **两个洞**（本仓库不实现）：

1. **drain 只等 Prefill。** `vLLMPDReplica.sleep` / `release_kv_cache` 改为对 `self.servers` 里每个 Prefill **和** Decode 先 `wait_for_requests_to_drain`，再 sleep / release。非 PD 的 `vLLMReplica.sleep` 不动。
2. **忽略 `reset_prefix_cache` 的 `False`。** `vLLMHttpServer` 用 `_reset_prefix_and_connector`：`False` 则本 server abort（`reset_prefix_cache=False`）再试一次；第二次 `True` 则本 server `resume_generation()` 再返回（解开 abort 留下的 pause）；第一次就 `True` 不 resume；再 `False` 抛 `RuntimeError` 且不 resume。colocate 靠 `ray.get(rollout.update_weights)` 失败即停；hybrid `wake_up` 对每个 HTTP server `abort_all_requests.remote(reset_prefix_cache=False)`。不另加 replica 级 wipe。不要走 replica 无参 `abort_all_requests`（默认 `True`）。

1P3D colocate_async **不会**在 `on_step_end` 调 `replica.wake_up()`。wipe 发生在 IPC 之后的 `clear_kv_cache`。`vLLMHttpServer.wake_up` 里的 reset 只属于 hybrid / 显式 wake，见 §5。

## 设计

### 1. Backend `reset()`

涉及文件：

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/base.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/yuanrong_backend.py`

`Backend.reset() -> bool`：

- **MooncakeBackend**（与上游一致）：
  1. 若 `store` 尚未初始化，返回 `True`（没有可清的内容）。
  2. 优先 `store.remove_all(force=True)`。若抛 `TypeError`（当前 binding 没有 `force`），再调 `store.remove_all()`。这是对本地 Mooncake binding 的适配，上游没有这条回退。
  3. **完全不解释返回码**。现网 `remove_all` 可能返回删除的 key 数，也可能返回负错误码且不抛异常。与上游相同：调用未抛异常则返回 `True`（含返回正整数、负整数、`None`）。禁止 `ret == 0` 或 `ret < 0` 分支。返回值可以打 DEBUG/INFO，但不得参与 True/False。
  4. 成功打 INFO。任何 `Exception` → 打 ERROR，返回 `False`。
- **MemcacheBackend / YuanrongBackend**：打 ERROR「sleep/wake reset_cache 当前不支持 backend=memcache/yuanrong」，返回 `False`。不要 no-op 成功。

不要 close 或重新 `setup()` store。TransferEngine 注册仍由 `vllm_ascend/worker/worker.py` 的 sleep/wake 负责。

### 2. LookupKey 管理协议：带 tag 的 RESET

当前 `LookupKeyServer` 把 frame 0 当作 `token_len`（u32），没有管理命令。lookup 的 client 和 server 都在本仓库，一起改。

协议常量 **只定义在** `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`（与 `LookupKeyClient` 同文件）。`ascend_store_connector.py` 里的 `LookupKeyServer` 从该模块 import（它已经 import `KVPoolScheduler` / `get_zmq_rpc_path_lookup`，不会循环导入）。禁止第二份拷贝，禁止 `pool_scheduler` 再 import `ascend_store_connector`。

```text
LOOKUP_MSG = b"lookup"
RESET_MSG  = b"reset"
RESP_OK    = b"\x01"
RESP_ERR   = b"\x00"
```

请求格式：`[msg_type] [payload...]`。

- `LOOKUP_MSG`：后续帧保持现有 lookup payload（`token_len`、kv groups、`hbm_hit_tokens`、hashes）。响应：4 字节大端 hit count（不变）。不跟随上游的 load-plan 响应。
- `RESET_MSG`：无 payload。响应：`RESP_OK` 或 `RESP_ERR`。
- 其它 `msg_type`（包括旧 client 仍把 4 字节 `token_len` 放在 frame 0）：打 WARNING，回复 `RESP_ERR`。不要当 lookup 解。

`b"lookup"` 是 6 字节帧，旧 `token_len` 是 4 字节帧，ZMQ 按整帧比较，不会撞车。本变更之后不再支持旧的无 tag client。可以接受：只有本仓库的 `LookupKeyClient` 连这个 socket。

### 3. Worker：先 `request_queue.join()` 再 `remove_all`

涉及文件：

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py`（`LookupKeyServer`）
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`

收到 `RESET_MSG` 时：

1. 若 `kv_send_thread is not None`：调用 `kv_send_thread.request_queue.join()`。
2. 若 `kv_recv_thread is not None`：调用 `kv_recv_thread.request_queue.join()`。这是相对上游的增强（上游只 join 发送队列）。
3. **禁止** `kv_send_thread.join()` / `kv_recv_thread.join()`（`Thread.join`）。这些线程是常驻循环，`Thread.join` 会永远阻塞。现码 `KVPoolWorker.wait_for_save` 用的就是 `request_queue.join()`。
4. 调用 `pool_worker.reset_store()` → `m_store.reset()`。
5. Mooncake 路径要清的 worker 本地状态 **只有** `_invalid_block_ids`（在 `_invalid_block_ids_lock` 下 `clear()`）。**无论 `reset_store()` 成败都清**：wake 之后旧 PA 已经无意义，失败时留下这批 id 会把 remap 前的 block 带进下一步。不清 lease、`_allocated_gvas`、发送线程 `stored_requests`：前两项属于 memcache/layerwise，第三项由 abort / 请求结束路径维护。
6. `reset_store()` 为 True 则发 `RESP_OK`，否则 `RESP_ERR`。`reset_store()` 内部异常也回 `RESP_ERR`，LookupKeyServer 的 handler 必须 catch，不能让 REP 线程死掉。步骤 5 在发 ACK/NACK 之前执行。

`KVPoolWorker.reset_store() -> bool` 是 worker 侧唯一入口。每个 engine 目前只有 rank 0 且非 layerwise 时跑 `LookupKeyServer`（`parallel_config.rank == 0`）；由该进程执行 `remove_all`。同一 engine 的其他 TP rank 不调用 `remove_all`（与上游相同）。TP>1 的影响见 §6。

**`load_async` 与 drain recv：**

- `load_async=True`：会创建 `kv_recv_thread`，Get 进这条队列。RESET 对其 `request_queue.join()` **是正确且必要的**：等到 rank 0 上已入队的异步 Get 结束后再 `remove_all`，避免 wipe 窗口里 DMA 打到 remap 后的页。Get 不会把旧 PA 写回 master。其它 TP rank 的 recv 队列 v1 仍然不 join（与 §6 相同限制）。join 期间不能再往队列里加新 Get，这一点靠 verl abort。
- `load_async=False`（1P3D 训练脚本默认）：**没有** `kv_recv_thread`，步骤 2 跳过。同步 Get 跑在 worker 前向线程上，LookupKeyServer 的 `request_queue.join()` join 不到。此时 drain recv 增强是空操作。安全依赖 verl abort，使 `reset_prefix_cache` 时没有 in-flight Get。

1P3D 有四个 engine（1 Prefill + 3 Decode），各自有 rank-0 lookup server，共用一个 Mooncake master。每个 engine 的 `reset_prefix_cache(reset_connector=True)` 都会对本 engine 清 `load_specs`，并对共享 master 调一次 `remove_all`（幂等）。前提是 **没有任何 engine 还在 Put**。verl 必须：

1. reset 之前 abort + HTTP drain **全部** P/D（verl 洞 1：现码只 drain Prefill，配套 spec 改成 `self.servers` 全 drain）。
2. 对 **每个** P/D HTTP server 调用 `clear_kv_cache` / `reset_prefix_cache`。现码已由各 engine TP0 的 `ServerAdapter` 做到；verl 洞 2 保证这次 reset 的 `False` 失败即停。

Decode 且 `consumer_is_to_put=False` 时通常没有 send 线程，步骤 1 跳过，仍要 `remove_all`。

### 4. Scheduler：`reset_store()` 与 `reset_cache()`

涉及文件：

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py`

现码里 LookupKey 客户端是懒创建的 `KVPoolScheduler.client`（`LookupKeyClient`），不是 `lookup_client`。`reset_store()` 必须按 lookup 同一方式创建：若 `self.client is None`，则 `self.client = LookupKeyClient(self.vllm_config)`，再 `self.client.reset()`。第一次 reset 发生在任何 lookup 之前时也要能工作。

`LookupKeyClient.reset() -> bool`：

- 在现有 REQ socket 上 `send(RESET_MSG)`，`recv()` 后 **仅当** `bytes(resp) == RESP_OK` 返回 `True`。禁止用 `!= RESP_ERR`（空帧、lookup 的 4 字节 hit count、任意垃圾都会被当成成功）。
- **不要**新增一条 RPC 通道。
- **不要**把 ZMQ `linger` 当成超时。`linger` 只影响 `socket.close()`。现有 `LookupKeyClient` 的 `recv()` **没有超时**。v1 保持这个行为，不新加超时。
- 现有 client 是同步 `send/recv`，没有上游那套 `ThreadPoolExecutor`。reset 直接在调用线程上发即可，不要照搬 executor。

`KVPoolScheduler.reset_store() -> bool`：

1. 不做 `load_specs.clear()`（清的位置在 connector，与上游一致）。
2. 确保 `self.client` 已创建，调用 `self.client.reset()`。
3. `send` / `recv` 抛出的任何异常 → catch，打 ERROR，返回 `False`。禁止把异常抛出 `reset_cache()`。
4. 否则返回 `client.reset()` 的 bool。

`AscendStoreConnector.reset_cache() -> bool | None`：

- Scheduler + `use_layerwise=True`：打 ERROR「layerwise reset_cache 未实现」，返回 `False`。不要调用 `reset_store()`，不要对 `store_scheduler` 做 `remove_all`。
- Scheduler + `use_layerwise=False`：
  1. `self.connector_scheduler.load_specs.clear()`
  2. `self._kv_cache_events = None`
  3. 返回 `connector_scheduler.reset_store()`。禁止返回 `None`。
- Worker 角色：返回 `None`（由 ZMQ RESET 驱动 worker，与上游一致）。`MultiConnector` 把 worker 的 `None` 当成成功；真正算数的是 scheduler 的结果，因为 `reset_prefix_cache` 跑在 scheduler 上。

v1 不清 scheduler 的 `_request_trackers` / `_loading_req_ids` / `_unfinished_requests`（上游 `reset_cache` 也不清同类状态）。这些在 verl abort + HTTP drain 之后应已空。

### 5. 调用点（不对 verl 新增 API）

vLLM scheduler 已从 `reset_prefix_cache(reset_connector=True)` 调用 `connector.reset_cache()`。本变更之后，这次调用在 `backend=mooncake` 且非 layerwise 时会清掉 Mooncake。不新增 engine 公有方法。

`vllm_ascend/worker/worker.py` 的 sleep/wake 保持：

```text
sleep:  global_te.unregister_buffer(); allocator.sleep(...)
wake:   allocator.wake_up(tags); global_te.reregister_buffer()  # 当 tags 含 kv_cache
```

unregister/reregister 修 DMA 注册，`remove_all` 修 master 元数据。危险顺序是：reregister 之后、wipe 之前发生对旧 key 的 `Get`。只要 abort 后到 wipe 前没有新流量，`remove_all` 在 reregister 之前或之后都可以；必须在 `resume_generation` 之前完成。

**1P3D colocate_async 主路径**（`PPOTrainerColocateAsync`，本仓库不改 verl）：

```text
on_sample_end:
  abort_replicas
  sleep_replicas                         → verl 洞 1：drain 全部 P/D，再 engine.sleep
                                           （不在这里 reset connector）

on_step_end:
  checkpoint_manager.update_weights:
    abort_replicas
    release_kv_cache                     → COLOCATED 时 HTTP server 内 no-op；
                                           verl 洞 1：replica 先 drain 全部 P/D
    各 rank ServerAdapter.update_weights:
      IPC
      若 _has_server：对本 engine HTTP server clear_kv_cache
                      → verl 洞 2：_reset_prefix_and_connector
                         （现码已覆盖全部 P/D；这一次是 wipe）
    resume_kv_cache                      → COLOCATED 时 no-op
    resume_generation_replicas           # manager 内部第 8 步
  resume_generation_replicas             # trainer on_step_end 再调一次（冗余幂等）
```

这条路径 **不调用** `replica.wake_up()` / `vLLMHttpServer.wake_up`。不要把 wake_up 画进 colocated 主链。IPC 过程中 worker 可能自行 restore HBM；无论 reregister 发生在 IPC 中还是之后，wipe 都在 `clear_kv_cache`，且在 `resume_generation` 之前。

现码 `ServerAdapter` 按 `_pd_role` 绑定 Prefill / Decode HTTP server，不是 replica 的 Prefill `server_handle`。配套 spec 不改这条绑定。verl 只修洞 1（全 P/D drain）和洞 2（`False` 失败即停）。`clear_kv_cache` 抛错发生在 manager 的 `ray.get(rollout.update_weights)`，走不到两次 resume。

**hybrid / 显式 `vLLMHttpServer.wake_up`：**

```text
engine.wake_up(tags=["kv_cache","weights"])  → reregister
verl 洞 2：_reset_prefix_and_connector       # wake 路径上的 wipe；False 失败即停
verl 洞 2 hybrid 附加：任一 server 抛错 →
  每个 HTTP server abort_all_requests.remote(reset_prefix_cache=False)
```

若同一 step 里随后又 `clear_kv_cache`，第二次 wipe 幂等，不是 bug。

### 6. TP>1：只 drain rank 0 还是 collective drain（取舍说明，v1 与上游一致选否）

1P3D 训练脚本默认 `GEN_TP=4`，Decode 也是 TP=4。`LookupKeyServer` 只在 `parallel_config.rank == 0`。RESET 只能 `request_queue.join()` **这一个进程**的发送/接收队列。其它 TP rank 各有自己的 `kv_send_thread` / `kv_recv_thread`。

谁会 Put：`tp_rank % put_step == 0` 的 rank 才会 save。`put_step` 在 `num_kv_head < tp_size` 时为 `tp_size // num_kv_head`，否则为 1。

- MLA（`num_kv_head=1`，TP=4）→ `put_step=4` → **只有 rank 0 Put**。只 drain rank 0 的发送队列，与「所有会 Put 的 rank」重合。
- GQA 且 `num_kv_head >= tp_size` → `put_step=1` → **每个 TP rank 都 Put**。只 drain rank 0 会漏掉 rank 1–3 的 in-flight Put。

HTTP drain / abort 不等于这些队列已空。abort 掉的请求可能没走 `wait_for_save`（那条路径才会对本 rank `request_queue.join()`）。worker restore HBM 之后，所有 TP 的传输线程接着跑残留队列；scheduler 随后才发 `RESET_MSG`。时间窗：rank 1–3 的 Put 可能在 rank 0 的 `remove_all` **之后**落到 master，旧 PA 的 key 被写回。

**否（v1，与上游一致，本 spec 采用）：**

- 只 join rank 0 的 `request_queue`，然后 `remove_all`。
- 实现简单，不新增 TP collective / 额外 ZMQ。
- 风险：`put_step=1` 时，abort 残留 Put 可在 wipe 之后写回。依赖配套 spec 的 abort + HTTP drain，使进入 wipe 时各 rank 队列实际上已空。正常跑完的请求会经 `wait_for_save` join 过本 rank 队列，这部分是安全的。
- MLA / 仅 rank 0 Put 的模型上，这个风险对 **Put** 不成立；`load_async=True` 时 **Get** 仍可能在其它 rank 的 recv 线程里（drain recv 也只覆盖 rank 0）。

**是（本 spec 不采用，供对照）：**

- rank 0 收到 `RESET_MSG` 后，先对会 Put 的 TP rank（至少 `tp_rank % put_step == 0`）做一次 collective / worker RPC：各 rank `request_queue.join()`，再回到 rank 0 `remove_all`。
- 能关掉「wipe 之后被其它 rank 写回」的窗口，和 1P3D `GEN_TP=4` 对齐更稳。
- 成本：要在 worker 之间加一条现有代码没有的同步；sleep/wake 后 collective 是否仍可用要单独验证；实现量和故障面都比 v1 大。
- 即便做了 collective，四个 **engine** 之间仍靠 verl 保证没有跨 engine 的 Put，本仓库仍然不做全局协调器。

v1 明确选择 **否**。若后续 1P3D 在 `put_step=1` 下仍出现 wipe 之后的 `HcclBatchPut` / 旧 hash `Get`，再升级为 collective，不在本次范围。

## 测试

在 `tests/ut/distributed/ascend_store/` 新增或扩展：

1. scheduler `reset_cache()`：`bytes(resp) == RESP_OK` → `True`；`RESP_ERR` 或非 `RESP_OK` 的字节 → `False`；RPC 抛异常 → `False` 且不往外抛；之后 `load_specs` 为空。NACK/异常用例要断言 `load_specs` 仍被清空（先清再 RPC），返回值仍是 `False`。
2. worker 角色 `reset_cache()` 返回 `None`。
3. `use_layerwise=True` 时 scheduler `reset_cache()` 返回 `False`，且不发 ZMQ、不对 `store_scheduler` 调 `remove_all`。
4. Lookup server：`RESET_MSG` 对 send/recv（非 `None`）调用的是 `request_queue.join()` 且顺序在 `reset_store` / `remove_all` 之前；`kv_send_thread is None` 时仍 `remove_all`；`kv_recv_thread is None`（`load_async=False`）时跳过 recv join 仍 `remove_all`；`LOOKUP_MSG` 仍返回 4 字节 hit count；未知 `msg_type` 回 `RESP_ERR`；`reset_store` 失败后仍清 `_invalid_block_ids`。
5. `MooncakeBackend.reset`：mock store，断言 `remove_all(force=True)`；`TypeError` 回退 `remove_all()`；未抛异常即 `True`（返回正整数、负整数、`None` 都是 `True`）；抛异常 → `False`；`store is None` → `True`。
6. memcache / yuanrong 的 `reset()` 返回 `False`。
7. `KVPoolScheduler.reset_store()`：`self.client is None` 时会创建 `LookupKeyClient`；`LookupKeyClient.reset()` 发送 `RESET_MSG`；只有 `bytes(resp) == RESP_OK` 为 True。

本 spec 不要求 NPU / HCCL 测试。

## 落地顺序

先在 `pd-kvpool-fix` 落地本 spec，再按配套 spec 修 verl 两个洞（全 P/D drain + `_reset_prefix_and_connector`）。直到本仓库的 `reset_cache()` 对 mooncake 返回真实 `True`/`False`，verl 的 `reset_connector=True` 清不了 pool。

## 验收（本仓库）

- `backend=mooncake` 且 `use_layerwise=False` 时，scheduler 上 `AscendStoreConnector.reset_cache()` 只返回 `True`/`False`，不返回 `None`。
- `True` 意味着 rank 0 已对非 `None` 的收发队列 `request_queue.join()`，`remove_all` 未抛异常，且 `bytes(resp) == RESP_OK`。
- `use_layerwise=True` 或 `backend=memcache/yuanrong` 时返回 `False`。
- 上述单测通过。
