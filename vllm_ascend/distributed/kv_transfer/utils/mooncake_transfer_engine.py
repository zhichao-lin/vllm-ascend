import threading

from vllm.logger import logger


class GlobalTE:
    def __init__(self):
        self.transfer_engine = None
        self.is_register_buffer: bool = False
        self.registered_buffers: list[tuple[int, int]] = []
        self.transfer_engine_lock = threading.Lock()
        self.register_buffer_lock = threading.Lock()

    def get_transfer_engine(self, hostname: str, device_name: str | None):
        if self.transfer_engine is None:
            with self.transfer_engine_lock:
                # Double-Checked Locking
                if self.transfer_engine is None:
                    try:
                        from mooncake.engine import TransferEngine  # type: ignore
                    except ImportError as e:
                        raise ImportError(
                            "Please install mooncake by following the instructions at "
                            "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                            "to run vLLM with MooncakeConnector."
                        ) from e
                    self.transfer_engine = TransferEngine()
                    device_name = device_name if device_name is not None else ""
                    ret_value = self.transfer_engine.initialize(hostname, "P2PHANDSHAKE", "ascend", device_name)
                    if ret_value != 0:
                        raise RuntimeError(f"TransferEngine initialization failed with ret_value: {ret_value}")
        return self.transfer_engine

    def register_buffer(self, ptrs: list[int], sizes: list[int]):
        with self.register_buffer_lock:
            assert self.transfer_engine is not None, "Transfer engine must be initialized"
            if self.is_register_buffer:
                return
            for ptr, size in zip(ptrs, sizes):
                ret_value = self.transfer_engine.register_memory(ptr, size)
                if ret_value != 0:
                    raise RuntimeError("Mooncake memory registration failed.")
            self.registered_buffers = list(zip(ptrs, sizes))
            self.is_register_buffer = True

    def unregister_buffer(self):
        with self.register_buffer_lock:
            n_buffers = len(self.registered_buffers)
            if not self.registered_buffers:
                logger.info(
                    "[kv_sleep] unregister_buffer action=skip skip_reason=buffers_empty "
                    "is_register_buffer=%s n_buffers=0",
                    self.is_register_buffer,
                )
                return
            if not self.is_register_buffer:
                logger.warning(
                    "[kv_sleep] unregister_buffer action=force_unregister "
                    "skip_reason=flag_false_buffers_remain is_register_buffer=%s n_buffers=%s",
                    self.is_register_buffer,
                    n_buffers,
                )
            else:
                logger.info(
                    "[kv_sleep] unregister_buffer action=unregister is_register_buffer=%s n_buffers=%s",
                    self.is_register_buffer,
                    n_buffers,
                )
            assert self.transfer_engine is not None, "Transfer engine must be initialized"

            unregistered_buffers: list[tuple[int, int]] = []
            for ptr, size in self.registered_buffers:
                ret_value = self.transfer_engine.unregister_memory(ptr)
                logger.info("[kv_sleep] unregister_memory ptr=%#x size=%s ret=%s", ptr, size, ret_value)
                if ret_value != 0:
                    rollback_failures = []
                    for unregistered_ptr, unregistered_size in reversed(unregistered_buffers):
                        rollback_ret = self.transfer_engine.register_memory(unregistered_ptr, unregistered_size)
                        if rollback_ret != 0:
                            rollback_failures.append((unregistered_ptr, rollback_ret))
                    self.is_register_buffer = not rollback_failures
                    raise RuntimeError(
                        f"Mooncake memory unregistration failed for ptr={ptr:#x}, "
                        f"ret_value={ret_value}, rollback_failures={rollback_failures}"
                    )
                unregistered_buffers.append((ptr, size))

            self.is_register_buffer = False

    def reregister_buffer(self):
        with self.register_buffer_lock:
            n_buffers = len(self.registered_buffers)
            if self.is_register_buffer or not self.registered_buffers:
                skip_reason = "already_registered" if self.is_register_buffer else "buffers_empty"
                logger.info(
                    "[kv_sleep] reregister_buffer action=skip skip_reason=%s "
                    "is_register_buffer=%s n_buffers=%s",
                    skip_reason,
                    self.is_register_buffer,
                    n_buffers,
                )
                return
            logger.info(
                "[kv_sleep] reregister_buffer action=reregister is_register_buffer=%s n_buffers=%s",
                self.is_register_buffer,
                n_buffers,
            )
            assert self.transfer_engine is not None, "Transfer engine must be initialized"

            reregistered_buffers: list[tuple[int, int]] = []
            for ptr, size in self.registered_buffers:
                ret_value = self.transfer_engine.register_memory(ptr, size)
                logger.info("[kv_sleep] register_memory ptr=%#x size=%s ret=%s", ptr, size, ret_value)
                if ret_value != 0:
                    rollback_failures = []
                    for reregistered_ptr, _ in reversed(reregistered_buffers):
                        rollback_ret = self.transfer_engine.unregister_memory(reregistered_ptr)
                        if rollback_ret != 0:
                            rollback_failures.append((reregistered_ptr, rollback_ret))
                    self.is_register_buffer = bool(rollback_failures)
                    raise RuntimeError(
                        f"Mooncake memory re-registration failed for ptr={ptr:#x}, "
                        f"ret_value={ret_value}, rollback_failures={rollback_failures}"
                    )
                reregistered_buffers.append((ptr, size))

            self.is_register_buffer = True


global_te = GlobalTE()
