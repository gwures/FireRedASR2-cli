import logging
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("fireredasr2s.auto_batch")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import nvidia_smi
    nvidia_smi.nvmlInit()
    NVML_AVAILABLE = True
    logger.info("nvidia_smi initialized successfully")
except (ImportError, Exception) as e:
    NVML_AVAILABLE = False
    logger.debug(f"nvidia_smi not available: {e}")


@dataclass
class BatchConfig:
    asr_batch_size: int = 8


class AutoBatchManager:
    DEFAULT_LOW = 0.80
    DEFAULT_HIGH = 0.90

    def __init__(self, target_low: Optional[float] = None, target_high: Optional[float] = None):
        self.config = BatchConfig()
        self._oom_count = 0
        self._max_oom_count = 3
        self._adjustment_cooldown = 0

        self._target_low = target_low if target_low is not None else self.DEFAULT_LOW
        self._target_high = target_high if target_high is not None else self.DEFAULT_HIGH

        self._oom_recovery_tasks = 0
        self._utilization_history = []
        self._window_size = 5

    def get_target_range(self) -> Tuple[float, float]:
        return self._target_low, self._target_high

    def get_gpu_memory_info(self) -> Tuple[int, int, int]:
        """获取 GPU 显存信息 (Total, Free, Allocated) in Bytes"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return 0, 0, 0

        try:
            device_id = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device_id)

            if NVML_AVAILABLE:
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_id)
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                total = info.total
                free = info.free
            else:
                total = torch.cuda.get_device_properties(device_id).total_memory
                reserved = torch.cuda.memory_reserved(device_id)
                free = total - reserved

            return total, free, allocated
        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {e}")
            return 0, 0, 0

    def get_gpu_utilization(self) -> float:
        total, _, allocated = self.get_gpu_memory_info()
        if total == 0:
            return 0.0
        return allocated / total

    def get_smoothed_utilization(self) -> float:
        current = self.get_gpu_utilization()
        self._utilization_history.append(current)
        if len(self._utilization_history) > self._window_size:
            self._utilization_history.pop(0)
        return sum(self._utilization_history) / len(self._utilization_history)

    def get_initial_config(self, initial_batch: int, use_fp16: bool = False) -> BatchConfig:
        config = BatchConfig(asr_batch_size=initial_batch)
        logger.info(f"Auto-batch starting from: {initial_batch}, target range: [{self._target_low:.2f}, {self._target_high:.2f}]")
        self.config = config
        return config

    def should_adjust(self) -> bool:
        if self._adjustment_cooldown > 0:
            self._adjustment_cooldown -= 1
            return False
        return True

    def _calculate_adjustment_step(self, current: int, is_increasing: bool, utilization: float, target_low: float, target_high: float) -> int:
        if is_increasing:
            gap = target_high - utilization
            if gap > 0.15:
                base_step = current // 3
            elif gap > 0.08:
                base_step = current // 4
            else:
                base_step = current // 6
        else:
            gap = utilization - target_high
            if gap > 0.15:
                base_step = current // 2
            elif gap > 0.08:
                base_step = current // 3
            else:
                base_step = current // 4

        if current <= 8:
            return 1
        elif current <= 32:
            return min(4, base_step)
        elif current <= 64:
            return min(8, base_step)
        else:
            return min(12, base_step)

    def adjust_batch_sizes(self, current_asr: int) -> Tuple[int, str]:
        if self._oom_recovery_tasks > 0:
            self._oom_recovery_tasks -= 1
            return current_asr, ""

        if not self.should_adjust():
            return current_asr, ""

        utilization = self.get_smoothed_utilization()
        low, high = self.get_target_range()

        new_asr = current_asr
        reason = ""

        if utilization < low:
            step = self._calculate_adjustment_step(current_asr, True, utilization, low, high)
            new_asr = new_asr + step
            reason = f"显存利用率 {utilization:.1%} < {low:.0%}，增大批次 +{step}"
        elif utilization > high:
            if new_asr > 1:
                step = self._calculate_adjustment_step(current_asr, False, utilization, low, high)
                new_asr = max(new_asr - step, 1)
                reason = f"显存利用率 {utilization:.1%} > {high:.0%}，减小批次 -{step}"
            else:
                reason = f"显存利用率 {utilization:.1%} > {high:.0%}，已达下限"
        else:
            reason = f"显存利用率 {utilization:.1%} 在目标区间 [{low:.0%}, {high:.0%}] 内"

        if new_asr != current_asr:
            self._adjustment_cooldown = 2 if new_asr < current_asr else 4
            logger.info(f"Auto-adjust: ASR {current_asr}→{new_asr} ({reason})")

        return new_asr, reason

    def handle_oom(self, current_asr: int) -> Tuple[int, bool]:
        """仅在真实捕获到 OOM 异常时调用"""
        self._oom_count += 1
        logger.warning(f"OOM detected ({self._oom_count}/{self._max_oom_count})")

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self._oom_count == 1:
            reduction_factor = 0.5
        elif self._oom_count == 2:
            reduction_factor = 0.35
        else:
            reduction_factor = 0.25

        new_asr = max(1, int(current_asr * reduction_factor))

        self._oom_recovery_tasks = 20
        self._adjustment_cooldown = 15

        logger.warning(f"Reduced batch size due to OOM: ASR {current_asr}→{new_asr} (factor={reduction_factor})")

        should_abort = self._oom_count >= self._max_oom_count and new_asr <= 1

        return new_asr, should_abort

    def __del__(self):
        if NVML_AVAILABLE:
            try:
                nvidia_smi.nvmlShutdown()
            except:
                pass
