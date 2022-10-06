"""Gives basic info on GPU memory."""

import nvidia_smi


def total(gpu_index: int = 0) -> int:
    """Returns total amount of memory in bytes on the GPU given by the index.

    Args:
        gpu_index (int, optional): Index of GPU to get memory info from. Defaults to 0.

    Returns:
        int: Total memory on the given GPU (in bytes).
    """
    nvidia_smi.nvmlInit()
    total_memory = \
        nvidia_smi.nvmlDeviceGetMemoryInfo(
            nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_index)
        ).total
    nvidia_smi.nvmlShutdown()
    return total_memory
