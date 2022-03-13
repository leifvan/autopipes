from typing import Protocol


class CUDAMoveable(Protocol):
    """
    A protocol for objects that provide ``.cpu()`` and ``.cuda()`` methods to move PyTorch tensors
    associated with it to the CPU / GPU.
    """

    def cpu(self) -> 'CUDAMoveable':
        """Moves all PyTorch tensors associated with this object to the CPU."""
        raise NotImplementedError

    def cuda(self) -> 'CUDAMoveable':
        """Moves all PyTorch tensors associated with this object to the GPU."""
        raise NotImplementedError
