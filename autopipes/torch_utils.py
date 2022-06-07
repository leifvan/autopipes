from typing import Protocol, runtime_checkable


@runtime_checkable
class CUDAMovable(Protocol):
    """
    A protocol for objects that provide ``.cpu()`` and ``.cuda()`` methods to move PyTorch tensors
    associated with it to the CPU / GPU.
    """

    def cpu(self) -> 'CUDAMovable':
        """Moves all PyTorch tensors associated with this object to the CPU."""
        raise NotImplementedError

    def cuda(self) -> 'CUDAMovable':
        """Moves all PyTorch tensors associated with this object to the GPU."""
        raise NotImplementedError
