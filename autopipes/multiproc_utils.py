from multiprocessing import Event, Queue
from multiprocessing.shared_memory import SharedMemory
from queue import Empty
from typing import Any, Tuple, Generic, TypeVar, runtime_checkable

import numpy as np
from numpy.typing import DTypeLike

T = TypeVar("T")


@runtime_checkable
class QueueProtocol(Generic[T]):
    """
    A basic protocol for queue-like objects. Both ``Queue`` and ``SimpleQueue`` from Python's
    ``multiprocessing`` module adhere to these.
    """

    def get(self, timeout: float = None) -> T:
        """Retrieves the first element from the queue."""
        raise NotImplementedError

    def put(self, data: T, timeout: float = None) -> None:
        """Adds an element to the end of the queue."""
        raise NotImplementedError

    def empty(self) -> bool:
        """Returns ``True`` if the queue is empty."""
        raise NotImplementedError

    def close(self) -> None:
        """Releases any resources this object uses."""
        raise NotImplementedError


class SharedBytes:
    """
    An efficient multiprocessing communication object using ``multiprocessing.shared_memory`` as
    its underlying data exchange technique. It mimics the behavior of a one-item queue, i.e. data
    is set via ``put()`` and can be read exactly once via ``get()``.
    """

    def __init__(self, size: int, name=None, can_write=None):
        """
        Creates a new SharedBytes object using ``size`` bytes of memory. Parameters ``name`` and
        ``can_write`` should not be set manually - they are only used for pickling.

        :param size: The number of bytes to be reserved for this object.
        :param name: (Should not be set manually) The name of the underlying SharedMemory object.
        :param can_write: (Should not be set manually) The event that handles the read/write state
            of the underlying SharedMemory.
        """
        self.shared_memory: SharedMemory = SharedMemory(name=name, create=name is None, size=size)
        self.can_write_event: Event = Event() if can_write is None else can_write

        if can_write is None:
            self.can_write_event.set()

    def __reduce__(self):
        args = (
            self.shared_memory.size,
            self.shared_memory.name,
            self.can_write_event,
        )
        return self.__class__, args

    def _acquire_read(self):
        while self.can_write_event.is_set():
            pass

    def _release_read(self):
        self.can_write_event.set()

    def _acquire_write(self):
        while not self.can_write_event.is_set():
            pass

    def _release_write(self):
        self.can_write_event.clear()

    def _get(self):
        return self.shared_memory.buf.tobytes()

    def get(self) -> Any:
        """Waits until a value is made available via a call to ``put`` and returns that value."""
        self._acquire_read()
        data = self._get()
        self._release_read()
        return data

    def _put(self, data):
        self.shared_memory.buf[:] = data

    def put(self, data) -> None:
        """
        Waits until a value is retrieved by a call to ``get`` and then makes ``data`` available
        for the next call to ``get``.
        """
        self._acquire_write()
        self._put(data)
        self._release_write()

    def close(self):
        """
        Frees the resources used by this object. This method should be called in all processes
        that use it.
        """
        self.shared_memory.unlink()

    def empty(self):
        return self.can_write_event.is_set()


class SharedNumpyArray(SharedBytes):
    """A convenience class that uses ``SharedBytes`` as a buffer for NumPy arrays."""

    @classmethod
    def like(cls, array: np.ndarray):
        """Creates a new shared array with the same shape and dtype as the given array."""
        return cls(shape=array.shape, dtype=array.dtype)

    def __init__(self, shape: Tuple[int, ...], dtype: DTypeLike, name=None, can_write=None):
        """
        Creates a new shared array with the given ``shape`` and ``dtype``. Parameters ``name`` and
        ``can_write`` should not be set manually - they are only used for pickling.

        :param shape: Tuple of integers describing the array shape.
        :param dtype: dtype of the array.
        :param name: (Should not be set manually) The name of the underlying SharedMemory object.
        :param can_write: (Should not be set manually) The event that handles the read/write state
            of the underlying SharedMemory.
        """
        dummy = np.zeros(shape, dtype)
        super().__init__(dummy.nbytes, name, can_write)
        self._array = np.ndarray(shape=shape, dtype=dtype, buffer=self.shared_memory.buf)

    def __reduce__(self):
        args = (
            self._array.shape,
            self._array.dtype,
            self.shared_memory.name,
            self.can_write_event
        )
        return self.__class__, args

    def _get(self) -> np.ndarray:
        return self._array.copy()

    def _put(self, data: np.ndarray):
        self._array[:] = data


class RingBufferQueue(QueueProtocol):
    """
    A wrapper around ``multiprocessing.Queue`` that removes the oldest elements when ``put`` is
    called and more than ``buffer_size`` many elements are currently queued.
    """

    def __init__(self, buffer_size: int, queue: Queue = None):
        """
        Creates a new wrapper around ``queue`` that removes the oldest item when ``put`` is called,
        if the number of queued elements is equal to ``buffer_size``.

        :param buffer_size: The maximum number of elements that are allowed in the queue
            simultaneously.
        :param queue: The underlying ``multiprocessing.Queue`` object wrapped by this instance. If
            ``None`` (default), a new queue object will be created.
        """
        if buffer_size < 1:
            raise ValueError("Buffer size must be at least 1.")

        self.buffer_size = buffer_size
        self._queue = Queue() if queue is None else queue

    def __reduce__(self):
        return self.__class__, (self.buffer_size, self._queue)

    def get(self, timeout: float = None) -> T:
        """
        Returns the first item from the queue. If no element is available, the function blocks
        until an item is queued.
        """
        return self._queue.get(timeout=timeout)

    def put(self, data: T, timeout: float = None) -> None:
        """Puts ``data`` into the queue and removes the oldest element if the queue is full."""
        overflow = self._queue.qsize() - self.buffer_size + 1

        try:
            for _ in range(overflow):
                self._queue.get_nowait()
        except Empty:
            pass

        self._queue.put(data, timeout=timeout)

    def empty(self) -> bool:
        """Returns ``True`` if the queue is empty."""
        return self._queue.empty()

    def close(self) -> None:
        """Calls the ``close`` method of the underlying queue object."""
        self._queue.close()
