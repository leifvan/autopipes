"""
A simple, low-overhead, UIMA-style pipeline that uses dictionaries as its basic data model. The
pipeline nodes are defined by transformation functions that require specific fields of the data to
be present and and add new fields to it.
"""

import queue
import sys
from dataclasses import dataclass
from multiprocessing import Queue, Process, Event
from typing import Generic, Collection, List, Container, Optional, Hashable, TypeVar, Callable

T = TypeVar("T", bound=Container)


class NoTransformationFoundException(Exception):
    """Raised if no transformation could be found for a data object."""


class NotASinkException(Exception):
    """Raised if a transformation does not return a value, even though it is not a sink."""


class SinkException(Exception):
    """Raised if a sink node returns a value."""


class NoSourceException(Exception):
    """Raised if the pipeline contains no source node."""


@dataclass
class Transformation:
    requires: Optional[Collection[Hashable]]
    adds: Optional[Collection[Hashable]]
    worker: Process = None
    in_queue: Queue = None
    out_queue: Queue = None
    abort_event: Event = None
    exception_queue: Queue = None

    def initialize(
            self,
            in_queue: Queue,
            out_queue: Queue,
            abort_event: Event,
            exception_queue: Queue
    ):
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.abort_event = abort_event
        self.exception_queue = exception_queue

        self.worker = Process(
            target=self.thread,
            args=tuple(),
            daemon=True
        )
        self.worker.start()

    def can_process(self, item: Container) -> bool:
        if self.requires is not None and any(r not in item for r in self.requires):
            return False
        if self.adds is not None and any(a in item for a in self.adds):
            return False
        return True

    def is_source(self) -> bool:
        return self.requires is None

    def is_sink(self) -> bool:
        return self.adds is None

    def thread(self):
        try:
            while not self.abort_event.is_set():
                data = self.in_queue.get()
                new_data = self.apply(data)
                if new_data is None and not self.is_sink():
                    raise NotASinkException(f"{self} did not return a value, even though it is "
                                            f"not a sink.")
                elif new_data is not None and self.is_sink():
                    raise SinkException(f"{self} is a sink, but returned a value.")
                elif new_data is not None:
                    self.out_queue.put(new_data)

        except Exception as e:
            print(f"[Autoflow] {self} has caused an exception: {e}", file=sys.stderr, flush=True)
            self.abort_event.set()
            self.exception_queue.put(e)

    def apply(self, data: T):
        raise NotImplementedError

    def __repr__(self):
        return f"Transformation {self.requires} -> {self.adds}"


class Autoflow(Generic[T]):
    """
    The main class of this module. Defines and runs a pipeline that consists of transformation
    functions that take data (optionally) with existing annotations and (optionally) adds new
    annotations to the data. These functions can either be added as subclasses of the
    ``Transformation`` base class or directly as a callable using ``add_transformation_fn``.

    You can create a source node by setting the required annotation to `None`. The function will
    then be periodically called with `None` as an input. Likewise, a sink node is created by
    setting the added annotations field to None

    Note: Each function will be running in its own Python process. The usual considerations for
    Python's multiprocessing library therefore apply here.
    """

    def __init__(self, queue_factory: Callable = lambda: Queue(maxsize=1)):
        self.transformations: List[Transformation] = []
        self.queue_factory = queue_factory

    def add_transformation(self, transformation: Transformation) -> None:
        """
        Adds a new ``Transformation`` object to the pipeline.
        :param transformation: The ``Transformation`` object to be added.
        """
        self.transformations.append(transformation)

    def add_transformation_fn(
            self,
            transformation_fn: Callable[[T], T],
            requires: Optional[Collection[Hashable]],
            adds: Optional[Collection[Hashable]]
    ) -> None:
        """
        Adds a transformation function in form of a callable to the pipeline.

        :param transformation_fn: A callable that accepts a data object and returns the modified
            data object. To work in multiprocessing environments, this function has to be in
            the global namespace to be accessible in subprocesses.
        :param requires: Keys that a data object requires to be transformed by this function.
        :param adds: Keys that this function adds to the data object.
        """
        t = Transformation(requires, adds)
        t.apply = transformation_fn
        self.add_transformation(t)

    def run(self) -> None:
        """
        Runs the pipeline until the abort_event is set.
        """
        abort_event = Event()
        dispatch_queue = self.queue_factory()
        exception_queue = self.queue_factory()

        for transformation in self.transformations:
            transformation.initialize(self.queue_factory(), dispatch_queue, abort_event, exception_queue)

        source_transformations = [t for t in self.transformations if t.is_source()]

        if len(source_transformations) == 0:
            raise NoSourceException

        while not abort_event.is_set():
            # try to send empty wrappers to sources
            for transformation in source_transformations:
                try:
                    transformation.in_queue.put_nowait(None)
                except queue.Full:
                    pass

            # get next item to be dispatched
            item = dispatch_queue.get()

            # find a suitable transformation
            try:
                transformation = next(t for t in self.transformations if t.can_process(item))
                transformation.in_queue.put(item)
            except StopIteration:
                raise NoTransformationFoundException(f"No transformation exists for {item}")

        for t in self.transformations:
            t.worker.join(timeout=1)
            t.worker.kill()

        try:
            while True:
                raise exception_queue.get_nowait()
        except queue.Empty:
            pass
