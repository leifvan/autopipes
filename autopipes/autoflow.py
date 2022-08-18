"""
A simple, low-overhead, UIMA-style pipeline that uses dictionaries as its basic data model. The
pipeline nodes are defined by transformation functions that require specific fields of the data to
be present and and add new fields to it.
"""

import queue
import sys
import traceback
from dataclasses import dataclass
from functools import partial
from multiprocessing import Queue, Process, Event
from typing import Generic, Collection, List, Container, Optional, Hashable, TypeVar, Callable, \
    Literal, Dict, Iterable, no_type_check

import networkx as nx

T = TypeVar("T", bound=Container)


class AutoflowDefinitionError(Exception):
    """Raised if an error due to an invalid definition of the flow."""


class AutoflowInitializationError(Exception):
    """Raised if an internal error occurs while initialization the flow."""


class AutoflowRuntimeError(RuntimeError):
    """Raised if an issue occured while the flow is running."""


@dataclass
class Transformation:
    """
    Dataclass representing an Autoflow transformation and its metadata needed during execution.
    Subclass this and override the ``apply`` method to define the transformation, or override
    ``thread`` or even ``_initialize`` if you need more flexibility.

    By default, the ``apply`` method will run in a separate Python process and be called with
    the output of the previous transformation (or possibly ``None`` if it is the first operation
    in the pipeline).
    """

    requires: Optional[Collection[Hashable]]
    adds: Optional[Collection[Hashable]]
    worker: Optional[Process] = None
    in_queue: Optional[Queue] = None
    out_queue: Optional[Queue] = None
    abort_event: Optional[Event] = None
    exception_queue: Optional[Queue] = None

    def _initialize(
            self,
            in_queue: Queue,
            out_queue: Queue,
            abort_event: Event,
            exception_queue: Queue,
            queue_timeout: float,
            debug_mode: bool
    ):
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.abort_event = abort_event
        self.exception_queue = exception_queue

        self.worker = Process(
            target=self.thread,
            args=(queue_timeout, debug_mode)
        )
        self.worker.start()

    def can_process(self, item: Container) -> bool:
        """
        Returns ``True`` if this transformation is applicable to ``item`` and ``False`` otherwise.
        """
        if self.requires is not None and any(r not in item for r in self.requires):
            return False
        if self.adds is not None and any(a in item for a in self.adds):
            return False
        return True

    def is_source(self) -> bool:
        """
        Returns ``True`` if this transformation is a source, i.e. adds annotations without
        requiring any. Otherwise, returns ``False``.
        """
        return self.requires is None

    def is_sink(self) -> bool:
        """
        Returns ``True`` if this transformation is a sink, i.e. requires annotations without adding
        any. Otherwise, returns ``False``.
        """
        return self.adds is None

    def thread(self, queue_timeout: float, debug_mode: bool):
        """
        This method will be run in a subprocess when ``_initialize`` is called. It is expected to
        process items from ``in_queue`` and put the results into the ``out_queue`` as long as the
        ``abort_event`` is not set. Additionally, it should add any unhandled exceptions into the
        ``exceptions_queue`` and set the ``abort_event`` if that happens.

        By default, it processes items by calling the ``apply`` method of this instance.
        """
        try:
            while not self.abort_event.is_set():

                # receive data

                data = None if self.in_queue is None else self.in_queue.get(timeout=queue_timeout)
                if debug_mode:
                    if data is not None:
                        expected_keys = [] if self.requires is None else self.requires
                        missing_keys = [r for r in expected_keys if r not in data]
                        if len(missing_keys) > 0:
                            raise AutoflowRuntimeError(f"Debug mode: Incoming data object is missing "
                                                       f"required keys. Expected {expected_keys}, but "
                                                       f"keys {missing_keys} are missing.")
                    try:
                        previous_keys = [k for k in data]
                    except TypeError:
                        previous_keys = []

                self.on_data_received(data)

                # transform data

                new_data = self.apply(data)
                if debug_mode and new_data is not None:
                    expected_keys = {
                        *([] if self.requires is None else self.requires),
                        *([] if self.adds is None else self.adds),
                        *previous_keys
                    }
                    missing_keys = {k for k in expected_keys if k not in new_data}
                    if len(missing_keys) > 0:
                        raise AutoflowRuntimeError(f"Debug mode: Transformed data object is "
                                                   f"missing keys. Expected {expected_keys}, but "
                                                   f"keys {missing_keys} are missing.")

                    try:
                        superfluous_keys = {k for k in new_data if k not in expected_keys}
                        if len(superfluous_keys) > 0:
                            raise AutoflowRuntimeError(f"Debug mode: Transformed data object has "
                                                       f"too many keys. Expected only "
                                                       f"{expected_keys}, but keys "
                                                       f"{superfluous_keys} are also present.")
                    except TypeError:
                        pass
                self.on_data_transformed(data)

                # queue data

                if self.out_queue is not None:
                    self.out_queue.put(new_data, timeout=queue_timeout)
                self.on_data_queued(data)

        except Exception as e:
            print(f"[Autoflow] {self} has caused an exception: {e}", file=sys.stderr, flush=True)
            traceback.print_tb(e.__traceback__, file=sys.stderr)
            self.abort_event.set()
            self.exception_queue.put(e, timeout=queue_timeout)

    def on_data_received(self, data: T) -> None:
        """
        Callback method that is called after the worker loop has received new data from the
        incoming queue.
        :param data: The incoming data.
        """
        pass

    def on_data_transformed(self, data: T):
        """
        Callback method that is called after the worker loop has transformed the data using the
        apply method.
        :param data: The transformed data.
        """
        pass

    def on_data_queued(self, data: T):
        """
        Callback method that is called after the worker loop queued the transformed data into the
        output queue.
        :param data: The transformed data.
        """
        pass

    def apply(self, data: T):
        """
        Processes a given ``data`` object and returns the result.
        """
        raise NotImplementedError

    def __repr__(self):
        return f"Transformation {self.requires} -> {self.adds}"


OptimizationMode = Literal[None, "earliest_sinks", "min_annotations"]


def _cleaned_topological_sorts(graph):
    for sort in nx.all_topological_sorts(graph):
        yield [i for i in sort if i != "source"]


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

    def _loss_earliest_sinks(self, sort: Iterable[int]) -> float:
        sink_indices = [i for i, t in enumerate(self.transformations) if t.is_sink()]
        num_sinks_left = len(sink_indices)
        weight = 0
        for s in sort:
            if s in sink_indices:
                num_sinks_left -= 1
            weight += num_sinks_left

        return weight

    def _loss_min_annotations(self, sort: Iterable[int]) -> float:
        weight = 0
        num_annotations = 0

        for s in sort:
            weight += num_annotations
            if not self.transformations[s].is_sink():
                num_annotations += len(self.transformations[s].adds)

        return weight

    _optimization_losses: Dict[OptimizationMode, Callable[[Iterable[int]], float]] = {
        "earliest_sinks": _loss_earliest_sinks,
        "min_annotations": _loss_min_annotations
    }

    def run(
            self,
            allow_unused_annotations: bool = False,
            optimize: OptimizationMode = None,
            queue_timeout: float = 60,
            debug_mode: bool = False
    ) -> None:
        """
        Runs the pipeline until the abort_event is set in one of the transformations.

        Before running, the method will check if the definition of the pipeline is correct in the
        sense that

         * at least one transformation is added,
         * the requirements of all annotations can be fulfilled,
         * no more than one transformation annotates the same value,
         * there are no cyclic dependencies,
         * all transformations will be used and
         * no annotations are unused (optional).

        By default, the order of execution of the transformations is any valid order given the
        constraints (required & adds parameters). This can be changed by choosing one of the
        following cost functions for the ``optimize`` parameter:

          * ``"earliest_sinks"``: Sink transformations will be placed as close to the beginning as
            possible.
          * ``"min_annotations"``: The number of annotations existing per transformation step is
            minimized.


        :param allow_unused_annotations: If ``True``, this method will not throw an exception if
            transformations are added whose annotations are not required by any other
            transformation.
        :param optimize: Can be ``earliest_sinks``, ``"min_annotations"`` or ``None``.
        :param queue_timeout: Timeout (in seconds) for the queues used in the flow.
        :param debug_mode: If ``True``, the transformations are supposed to check the validity
            of incoming and transformed data during runtime.
        """

        # check if there are any transformations
        if len(self.transformations) == 0:
            raise AutoflowDefinitionError("No transformations were added to the flow.")

        # create a map: annotation -> transformation adding it
        producers = dict()
        for idx, t in enumerate(self.transformations):
            if t.adds is not None:
                for add in t.adds:
                    if add in producers:
                        raise AutoflowDefinitionError("More than one transformation adds the same value.")
                    producers[add] = idx

        # create a graph that represents a partial ordering of the transformations
        graph = nx.DiGraph()
        try:
            for idx, t in enumerate(self.transformations):
                if t.requires is not None:
                    for req in t.requires:
                        graph.add_edge(producers[req], idx)
                else:
                    graph.add_edge("source", idx)
        except KeyError as e:
            raise AutoflowDefinitionError(f"No transformation exists that produces {e}.")

        # find a topological sort of the transformations based on the graph

        try:
            if optimize is None:
                top_sort_indices = list(nx.algorithms.dag.topological_sort(graph))[1:]
            else:
                compute_weight = partial(self._optimization_losses[optimize], self)
                top_sort_indices = min(_cleaned_topological_sorts(graph), key=compute_weight)
        except nx.exception.NetworkXUnfeasible:
            raise AutoflowDefinitionError("Cyclic dependency was detected.")

        if len(top_sort_indices) < len(self.transformations):
            raise AutoflowInitializationError("The topological ordering does not include all "
                                              "nodes. This might be due to missing connectivity.")

        # check if the output is valid in the sense that
        #  - every required annotation is present
        #  - there are no superfluous annotations (optional)
        annotated_usages: Dict[Hashable, int] = dict()

        for s in top_sort_indices:
            t = self.transformations[s]

            if t.requires is not None:
                for r in t.requires:
                    # this throws a KeyError if r is not annotated in previous transformations
                    annotated_usages[r] += 1

            if t.adds is not None:
                for a in t.adds:
                    annotated_usages[a] = 0

        if not allow_unused_annotations:
            for key, count in annotated_usages.items():
                if count == 0:
                    raise AutoflowDefinitionError(f"Flow annotates unused key '{key}'.")

        # sort transformations array according to topological order
        self.transformations = [self.transformations[i] for i in top_sort_indices]

        # create queues and events
        abort_event = Event()
        exception_queue = Queue()
        queues = [self.queue_factory() for _ in range(len(self.transformations) - 1)]

        for t, last_queue, next_queue in zip(self.transformations, [None, *queues], [*queues, None]):
            t._initialize(
                in_queue=last_queue,
                out_queue=next_queue,
                abort_event=abort_event,
                exception_queue=exception_queue,
                queue_timeout=queue_timeout,
                debug_mode=debug_mode
            )

        # run and wait for exceptions in the respective queue
        abort_exception = None

        while not abort_event.is_set():
            try:
                abort_exception = exception_queue.get(timeout=1)
                abort_event.set()
            except queue.Empty:
                pass

        # abort_event was set; briefly try to join the workers, kill them if it takes too long
        for t in self.transformations:
            t.worker.join(timeout=1)
            t.worker.kill()

        # throw the next exception, if any
        if abort_exception is None:
            try:
                while True:
                    raise exception_queue.get(timeout=0)
            except queue.Empty:
                pass
        else:
            raise abort_exception


@no_type_check
def main():
    flow = Autoflow()
    flow.add_transformation_fn(lambda: None, None, ["rgbd"])
    flow.add_transformation_fn(lambda: None, None, ["server_pose"])
    flow.add_transformation_fn(lambda: None, ["rgbd"], ["seg"])
    flow.add_transformation_fn(lambda: None, ["rgbd"], ["flow"])
    flow.add_transformation_fn(lambda: None, ["rgbd", "seg"], ["odom"])
    flow.add_transformation_fn(lambda: None, ["seg"], ["unused"])
    flow.add_transformation_fn(lambda: None, ["rgbd", "odom"], ["exp_flow"])
    flow.add_transformation_fn(lambda: None, ["rgbd", "flow", "exp_flow"], ["dyn"])
    flow.add_transformation_fn(lambda: None, ["dyn"], ["static_rgbd", "dynamic_rgbd"])
    flow.add_transformation_fn(lambda: None, ["static_rgbd", "server_pose"], None)
    flow.add_transformation_fn(lambda: None, ["dynamic_rgbd", "server_pose"], None)
    flow.add_transformation_fn(lambda: None, ["unused"], None)
    flow.run(optimize="min_annotations")


if __name__ == '__main__':
    main()
