"""A simple module that automates the creation of multiprocessing pipelines."""

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Process, Event, Queue
from queue import Empty, Full
from time import time
from typing import Callable, Dict, Tuple, Iterable, List, Union, Optional
from random import choice

import numpy as np

from autopipes.multiproc_utils import QueueProtocol

QueueMappingType = Dict[str, QueueProtocol]
QueueFactoryType = Callable[[], QueueProtocol]
EventMappingType = Dict[str, Event]
InitFnType = Callable[[QueueMappingType, QueueMappingType, EventMappingType], None]


@dataclass
class _Node:
    name: str
    factory: Callable
    inlet_names: Tuple[str, ...] = tuple()
    outlet_names: Tuple[str, ...] = tuple()
    outlet_factories: Tuple[Optional[QueueFactoryType], ...] = tuple()
    destructor: Callable = lambda: None


@dataclass
class _Pipe:
    inlet_node: _Node = None
    outlet_nodes: List[_Node] = field(default_factory=list)
    base_queue: QueueProtocol = None
    out_queues: QueueMappingType = None
    queue_factory: QueueFactoryType = None


def _make_util_process(target, *args):
    process = Process(target=target, args=args, daemon=True)
    process.start()
    return process


def _sink_thread(queue: QueueProtocol):
    while True:
        queue.get()


def _splitter_thread(base_queue: QueueProtocol, split_queues: Iterable[QueueProtocol]):
    while True:
        item = base_queue.get()

        split_targets = list(split_queues)

        while len(split_targets) > 0:
            random_queue = choice(split_targets)
            try:
                random_queue.put(item, timeout=0)
                split_targets.remove(random_queue)
            except (TimeoutError, Full):
                pass


_make_sink = partial(_make_util_process, _sink_thread)


def _make_splitter(base_queue: QueueProtocol, split_queues: Iterable[QueueProtocol]):
    process = _make_util_process(_splitter_thread, base_queue, split_queues)
    return process, split_queues


def _bridge_thread(in_queue: QueueProtocol, out_queue: QueueProtocol):
    while True:
        item = in_queue.get()
        out_queue.put(item)


def _make_bridge(in_queues: QueueMappingType, out_queues: QueueMappingType, events: EventMappingType):
    assert len(in_queues) == 1 and len(out_queues) == 1
    return _make_util_process(
        _bridge_thread,
        next(iter(in_queues.values())),
        next(iter(out_queues.values()))
    )


@dataclass
class _FlowMeterConfig:
    queue_name: str
    window_size: int
    print_every: int


class _FlowTimeTracker:
    def __init__(self, window_size: int):
        self.index = 0
        self.last_time = time()
        self.durations = np.zeros(window_size)

    def measure(self):
        cur_time = time()
        self.durations[self.index % len(self.durations)] = cur_time - self.last_time
        self.last_time = cur_time
        self.index += 1

    @property
    def rate(self):
        try:
            return 1 / self.durations[:self.index].mean()
        except AttributeError:
            return 0


def _flow_meter_thread(in_queue: QueueProtocol, out_queue: Queue, config: _FlowMeterConfig):
    inflow = _FlowTimeTracker(config.window_size)
    outflow = _FlowTimeTracker(config.window_size)

    index = 0

    while True:
        try:
            item = in_queue.get(timeout=60)
            inflow.measure()
        except Empty:
            warnings.warn(RuntimeWarning(f"Queue {config.queue_name} seems to be stuck, as no item "
                                         f"was put into it for 60 seconds."))
            item = None

        try:
            if item is not None:
                out_queue.put(item, timeout=60)
                outflow.measure()
        except Empty:
            warnings.warn(RuntimeWarning(f"Queue {config.queue_name} seems to be stuck, as the "
                                         f"output queue was blocked for 60 seconds."))

        if index % config.print_every == 0:
            print(f"[{config.queue_name}] {inflow.rate:.2f} items/s in, "
                  f"{outflow.rate:.2f} items/s out, {out_queue.qsize()} items queued", flush=True)

        index += 1


class _FlowMeter:
    def __init__(self, queue_name, window_size, print_every):
        self.config = _FlowMeterConfig(queue_name, window_size, print_every)
        self.worker: Process = None

    def start(self, in_queues: QueueMappingType, out_queues: QueueMappingType, events: EventMappingType):
        self.worker = Process(
            target=_flow_meter_thread,
            args=(
                next(iter(in_queues.values())),
                next(iter(out_queues.values())),
                self.config
            ),
            daemon=True
        )
        self.worker.start()

    def stop(self) -> None:
        self.worker.terminate()


@dataclass
class _BridgeDescriptor:
    inlet_name: str
    outlet_name: str
    outlet_factory: QueueFactoryType


class OutletWithoutInletException(Exception):
    """Thrown if an outlet exists without any corresponding inlet."""
    pass


class Pipeline:
    """
    A Pipeline instance is a descriptor for a multiprocessing pipeline. The pipeline consists of
    nodes that receive inputs from other nodes via inlets and send outputs to other nodes via
    outlets (both optional).

    The pipeline is first described using the `pipeline.add_*` methods and can then be initialized
    with `pipeline.initialize`.
    """

    def __init__(
            self,
            default_queue_factory: QueueFactoryType = Queue,
            debug_flow_queues: Union[Iterable[str], str, None] = None
    ):
        """
        Creates a new pipeline descriptor using the given queue implementation. Each call to the
        ``queue_factory`` function will receive ``queue_factory_args`` as parameters.

        :param default_queue_factory: A factory function that produces queues to be used as a pipe between
            nodes. Possible choices are ``Queue`` or ``SimpleQueue`` from the ``multiprocessing``
            module, but any custom picklable objects are possible in a multiprocessing context.
        :param debug_flow_queues: An iterable of queue names to debug the flow of. Can also be
            'all' to debug all queues or ``None`` to disable debugging (default).
        """
        self.default_queue_factory = default_queue_factory
        self.debug_flow_queues = debug_flow_queues

        self.nodes: Dict[str, _Node] = dict()
        self.events: Dict[str, Event] = dict()
        self.initialized: bool = False
        self.queues: QueueMappingType = dict()
        self.optional_bridges: List[_BridgeDescriptor] = []
        self.helper_processes: List[Process] = []

        if self.debug_flow_queues is not None:
            warnings.warn(RuntimeWarning("Note that using debug_queue_flow can add significant "
                                         "overhead to the pipeline."))

    def add_node(
            self,
            name: str,
            init_fn: InitFnType,
            inlet_names: Iterable[str] = tuple(),
            outlet_names: Iterable[str] = tuple(),
            outlet_factories: Iterable[Optional[QueueFactoryType]] = None,
            destructor: Callable = lambda: None
    ) -> None:
        """
        Adds a new node to the pipeline description.

        :param name: The unique identifier of the node.
        :param init_fn: A function that is called when the node is initialized. It receives three
            parameters: A dictionary of input queues, a dictionary of output queues and a dictionary
            of events.
        :param inlet_names: A tuple of names of the inlets of this node.
        :param outlet_names: A tuple of names of the outlets of this node.
        :param outlet_factories: A tuple of factory function for queues, where
            ``outlet_factories[i]`` corresponds to the outlet with name ``outlet_names[i]``. If
            ``None``, the default factory from the ``Pipeline`` constructor will be used.
        :param destructor: An optional clean-up function that frees all resources used by the
            node once the pipeline is stopped.
        """
        assert not self.initialized
        assert name not in self.nodes

        if outlet_factories is None:
            outlet_factories = [None] * len(outlet_names)

        if self.debug_flow_queues is not None:
            # create a flow meter for outlets

            new_outlet_names = []

            for outlet_name, outlet_factory in zip(outlet_names, outlet_factories):
                if self.debug_flow_queues == 'all' or outlet_name in self.debug_flow_queues:
                    # TODO remove hardcoded values
                    meter = _FlowMeter(name, window_size=20, print_every=20)
                    new_outlet_name = outlet_name + "=>flow_meter"
                    debug_node = _Node(
                        name=outlet_name + "_flow_meter",
                        factory=meter.start,
                        inlet_names=(new_outlet_name,),
                        outlet_names=(outlet_name,),
                        outlet_factories=(partial(Queue, maxsize=32),),
                        destructor=meter.stop
                    )
                    self.nodes[debug_node.name] = debug_node
                    new_outlet_names.append(new_outlet_name)
                else:
                    new_outlet_names.append(outlet_name)

            # rename outlets accordingly
            outlet_names = new_outlet_names

        node = _Node(
            name=name,
            factory=init_fn,
            inlet_names=tuple(inlet_names),
            outlet_names=tuple(outlet_names),
            outlet_factories=tuple(outlet_factories),
            destructor=destructor
        )
        self.nodes[node.name] = node

    def _as_helper_process(self, process_factory):
        def wrapper(*args, **kwargs):
            process = process_factory(*args, **kwargs)
            self.helper_processes.append(process)

        return wrapper

    def add_bridge(
            self,
            inlet_name: str,
            outlet_name: str,
            outlet_factory: QueueFactoryType = None
    ) -> None:
        """
        Adds a dummy node that shortcuts an outlet with an with an inlet.

        :param inlet_name: Name of the inlet of the bridge.
        :param outlet_name: Name of the outlet of the bridge.
        """
        self.add_node(
            name=inlet_name + "=>" + outlet_name,
            init_fn=self._as_helper_process(_make_bridge),
            inlet_names=(inlet_name,),
            outlet_names=(outlet_name,),
            outlet_factories=(outlet_factory,),

        )

    def add_optional_bridge(
            self,
            inlet_name: str,
            outlet_name: str,
            outlet_factory: QueueFactoryType = None
    ) -> None:
        """
        Adds a bridge using `add_bridge`, but only if no node exists that has `outlet_name` in
        its list of outlet names at initialization time.

        :param inlet_name: Name of the inlet of the bridge.
        :param outlet_name: Name of the outlet of the bridge.
        """
        self.optional_bridges.append(_BridgeDescriptor(inlet_name, outlet_name, outlet_factory))

    def add_event(self, name: str) -> None:
        """
        Adds a `multiprocessing.Event` to the pipeline that is shared between all nodes.

        :param name: Unique name of the event.
        """
        assert name not in self.events
        self.events[name] = Event()

    def _make_queue(self, name, factory) -> QueueProtocol:
        assert name not in self.queues
        queue = self.default_queue_factory() if factory is None else factory()
        self.queues[name] = queue
        return queue

    def _get_pipes_dict(self) -> Dict[str, _Pipe]:
        pipes = defaultdict(_Pipe)
        for node in self.nodes.values():
            for inlet_name in node.inlet_names:
                pipes[inlet_name].outlet_nodes.append(node)
            for outlet_name, outlet_factory in zip(node.outlet_names, node.outlet_factories):
                assert pipes[outlet_name].inlet_node is None
                pipes[outlet_name].inlet_node = node
                pipes[outlet_name].queue_factory = outlet_factory
        return pipes

    def initialize(self, auto_create_sinks: bool = True) -> None:
        """
        Initializes the pipeline by going through all added nodes and computing the corresponding
        connections. Each connection is represented by a `multiprocessing.Queue` and handed over
        to the `init_fn` of the corresponding nodes.

        If multiple nodes take the same outlet as an inlet, each item sent to the output queue is
        duplicated for each input queue. If an outlet is not connect to any inlet, a sink thread
        is created that automatically removes items from the queue. This behavior can be disabled
        using the `auto_create_sinks` parameter.

        :param auto_create_sinks: If `True`, outlets without any inlets will automatically be
        redirected to a sink thread that removes and discards items in the corresponding queue.
        If `False`, an error is thrown if such an outlet is detected.
        """
        assert not self.initialized

        # check if optional bridges are required

        for bridge in self.optional_bridges:
            # check if any node outputs the output of the bridge
            if not any(bridge.outlet_name in node.outlet_names for node in self.nodes.values()):
                self.add_bridge(bridge.inlet_name, bridge.outlet_name, bridge.outlet_factory)

        self.initialized = True

        # find all pipes required

        pipes = self._get_pipes_dict()

        # construct base queues, splitters and sinks

        for name, pipe in pipes.items():
            pipe.base_queue = self._make_queue(name=name, factory=pipe.queue_factory)
            self.queues[name] = pipe.base_queue

            if len(pipe.outlet_nodes) == 0:
                if auto_create_sinks:
                    # If there are no outlets, add sink
                    warnings.warn(
                        RuntimeWarning(f"Warning: Pipe '{name}' has no outlets. Adding sink node. "
                                       f"Pipe object: {pipe}.")
                    )
                    proc = _make_sink(pipe.base_queue)
                    self.helper_processes.append(proc)
                    pipe.out_queues = dict()
                else:
                    raise OutletWithoutInletException(f"Pipe '{name}' has no outlets. Pipe object: {pipe} ")

            elif len(pipe.outlet_nodes) > 1:
                queue_names = [f"{name}_{out_node.name}_split" for out_node in pipe.outlet_nodes]
                proc, out_queues = _make_splitter(
                    pipe.base_queue,
                    [self._make_queue(name=qname, factory=pipe.queue_factory) for qname in queue_names]
                )
                pipe.out_queues = {n.name: q for n, q in zip(pipe.outlet_nodes, out_queues)}
                self.helper_processes.append(proc)
            else:
                pipe.out_queues = {pipe.outlet_nodes[0].name: pipe.base_queue}

        # initialize nodes with queues
        for node in self.nodes.values():
            in_queues = {inlet_name: pipes[inlet_name].out_queues[node.name]
                         for inlet_name in node.inlet_names}
            out_queues = {outlet_name: pipes[outlet_name].base_queue
                          for outlet_name in node.outlet_names}

            if self.debug_flow_queues is not None:
                def _replace_flow_names(queue_dict):
                    return {qname.replace("=>flow_meter", ""): queue
                            for qname, queue in queue_dict.items()}

                in_queues = _replace_flow_names(in_queues)
                out_queues = _replace_flow_names(out_queues)

            node.factory(in_queues, out_queues, self.events)

    def close(self):
        """Calls ``close`` an all queue objects created by this pipeline."""
        assert self.initialized
        for queue in self.queues.values():
            queue.close()

        for helper in self.helper_processes:
            helper.terminate()

        for node in self.nodes.values():
            node.destructor()
