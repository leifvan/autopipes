import unittest
from multiprocessing import Semaphore
from time import time, sleep
from typing import List, Optional

import queue

from .autoflow import Transformation, Autoflow, AutoflowDefinitionError, AutoflowRuntimeError
from .autoflow_test_utils import RaiseExceptionTransformation, MeasureSpeedTransformation, TargetSpeedReachedException, \
    CountItemsTransformation, SuccessfulComparisonException


class TestException(Exception):
    pass


class FaultyTestTransformation(Transformation):
    def __init__(
            self,
            requires: Optional[List[str]],
            should_add: Optional[List[str]],
            actually_adds: Optional[List[str]],
            erases: Optional[List[str]] = None,
            disable_debug_mode: bool = False
    ):
        super(FaultyTestTransformation, self).__init__(requires=requires, adds=should_add)
        self.actually_adds = actually_adds
        self.erases = erases
        self.disable_debug_mode = disable_debug_mode

    def apply(self, data):
        if data is None:
            data = dict()

        if self.erases is not None:
            for field in self.erases:
                del data[field]

        if self.actually_adds is not None:
            for field in self.actually_adds:
                data[field] = 0

        return data

    def thread(self, queue_timeout: float, debug_mode: bool):
        if self.disable_debug_mode:
            debug_mode = False
        super(FaultyTestTransformation, self).thread(queue_timeout, debug_mode)


class TestTransformation(FaultyTestTransformation):
    def __init__(
            self,
            requires: Optional[List[str]],
            adds: Optional[List[str]],
    ):
        super(TestTransformation, self).__init__(requires=requires, should_add=adds, actually_adds=adds)


class CountingCompleteException(Exception):
    pass


class CountingTestTransformation(TestTransformation):
    def __init__(
            self,
            requires: Optional[List[str]],
            adds: Optional[List[str]],
            semaphore: Semaphore
    ):
        super(CountingTestTransformation, self).__init__(requires, adds)
        self.semaphore = semaphore
        self.counted = False

    def apply(self, data):
        if not self.counted:
            self.semaphore.acquire()
            self.counted = True

        if self.semaphore.acquire(block=False):
            self.semaphore.release()
        else:
            raise CountingCompleteException("Semaphore can't be acquired (which is a good thing).")

        return super(CountingTestTransformation, self).apply(data)


def sleep_transformation(data):
    sleep(20)
    return data


class AutoflowTest(unittest.TestCase):
    def test_debug_mode_accepts_not_required_keys_being_present(self):
        flow = Autoflow()
        flow.add_transformation(TestTransformation(requires=None, adds=["a"]))
        flow.add_transformation(TestTransformation(requires=None, adds=["b"]))
        flow.add_transformation(CountItemsTransformation(requires=["a", "b"], num_items=10))

        with self.assertRaises(SuccessfulComparisonException):
            flow.run(debug_mode=True)

    def test_debug_mode_recognizes_missing_keys_before_transformation(self):
        flow = Autoflow()
        flow.add_transformation(TestTransformation(requires=None, adds=["a"]))
        flow.add_transformation(FaultyTestTransformation(
            requires=["a"],
            should_add=["b"],
            actually_adds=None,
            disable_debug_mode=True
        ))
        flow.add_transformation(TestTransformation(requires=["b"], adds=None))

        with self.assertRaises(AutoflowRuntimeError):
            flow.run(allow_unused_annotations=True, debug_mode=True)

    def test_debug_mode_recognizes_missing_keys_after_transformation(self):
        flow = Autoflow()
        flow.add_transformation(TestTransformation(requires=None, adds=["a"]))
        flow.add_transformation(FaultyTestTransformation(requires=["a"], should_add=["b"], actually_adds=["c"]))

        with self.assertRaises(AutoflowRuntimeError):
            flow.run(allow_unused_annotations=True, debug_mode=True)

    def test_debug_mode_recognizes_additional_keys_after_transformation(self):
        flow = Autoflow()
        flow.add_transformation(TestTransformation(requires=None, adds=["a"]))
        flow.add_transformation(FaultyTestTransformation(requires=["a"], should_add=["b"], actually_adds=["b", "c"]))

        with self.assertRaises(AutoflowRuntimeError):
            flow.run(allow_unused_annotations=True, debug_mode=True)

    def test_global_timeout_works(self):
        flow = Autoflow()
        flow.add_transformation(TestTransformation(None, ["a"]))
        flow.add_transformation_fn(sleep_transformation, requires=["a"], adds=None)
        with self.assertRaises((queue.Empty, queue.Full)):
            flow.run(queue_timeout=10)

    def test_workers_terminate(self):
        flow = Autoflow()
        flow.add_transformation(TestTransformation(None, ["a"]))
        flow.add_transformation(RaiseExceptionTransformation(["a"], None, TestException))
        with self.assertRaises(TestException):
            flow.run()

        sleep(2)  # give worker threads a short amount of time to finish
        self.assertTrue(all(not t.worker.is_alive() for t in flow.transformations))

    def test_missing_transformation(self):
        """
        Flow is missing a transformation to reach the sink node. It should detect this.
        """
        flow = Autoflow()
        flow.add_transformation(TestTransformation(None, ["a"]))
        flow.add_transformation(TestTransformation(["b"], None))
        with self.assertRaises(AutoflowDefinitionError):
            flow.run()

    def test_cyclic_dependency(self):
        """
        Flow has a cyclic dependency and cannot be computed. That should throw an exception.
        """
        flow = Autoflow()
        # flow.add_transformation(TestTransformation(None, ["a"]))
        flow.add_transformation(TestTransformation(["a"], ["b"]))
        flow.add_transformation(TestTransformation(["b"], ["a"]))
        with self.assertRaises(AutoflowDefinitionError):
            flow.run()

    def test_multiple_producers_of_same_key(self):
        """
        It is not allowed to have multiple transformations annotating the same key. This should be
        detected on startup.
        """
        flow = Autoflow()
        flow.add_transformation(TestTransformation(None, ["a"]))
        flow.add_transformation(TestTransformation(["a"], ["b"]))
        flow.add_transformation(TestTransformation(None, ["b"]))
        with self.assertRaises(AutoflowDefinitionError):
            flow.run()

    def test_missing_source(self):
        """
        Flow should recognize that no source node is present.
        """
        flow = Autoflow()
        flow.add_transformation(TestTransformation(["a"], ["b"]))
        with self.assertRaises(AutoflowDefinitionError):
            flow.run()

    def test_exception_caught(self):
        """
        Test if exceptions thrown inside the transformation functions are caught and reraised.
        """
        flow = Autoflow()
        flow.add_transformation(TestTransformation(None, ["a"]))
        flow.add_transformation(RaiseExceptionTransformation(["a"], None, TestException))
        with self.assertRaises(TestException):
            flow.run()

    def test_earliest_sinks_opt(self):
        """
        Check if sinks are executed at the earliest possible time using a simple example.

        Given the transformations
         * ``None -> a``
         * ``a -> b``
         * ``b -> c``
         * ``c -> d``
         * ``d -> None``
         * ``a -> None``

        ``a -> None`` one should be executed immediately after ``None -> a``.
        """
        flow = Autoflow()
        flow.add_transformation(TestTransformation(None, ["a"]))
        flow.add_transformation(TestTransformation(["a"], ["b"]))
        flow.add_transformation(TestTransformation(["b"], ["c"]))
        flow.add_transformation(TestTransformation(["c"], ["d"]))
        flow.add_transformation(TestTransformation(["d"], None))
        flow.add_transformation(RaiseExceptionTransformation(["a"], None, TestException))

        source = flow.transformations[0]
        sink = flow.transformations[-1]

        with self.assertRaises(TestException):
            flow.run(optimize="earliest_sinks")

        source_idx = flow.transformations.index(source)
        sink_idx = flow.transformations.index(sink)
        self.assertEqual(source_idx + 1, sink_idx)

    def test_min_annotations(self):
        """
        Checks if the min_annotations optimizer executes a transformation with a lot of annotations
        at the latest possible time.

        Given the transformations
         * ``None -> a``
         * ``a -> b``
         * ``b -> c``
         * ``a -> e, f, g, h, i, j``
         * ``c -> None``
         * ``e, f, g, h, i -> None``
         * ``j -> None``
         ``a -> e, f, g, h, i, j`` should be executed directly before ``e, f, g, h, i -> None`` and
         ``j -> None``, i.e. it should be in 5. position.

        """
        flow = Autoflow()
        flow.add_transformation(TestTransformation(None, ["a"]))
        flow.add_transformation(TestTransformation(["a"], ["b"]))
        flow.add_transformation(TestTransformation(["b"], ["c"]))
        flow.add_transformation(TestTransformation(["a"], ["e", "f", "g", "h", "i", "j"]))
        flow.add_transformation(TestTransformation(["c"], None))
        flow.add_transformation(TestTransformation(["e", "f", "g", "h", "i"], None))
        flow.add_transformation(RaiseExceptionTransformation(["j"], None, TestException))

        big_node = flow.transformations[3]

        with self.assertRaises(TestException):
            flow.run(optimize="min_annotations")

        big_node_idx = flow.transformations.index(big_node)

        self.assertEqual(big_node_idx, 4)

    def test_unused_annotations_detected(self):
        """
        Check if unused annotations raise an exception if ``allow_unused_annotations=False``.
        """
        flow = Autoflow()
        flow.add_transformation(TestTransformation(None, ["a"]))
        flow.add_transformation(TestTransformation(["a"], ["b"]))
        flow.add_transformation(RaiseExceptionTransformation(["a"], None, TestException))
        with self.assertRaises(AutoflowDefinitionError):
            flow.run(allow_unused_annotations=False)

    def test_unused_annotations_allowed_if_option_is_set(self):
        """
        Check if unused annotations do not raise an exception if ``allow_unused_annotations=True``.
        """
        flow = Autoflow()
        flow.add_transformation(TestTransformation(None, ["a"]))
        flow.add_transformation(TestTransformation(["a"], ["b"]))
        flow.add_transformation(RaiseExceptionTransformation(["a"], None, TestException))
        with self.assertRaises(TestException):
            flow.run(allow_unused_annotations=True)

    def test_if_all_connected_components_run(self):
        """
        The flow should also work if the DAG contains multiple connected components. Checks if all
        transformations are called at least once using a shared semaphore.
        """
        semaphore = Semaphore(4)
        flow = Autoflow()
        flow.add_transformation(CountingTestTransformation(None, ["a"], semaphore))
        flow.add_transformation(CountingTestTransformation(None, ["b"], semaphore))
        flow.add_transformation(CountingTestTransformation(["a"], None, semaphore))
        flow.add_transformation(CountingTestTransformation(["b"], None, semaphore))
        with self.assertRaises(CountingCompleteException):
            flow.run()

    def test_realistic_speed(self):
        """
        Test if the pipeline reaches > 1000 items/min.
        """
        flow = Autoflow()
        flow.add_transformation(TestTransformation(None, ["a"]))
        flow.add_transformation(TestTransformation(["a"], ["b"]))
        flow.add_transformation(TestTransformation(["a"], ["c"]))
        flow.add_transformation(TestTransformation(["b", "c"], ["d", "e"]))
        flow.add_transformation(TestTransformation(["e"], ["f"]))
        flow.add_transformation(TestTransformation(["d", "f"], ["g"]))
        flow.add_transformation(MeasureSpeedTransformation(
            requires=["g"],
            target_items_per_sec=1000,
            max_num_items=1000,
            print_speed_every=100)
        )
        with self.assertRaises(TargetSpeedReachedException):
            flow.run()


if __name__ == '__main__':
    unittest.main()
