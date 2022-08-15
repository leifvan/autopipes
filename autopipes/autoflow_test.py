import unittest

from time import time
from typing import List, Optional
from multiprocessing import Semaphore
from autoflow import Transformation, Autoflow, AutoflowDefinitionError


class TestException(Exception):
    pass


class FastEnoughException(Exception):
    pass


class TestTransformation(Transformation):
    def __init__(
            self,
            requires: Optional[List[str]],
            adds: Optional[List[str]],
            test_speed: bool = False
    ):
        super(TestTransformation, self).__init__(requires=requires, adds=adds)
        self.start_time = time()
        self.num_items = 0
        self.test_speed = test_speed

    def apply(self, data):
        if data is None:
            data = dict()
        if self.adds is not None:
            for field in self.adds:
                data[field] = 0
            return data
        else:
            if self.test_speed and self.num_items > 0 and self.num_items % 100 == 0:
                duration = time() - self.start_time
                print(f"{self.num_items / duration:.2f} items/s")
                if self.num_items / duration > 1000:
                    raise FastEnoughException(f"Reached {self.num_items / duration:.2f} items/s.")
            self.num_items += 1


class CountingCompleteException(Exception):
    pass

class CountingTestTransformation(TestTransformation):
    def __init__(
            self,
            requires: Optional[List[str]],
            adds: Optional[List[str]],
            semaphore: Semaphore
    ):
        super(CountingTestTransformation, self).__init__(requires, adds, test_speed=False)
        self.semaphore = semaphore
        self.counted = False

    def apply(self, data):
        if not self.counted:
            self.semaphore.acquire()
            self.counted = True

        if self.semaphore.acquire(block=False):
            self.semaphore.release()
        else:
            raise CountingCompleteException("Semaphore can't be acquired.")

        return super(CountingTestTransformation, self).apply(data)


def test_faulty_transformation_fn(data):
    raise TestException("Just a test exception.")


def test_transformation_fn_x(data):
    data["x"] = 0
    return data


def test_transformation_fn_none(data):
    pass


class AutoflowTest(unittest.TestCase):
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
        flow.add_transformation_fn(test_faulty_transformation_fn, ["a"], None)
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
        flow.add_transformation_fn(test_faulty_transformation_fn, ["a"], None)

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
        flow.add_transformation_fn(test_faulty_transformation_fn, ["j"], None)

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
        flow.add_transformation_fn(test_faulty_transformation_fn, ["a"], None)
        with self.assertRaises(AutoflowDefinitionError):
            flow.run(allow_unused_annotations=False)

    def test_unused_annotations_allowed_if_option_is_set(self):
        """
        Check if unused annotations do not raise an exception if ``allow_unused_annotations=True``.
        """
        flow = Autoflow()
        flow.add_transformation(TestTransformation(None, ["a"]))
        flow.add_transformation(TestTransformation(["a"], ["b"]))
        flow.add_transformation_fn(test_faulty_transformation_fn, ["a"], None)
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
        flow.add_transformation(TestTransformation(["g"], None, test_speed=True))
        with self.assertRaises(FastEnoughException):
            flow.run()


if __name__ == '__main__':
    unittest.main()
