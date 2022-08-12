import unittest

from time import time
from typing import List, Optional

from autoflow import Transformation, Autoflow, NoTransformationFoundException, \
    NotASinkException, NoSourceException, SinkException


class TestException(Exception):
    pass


class FastEnoughException(Exception):
    pass


class TestTransformation(Transformation):
    def __init__(self, requires: Optional[List[str]], adds: Optional[List[str]]):
        super(TestTransformation, self).__init__(requires=requires, adds=adds)
        self.start_time = time()
        self.num_items = 0

    def apply(self, data):
        if data is None:
            data = dict()
        if self.adds is not None:
            for field in self.adds:
                data[field] = 0
            return data
        else:
            if self.num_items > 0 and self.num_items % 10 == 0:
                duration = time() - self.start_time
                print(f"{self.num_items / duration:.2f} items/s")
                if self.num_items / duration > 1000:
                    raise FastEnoughException(f"Reached {self.num_items / duration:.2f} items/s.")
            self.num_items += 1


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
        Flow has a source but no sink. It should complain about not being able to process data
        further.
        """
        flow = Autoflow()
        flow.add_transformation(TestTransformation(None, ["a"]))
        with self.assertRaises(NoTransformationFoundException):
            flow.run()

    def test_missing_source(self):
        """
        Flow should recognize that no source node is present.
        """
        flow = Autoflow()
        flow.add_transformation(TestTransformation(["a"], ["b"]))
        with self.assertRaises(NoSourceException):
            flow.run()

    def test_exception_caught(self):
        flow = Autoflow()
        flow.add_transformation(TestTransformation(None, ["a"]))
        flow.add_transformation_fn(test_faulty_transformation_fn, ["a"], ["b"])
        with self.assertRaises(TestException):
            flow.run()

    def test_non_sink_returns_none(self):
        flow = Autoflow()
        flow.add_transformation(TestTransformation(None, ["a"]))
        flow.add_transformation_fn(test_transformation_fn_none, ["a"], ["b"])
        with self.assertRaises(NotASinkException):
            flow.run()

    def test_sink_returns_something(self):
        flow = Autoflow()
        flow.add_transformation(TestTransformation(None, ["a"]))
        flow.add_transformation_fn(test_transformation_fn_x, ["a"], None)
        with self.assertRaises(SinkException):
            flow.run()

    def test_realistic_speed(self):
        flow = Autoflow()
        flow.add_transformation(TestTransformation(None, ["a"]))
        flow.add_transformation(TestTransformation(["a"], ["b"]))
        flow.add_transformation(TestTransformation(["a"], ["c"]))
        flow.add_transformation(TestTransformation(["b", "c"], ["d", "e"]))
        flow.add_transformation(TestTransformation(["e"], ["f"]))
        flow.add_transformation(TestTransformation(["d", "f"], ["g"]))
        flow.add_transformation(TestTransformation(["g"], None))
        with self.assertRaises(FastEnoughException):
            flow.run()


if __name__ == '__main__':
    unittest.main()
