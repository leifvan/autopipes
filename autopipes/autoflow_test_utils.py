from multiprocessing import Queue
from time import time
from typing import Hashable, Callable, List, Container, TypeVar, Optional, Collection

from .autoflow import Transformation

T = TypeVar("T", bound=Container)


class SuccessfulComparisonException(Exception):
    """Raised if all items were successfully compared."""


class FailedComparisonException(Exception):
    """Raised if at least one item is not equal to the expected value."""


class CompareItemsTransformation(Transformation):
    """
    Compares a list of given data objects using the ``comparator_fn`` and
    raises a ``SuccessfulComparisonException`` if all given data objects
    are equal to the ones received in the pipeline. Otherwise, a
    ``FailedComparisonException`` is raised.

    If an expected item is ``None``, this item will not be compared.
    """

    def __init__(
            self,
            requires: Optional[Collection[Hashable]],
            expected_items: List[Optional[T]],
            comparator_fn: Callable[[T, T], T] = None
    ):
        super(CompareItemsTransformation, self).__init__(requires=requires, adds=None)
        self.expected_items = expected_items
        self.compare = comparator_fn
        self.num_items_compared = 0

    # noinspection PyMissingOrEmptyDocstring
    def apply(self, data: T):
        next_item = self.expected_items.pop(0)
        if next_item is not None and not self.compare(data, next_item):
            raise FailedComparisonException(f"Comparison failed. Expected '{next_item}', but got "
                                            f"'{data}'.")

        self.num_items_compared += 1

        if len(self.expected_items) == 0:
            raise SuccessfulComparisonException(f"Successfully compared {self.num_items_compared} "
                                                f"items.")


class CountItemsTransformation(CompareItemsTransformation):
    """
    Raises a ``SuccessfulComparisonException`` as soon as ``num_items`` items were processed
    by this transformation.
    """

    def __init__(self, requires: Optional[Collection[Hashable]], num_items: int):
        super(CountItemsTransformation, self).__init__(
            requires=requires,
            expected_items=[None] * num_items
        )


class TargetSpeedReachedException(Exception):
    """
    Raised if the processing speed of a ``MeasureSpeedTransformation`` reaches the given threshold.
    """


class TargetSpeedNotReachedException(Exception):
    """
    Raised if the processing speed of a ``MeasureSpeedTransformation`` does not reach the given
    threshold after the maximum number of items.
    """


class MeasureSpeedTransformation(CountItemsTransformation):
    """
    Runs until the arrival speed of items from the input queue reaches ``taret_items_per_sec``.
    Then, a ``TargetSpeedReachedException`` is thrown.
    """

    def __init__(
            self,
            requires: Optional[Collection[Hashable]],
            target_items_per_sec: float,
            max_num_items: int,
            print_speed_every: Optional[int] = None
    ):
        super(MeasureSpeedTransformation, self).__init__(requires=requires, num_items=max_num_items)
        self.target_items_per_sec = target_items_per_sec
        self.max_num_items = max_num_items
        self.start_time = None
        self.print_every = print_speed_every

    # noinspection PyMissingOrEmptyDocstring
    def thread(self):
        self.start_time = time()
        super(MeasureSpeedTransformation, self).thread()

    # noinspection PyMissingOrEmptyDocstring
    def apply(self, data: T):
        if self.num_items_compared >= self.max_num_items:
            raise TargetSpeedNotReachedException(f"Did not reach target speed of "
                                                 f"{self.target_items_per_sec} items/s after "
                                                 f"{self.max_num_items} items.")

        super(MeasureSpeedTransformation, self).apply(data)

        try:
            items_per_sec = self.num_items_compared / (time() - self.start_time)
        except ZeroDivisionError:
            items_per_sec = 0

        if items_per_sec >= self.target_items_per_sec:
            raise TargetSpeedReachedException(f"Reached target speed of "
                                              f"{self.target_items_per_sec} items/s.")
        elif self.print_every is not None and self.num_items_compared % self.print_every == 0:
            print(f"{items_per_sec} items/s")


class AllItemsPutException(Exception):
    """
    Raised by ``DummySourceTransformation`` if all items are put into the output queue and
    ``throw_exception_when_done`` was set to ``True``.
    """


class DummySourceTransformation(Transformation):
    """
    Puts given ``items`` into its output queue.
    """

    def __init__(
            self,
            adds: Optional[Collection[Hashable]],
            items: List[T],
            throw_exception_when_done: bool
    ):
        super(DummySourceTransformation, self).__init__(requires=None, adds=adds)
        self.items = items
        self.throw_exception_when_done = throw_exception_when_done

    # noinspection PyMissingOrEmptyDocstring
    def apply(self, data: T):
        if len(self.items) > 0:
            return self.items.pop(0)
        elif self.throw_exception_when_done:
            raise AllItemsPutException("All items were put into the output queue.")


class RaiseExceptionTransformation(Transformation):
    """
    Raises an exception of the given type on the first call of the apply method.
    """

    def __init__(
            self,
            requires: Optional[Collection[Hashable]],
            adds: Optional[Collection[Hashable]],
            exception_factory: Callable[[], Exception]
    ):
        super(RaiseExceptionTransformation, self).__init__(
            requires=requires,
            adds=adds
        )
        self.exception_factory = exception_factory

    def apply(self, data: T):
        raise self.exception_factory()


class AllItemsCollectedException(Exception):
    """Raised if ``CollectInputsTransformation`` has collected the required number of items."""


class CollectInputsTransformation(Transformation):
    """
    Collects ``num_items`` items and puts them into a queue ``self.collected_items``. Raises an
    ``AllItemsCollectedException`` when the desired number of items was collected.
    """

    def __init__(self, requires: Optional[Collection[Hashable]], num_items: int):
        super(CollectInputsTransformation, self).__init__(requires=requires, adds=None)
        self.num_items_to_collect = num_items
        self.num_items_collected = 0
        self.collected_items = Queue()

    def apply(self, data: T):
        self.collected_items.put(data)
        self.num_items_collected += 1
        if self.num_items_collected >= self.num_items_to_collect:
            raise AllItemsCollectedException(f"Successfully collected {self.num_items_collected} "
                                             f"items.")
