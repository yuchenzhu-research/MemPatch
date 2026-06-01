import random
from typing import List, Tuple, TypeVar

T = TypeVar("T")


def split_list(
    items: List[T],
    dev_ratio: float = 0.2,
    test_ratio: float = 0.7,
    seed: int = 42,
) -> Tuple[List[T], List[T], List[T]]:
    """Deterministically split items into dev, public_test, and private_test."""
    # We want to be deterministic, so copy and sort
    sorted_items = list(items)
    # Shuffle with a local random instance to avoid global state changes
    rng = random.Random(seed)
    rng.shuffle(sorted_items)

    n = len(sorted_items)
    n_dev = int(n * dev_ratio)
    n_test = int(n * test_ratio)

    dev = sorted_items[:n_dev]
    test = sorted_items[n_dev : n_dev + n_test]
    private = sorted_items[n_dev + n_test :]

    return dev, test, private
