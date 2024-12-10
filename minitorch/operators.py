"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    """$f(x, y) = x * y$"""
    # TODO: Implement for Task 0.1.
    return x * y


def id(x: float) -> float:
    """$f(x) = x$"""
    return x


def add(x: float, y: float) -> float:
    """$f(x, y) = x + y$"""
    return x + y
    raise NotImplementedError("Need to implement for Task 0.1")


def neg(x: float) -> float:
    """$f(x) = -x$"""
    return -1.0 * x


def lt(x: float, y: float) -> float:
    """$f(x) =$ 1.0 if x is less than y else 0.0"""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """$f(x) =$ 1.0 if x is equal to y else 0.0"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """$f(x) =$ x if x is greater than y else y"""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """$f(x) = |x - y| < 1e-2$"""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    r"""$f(x) =  \frac{1.0}{(1.0 + e^{-x})}$"""
    # TODO: Implement for Task 0.1.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))
    raise NotImplementedError("Need to implement for Task 0.1")


def relu(x: float) -> float:
    """$f(x) =$ x if x is greater than 0, else 0"""
    # TODO: Implement for Task 0.1.
    return x if x > 0 else 0.0
    raise NotImplementedError("Need to implement for Task 0.1")


EPS = 1e-6


def log(x: float) -> float:
    """$f(x) = log(x)$"""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """$f(x) = e^{x}$"""
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    r"""If $f = log$ as above, compute $d \times f'(x)$"""
    return d / (x + EPS)


def inv(x: float) -> float:
    """$f(x) = 1/x$"""
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    r"""If $f(x) = 1/x$ compute $d \times f'(x)$"""
    return -(1.0 / x**2) * d


def relu_back(x: float, d: float) -> float:
    r"""If $f = relu$ compute $d \times f'(x)$"""
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order map."""

    # TODO: Implement for Task 0.3.
    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Use `map` and `neg` to negate each element in `ls`"""
    return map(neg)(ls)


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order zipwith (or map2)."""

    # TODO: Implement for Task 0.3.
    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipWith
    raise NotImplementedError("Need to implement for Task 0.3")


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add the elements of `ls1` and `ls2` using `zipWith` and `add`"""
    # TODO: Implement for Task 0.3.
    return zipWith(add)(ls1, ls2)
    raise NotImplementedError("Need to implement for Task 0.3")


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""Higher-order reduce."""

    # TODO: Implement for Task 0.3.
    def reduce_func(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return reduce_func
    raise NotImplementedError("Need to implement for Task 0.3")


def sum(ls: Iterable[float]) -> float:
    """Sum up a list using `reduce` and `add`."""
    # TODO: Implement for Task 0.3.
    return reduce(add, 0.0)(ls)
    raise NotImplementedError("Need to implement for Task 0.3")


def prod(ls: Iterable[float]) -> float:
    """Product of a list using `reduce` and `mul`."""
    # TODO: Implement for Task 0.3.
    return reduce(mul, 1.0)(ls)
    raise NotImplementedError("Need to implement for Task 0.3")
