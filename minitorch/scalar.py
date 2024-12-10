from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    EQ,
    LT,
    Add,
    Exp,
    Inv,
    Log,
    Mul,
    Neg,
    ReLU,
    ScalarFunction,
    Sigmoid,
)

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """`ScalarHistory` stores the history of `Function` operations that was used to construct the current Variable."""

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()


_var_count = 0


class Scalar:
    """A reimplementation of scalar values for autodifferentiation tracking."""

    history: Optional[ScalarHistory]
    derivative: Optional[float]
    data: float
    unique_id: int
    name: str

    def __init__(
        self,
        v: float,
        back: ScalarHistory = ScalarHistory(),
        name: Optional[str] = None,
    ):
        global _var_count
        _var_count += 1
        self.unique_id = _var_count
        self.data = float(v)
        self.history = back
        self.derivative = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

    def __repr__(self) -> str:
        """Return a string representation of the scalar."""
        return "Scalar(%f)" % self.data

    def __mul__(self, b: ScalarLike) -> Scalar:
        """Multiply two scalars."""
        return Mul.apply(self, b)

    def __truediv__(self, b: ScalarLike) -> Scalar:
        """Divide two scalars."""
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        """Divide two scalars."""
        return Mul.apply(b, Inv.apply(self))

    def __add__(self, b: ScalarLike) -> Scalar:
        """Add two scalars."""
        return Add.apply(self, b)

    def __bool__(self) -> bool:
        """Convert a scalar to a boolean."""
        return bool(self.data)

    def __lt__(self, b: ScalarLike) -> Scalar:
        """Less than comparison between two scalars."""
        return LT.apply(self, b)

    def __gt__(self, b: ScalarLike) -> Scalar:
        """Greater than comparison between two scalars."""
        return LT.apply(b, self)

    def __eq__(self, b: ScalarLike) -> Scalar:
        """Equal comparison between two scalars."""
        return EQ.apply(self, b)

    def __sub__(self, b: ScalarLike) -> Scalar:
        """Subtract two scalars."""
        return Add.apply(self, Neg.apply(b))

    def __neg__(self) -> Scalar:
        """Negate a scalar."""
        return Neg.apply(self)

    def __radd__(self, b: ScalarLike) -> Scalar:
        """Right add a scalar."""
        return self + b

    def __rmul__(self, b: ScalarLike) -> Scalar:
        """Right multiply a scalar."""
        return self * b

    def log(self) -> Scalar:
        """Logarithm of a scalar."""
        return Log.apply(self)

    def exp(self) -> Scalar:
        """Exponential of a scalar."""
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        """Sigmoid of a scalar."""
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        """ReLU of a scalar."""
        return ReLU.apply(self)

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable."""
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.derivative = 0.0
        self.derivative += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """True if this variable is a constant."""
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """The parent variables of this variable."""
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """The chain rule for this variable."""
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        return list(zip(h.inputs, x))

    def backward(self, d_output: Optional[float] = None) -> None:
        """Calls autodiff to fill in the derivatives for the history of this object."""
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """Checks that autodiff works on a python function."""
    out = f(*scalars)
    out.backward()

    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        print(str([x.data for x in scalars]), x.derivative, i, check)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )
