from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple
from typing_extensions import Protocol


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg."""
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    """Represents a node in the computation graph."""

    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the derivative value `x` into the variable."""
        pass

    @property
    def unique_id(self) -> int:
        """Unique ID for every variable."""
        return 0

    def is_leaf(self) -> bool:
        """Check if the variable is a leaf node."""
        return True

    def is_constant(self) -> bool:
        """Check if the variable is a constant."""
        return True

    @property
    def parents(self) -> Iterable["Variable"]:
        """Get the parent variables."""
        return []

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        """Apply the chain rule to compute the derivative of the variable."""
        return []


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    order: List[Variable] = []
    seen = set()

    def visit(var: Variable) -> None:
        if var.unique_id in seen or var.is_constant():
            return
        if not var.is_leaf():
            for m in var.parents:
                if not m.is_constant():
                    visit(m)
        seen.add(var.unique_id)
        order.insert(0, var)

    visit(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to compute derivatives for the leave nodes."""
    queue = topological_sort(variable)
    derivatives = {}
    derivatives[variable.unique_id] = deriv

    for var in queue:
        deriv = derivatives[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(deriv)
        else:
            for v, d in var.chain_rule(deriv):
                if v.is_constant():
                    continue
                derivatives.setdefault(v.unique_id, 0.0)
                derivatives[v.unique_id] = derivatives[v.unique_id] + d


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Get the saved values."""
        return self.saved_values
