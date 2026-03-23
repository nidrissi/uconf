"""Display helpers shared by custom Sage element classes."""

from __future__ import annotations

from typing import Callable, Any


def latex_linear_combination(
    element,
    term_latex: Callable[[Any], str],
) -> str:
    """Format a module element as a LaTeX linear combination.

    Args:
        element: A Sage free-module element with iterable ``(basis, coeff)`` terms.
        term_latex: Callable mapping one basis key to a LaTeX term string
            without outer ``$...$``.
    """
    if not element:
        return "$0$"

    pieces: list[str] = []
    for basis, coeff in element:
        term = term_latex(basis)
        if coeff == 1:
            pieces.append(term)
        elif coeff == -1:
            pieces.append(f"-{term}")
        else:
            pieces.append(f"{coeff} \\left({term}\\right)")

    return "$" + " + ".join(pieces).replace("+ -", "- ") + "$"
