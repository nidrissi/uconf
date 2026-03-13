"""Regression tests for protocols."""

import pytest

from uconf import (
    Associative,
    BarConstruction,
    BarrattEccles,
    HadamardProduct,
    CoAssociative,
    CoCommutative,
    Commutative,
    CooperadComponent,
    Lie,
    OperadComponent,
    Surjection,
    SurjectionDual,
    CobarConstruction,
)


@pytest.mark.parametrize("r", range(1, 5))
@pytest.mark.parametrize(
    "operad_cls",
    [Surjection, BarrattEccles, Associative, Commutative, Lie],
)
def test_operad_protocol(operad_cls: type, r: int) -> None:
    """Check that arity components satisfy the component-level protocol."""
    assert isinstance(operad_cls(r), OperadComponent), (
        f"{operad_cls.__name__} should satisfy OperadComponent."
    )


@pytest.mark.parametrize("r", range(1, 5))
@pytest.mark.parametrize(
    "cooperad_cls",
    [SurjectionDual, CoAssociative, CoCommutative],
)
def test_cooperad_protocol(cooperad_cls: type, r: int) -> None:
    """Check that SurjectionDual components satisfy CooperadComponent."""

    assert isinstance(cooperad_cls(r), CooperadComponent), (
        f"{cooperad_cls.__name__} should satisfy CooperadComponent."
    )


@pytest.mark.parametrize("r", range(1, 5))
@pytest.mark.parametrize(
    "operad_cls",
    [Surjection, Associative, Commutative, Lie, BarrattEccles],
)
def test_cooperad_protocol_bar(operad_cls: type, r: int) -> None:
    """Check that the bar construction of Surjection satisfies CooperadComponent."""

    BarP = BarConstruction(operad_cls)(r)
    assert isinstance(BarP, CooperadComponent), (
        f"BarConstruction({operad_cls.__name__}) should satisfy CooperadFactory."
    )


@pytest.mark.parametrize("r", range(1, 5))
@pytest.mark.parametrize(
    "cooperad_cls",
    [SurjectionDual, CoAssociative, CoCommutative],
)
def test_operad_protocol_cobar(cooperad_cls: type, r: int) -> None:
    """Check that the cobar construction of SurjectionDual satisfies CooperadComponent."""

    OmegaP = CobarConstruction(cooperad_cls)(r)
    assert isinstance(OmegaP, OperadComponent), (
        f"CobarConstruction({cooperad_cls.__name__}) should satisfy OperadFactory."
    )


@pytest.mark.parametrize("r", range(1, 5))
@pytest.mark.parametrize(
    ("left_operad_cls", "right_operad_cls"),
    [
        (Surjection, Associative),
        (Lie, Commutative),
        (BarrattEccles, Surjection),
    ],
)
def test_operad_protocol_hadamard(
    left_operad_cls: type,
    right_operad_cls: type,
    r: int,
) -> None:
    """Check that Hadamard-product components satisfy OperadComponent."""

    had_component = HadamardProduct(left_operad_cls, right_operad_cls)(r)
    assert isinstance(had_component, OperadComponent), (
        f"HadamardProduct({left_operad_cls.__name__}, {right_operad_cls.__name__}) "
        "should satisfy OperadComponent."
    )
