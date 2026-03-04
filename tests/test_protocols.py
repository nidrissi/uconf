"""Regression tests for protocols."""

import pytest

from uconf import (
    Associative,
    BarConstruction,
    BarrattEccles,
    CoAssociative,
    CoCommutative,
    Commutative,
    CooperadProtocol,
    Lie,
    OperadProtocol,
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
    assert isinstance(operad_cls(r), OperadProtocol), (
        f"{operad_cls.__name__} should satisfy OperadProtocol."
    )


@pytest.mark.parametrize("r", range(1, 5))
@pytest.mark.parametrize(
    "cooperad_cls",
    [SurjectionDual, CoAssociative, CoCommutative],
)
def test_cooperad_protocol(cooperad_cls: type, r: int) -> None:
    """Check that SurjectionDual components satisfy CooperadProtocol."""

    assert isinstance(cooperad_cls(r), CooperadProtocol), (
        f"{cooperad_cls.__name__} should satisfy CooperadProtocol."
    )


@pytest.mark.parametrize("r", range(1, 5))
@pytest.mark.parametrize(
    "operad_cls",
    [Surjection, Associative, Commutative, Lie, BarrattEccles],
)
def test_cooperad_protocol_bar(operad_cls: type, r: int) -> None:
    """Check that the bar construction of Surjection satisfies CooperadProtocol."""

    BarP = BarConstruction(operad_cls)(r)
    assert isinstance(BarP, CooperadProtocol), (
        f"BarConstruction({operad_cls.__name__}) should satisfy CooperadFactoryProtocol."
    )


@pytest.mark.parametrize("r", range(1, 5))
@pytest.mark.parametrize(
    "cooperad_cls",
    [SurjectionDual, CoAssociative, CoCommutative],
)
def test_operad_protocol_cobar(cooperad_cls: type, r: int) -> None:
    """Check that the cobar construction of SurjectionDual satisfies CooperadProtocol."""

    OmegaP = CobarConstruction(cooperad_cls)(r)
    assert isinstance(OmegaP, OperadProtocol), (
        f"CobarConstruction({cooperad_cls.__name__}) should satisfy OperadFactoryProtocol."
    )
