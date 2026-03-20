"""Regression tests for protocols."""

import pytest
from sage.all import QQ

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
    ShiftedCooperad,
    ShiftedOperad,
    Surjection,
    SurjectionDual,
    CobarConstruction,
)
from uconf.core.quasi_planar import QuasiPlanarProtocol


@pytest.mark.parametrize("r", range(1, 5))
@pytest.mark.parametrize(
    "operad_cls",
    [Surjection, BarrattEccles, Associative, Commutative, Lie],
)
def test_operad_protocol(operad_cls: type, r: int) -> None:
    """Check that arity components satisfy the component-level protocol."""
    assert isinstance(operad_cls(r, QQ), OperadComponent), (
        f"{operad_cls.__name__} should satisfy OperadComponent."
    )


@pytest.mark.parametrize("r", range(1, 5))
@pytest.mark.parametrize(
    "operad_cls",
    [Surjection, BarrattEccles, Associative],
)
def test_planar_operad_protocol(operad_cls: type, r: int) -> None:
    """Check that arity components satisfy the component-level protocol."""
    assert isinstance(operad_cls(r, QQ), QuasiPlanarProtocol), (
        f"{operad_cls.__name__} should satisfy QuasiPlanarProtocol."
    )


@pytest.mark.parametrize("r", range(1, 5))
@pytest.mark.parametrize(
    "cooperad_cls",
    [SurjectionDual, CoAssociative, CoCommutative],
)
def test_cooperad_protocol(cooperad_cls: type, r: int) -> None:
    """Check that SurjectionDual components satisfy CooperadComponent."""
    assert isinstance(cooperad_cls(r, QQ), CooperadComponent), (
        f"{cooperad_cls.__name__} should satisfy CooperadComponent."
    )


@pytest.mark.parametrize("r", range(1, 5))
@pytest.mark.parametrize(
    "operad_cls",
    [Surjection, Associative, Commutative, Lie, BarrattEccles],
)
def test_cooperad_protocol_bar(operad_cls: type, r: int) -> None:
    """Check that the bar construction of Surjection satisfies CooperadComponent."""
    BarP = BarConstruction(operad_cls)(r, QQ)
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
    OmegaP = CobarConstruction(cooperad_cls)(r, QQ)
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
    had_component = HadamardProduct(left_operad_cls, right_operad_cls)(r, QQ)
    assert isinstance(had_component, OperadComponent), (
        f"HadamardProduct({left_operad_cls.__name__}, {right_operad_cls.__name__}) "
        "should satisfy OperadComponent."
    )


# ---------------------------------------------------------------------------
# unit_key tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("operad_cls", "expected_key"),
    [
        (Associative, (1,)),
        (Commutative, ()),
        (Lie, ()),
        (Surjection, (1,)),
    ],
)
def test_operad_class_unit_key(operad_cls: type, expected_key) -> None:
    """unit_key() on operad classes returns the correct constant."""
    assert operad_cls.unit_key() == expected_key, (
        f"{operad_cls.__name__}.unit_key() should return {expected_key!r}."
    )


@pytest.mark.parametrize(
    ("operad_cls", "expected_key"),
    [
        (Associative, (1,)),
        (Commutative, ()),
        (Lie, ()),
        (Surjection, (1,)),
    ],
)
def test_operad_instance_unit_key(operad_cls: type, expected_key) -> None:
    """unit_key() on arity-1 component instances returns the same key."""
    component = operad_cls(1, QQ)
    assert component.unit_key() == expected_key, (
        f"{operad_cls.__name__}(1, QQ).unit_key() should return {expected_key!r}."
    )


@pytest.mark.parametrize(
    "operad_cls",
    [Associative, Commutative, Lie, Surjection, BarrattEccles],
)
def test_unit_key_consistent_with_unit(operad_cls: type) -> None:
    """unit_key() should equal the sole basis key of unit()."""
    unit_elem = operad_cls.unit(QQ)
    support = list(unit_elem.support())
    assert len(support) == 1, f"{operad_cls.__name__}.unit() should have exactly one basis key."
    assert operad_cls.unit_key() == support[0], (
        f"{operad_cls.__name__}.unit_key() should match the basis key of unit()."
    )


@pytest.mark.parametrize(
    ("cooperad_cls", "expected_key"),
    [
        (CoAssociative, (1,)),
        (CoCommutative, ()),
        (SurjectionDual, (1,)),
    ],
)
def test_cooperad_unit_key(cooperad_cls: type, expected_key) -> None:
    """unit_key() on cooperad classes returns the counit generator key."""
    assert cooperad_cls.unit_key() == expected_key, (
        f"{cooperad_cls.__name__}.unit_key() should return {expected_key!r}."
    )


@pytest.mark.parametrize(
    "operad_cls",
    [Surjection, Associative, Commutative, Lie, BarrattEccles],
)
def test_bar_construction_unit_key(operad_cls: type) -> None:
    """BarConstruction unit_key is 1 (the single-leaf tree key)."""
    bar = BarConstruction(operad_cls)
    assert bar.unit_key() == 1, (
        f"BarConstruction({operad_cls.__name__}).unit_key() should return 1."
    )
    assert bar(1, QQ).unit_key() == 1


@pytest.mark.parametrize(
    "cooperad_cls",
    [SurjectionDual, CoAssociative, CoCommutative],
)
def test_cobar_construction_unit_key(cooperad_cls: type) -> None:
    """CobarConstruction unit_key is 1 (the single-leaf tree key)."""
    cobar = CobarConstruction(cooperad_cls)
    assert cobar.unit_key() == 1, (
        f"CobarConstruction({cooperad_cls.__name__}).unit_key() should return 1."
    )
    assert cobar(1, QQ).unit_key() == 1


@pytest.mark.parametrize(
    ("left_operad_cls", "right_operad_cls"),
    [
        (Surjection, Associative),
        (Lie, Commutative),
    ],
)
def test_hadamard_unit_key(left_operad_cls: type, right_operad_cls: type) -> None:
    """HadamardProduct unit_key is the pair of the factors' unit keys."""
    had = HadamardProduct(left_operad_cls, right_operad_cls)
    expected = (left_operad_cls.unit_key(), right_operad_cls.unit_key())
    assert had.unit_key() == expected, (
        f"HadamardProduct({left_operad_cls.__name__}, {right_operad_cls.__name__}).unit_key() "
        f"should return {expected!r}."
    )
    assert had(1, QQ).unit_key() == expected


@pytest.mark.parametrize(
    ("operad_cls", "shift", "expected_key"),
    [
        (Surjection, 1, (1,)),
        (Lie, 2, ()),
        (Associative, 1, (1,)),
    ],
)
def test_shifted_operad_unit_key(operad_cls: type, shift: int, expected_key) -> None:
    """ShiftedOperad unit_key delegates to the underlying operad."""
    shifted = ShiftedOperad(operad_cls, shift)
    assert shifted.unit_key() == expected_key, (
        f"ShiftedOperad({operad_cls.__name__}, {shift}).unit_key() should return {expected_key!r}."
    )
    assert shifted(1, QQ).unit_key() == expected_key


@pytest.mark.parametrize(
    ("cooperad_cls", "shift", "expected_key"),
    [
        (SurjectionDual, 1, (1,)),
        (CoCommutative, 2, ()),
        (CoAssociative, 1, (1,)),
    ],
)
def test_shifted_cooperad_unit_key(cooperad_cls: type, shift: int, expected_key) -> None:
    """ShiftedCooperad unit_key delegates to the underlying cooperad."""
    shifted = ShiftedCooperad(cooperad_cls, shift)
    assert shifted.unit_key() == expected_key, (
        f"ShiftedCooperad({cooperad_cls.__name__}, {shift}).unit_key() should return {expected_key!r}."
    )
    assert shifted(1, QQ).unit_key() == expected_key
