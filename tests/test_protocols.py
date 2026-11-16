"""Regression tests for the operad/cooperad base classes and protocols."""

import pickle

import pytest
from sage.all import QQ

from uconf import (
    Associative,
    BarConstruction,
    BarrattEccles,
    CoAssociative,
    CobarConstruction,
    CoCommutative,
    Commutative,
    ComponentBase,
    Cooperad,
    ElementBase,
    HadamardProduct,
    Lie,
    Operad,
    ShiftedCooperad,
    ShiftedOperad,
    Surjection,
    SurjectionDual,
)
from uconf.core.cooperad import CooperadElement
from uconf.core.quasi_planar import QuasiPlanarProtocol

OPERADS = [
    Surjection,
    BarrattEccles,
    Associative,
    Commutative,
    Lie,
    ShiftedOperad(Lie, 1),
    HadamardProduct(Surjection, Associative),
    HadamardProduct(Lie, Commutative),
    HadamardProduct(BarrattEccles, Surjection),
    CobarConstruction(SurjectionDual),
    CobarConstruction(CoAssociative),
    CobarConstruction(CoCommutative),
]

COOPERADS = [
    SurjectionDual,
    CoAssociative,
    CoCommutative,
    ShiftedCooperad(SurjectionDual, 1),
    BarConstruction(Surjection),
    BarConstruction(Associative),
    BarConstruction(Commutative),
    BarConstruction(Lie),
    BarConstruction(BarrattEccles),
]


@pytest.mark.parametrize("r", range(1, 5))
@pytest.mark.parametrize("operad", OPERADS)
def test_operad_component(operad: Operad, r: int) -> None:
    """Operads are ``Operad`` instances whose components point back to them."""
    assert isinstance(operad, Operad)
    assert not isinstance(operad, Cooperad)
    component = operad(r, QQ)
    assert isinstance(component, ComponentBase)
    assert component.factory is operad
    assert component.arity() == r
    assert component.connectivity == operad.connectivity
    assert isinstance(operad.unit(QQ), ElementBase)


@pytest.mark.parametrize("r", range(1, 5))
@pytest.mark.parametrize("cooperad", COOPERADS)
def test_cooperad_component(cooperad: Cooperad, r: int) -> None:
    """Cooperads are ``Cooperad`` instances whose elements carry the structure maps."""
    assert isinstance(cooperad, Cooperad)
    assert not isinstance(cooperad, Operad)
    component = cooperad(r, QQ)
    assert isinstance(component, ComponentBase)
    assert component.factory is cooperad
    assert issubclass(component.element_class, CooperadElement)


@pytest.mark.parametrize("sequence", OPERADS + COOPERADS)
def test_sequence_is_unique_and_picklable(sequence) -> None:
    """Operads and their components survive pickling as the same objects."""
    assert pickle.loads(pickle.dumps(sequence)) is sequence
    component = sequence(2, QQ)
    assert pickle.loads(pickle.dumps(component)) is component


@pytest.mark.parametrize("r", range(1, 5))
@pytest.mark.parametrize(
    "operad_cls",
    [Surjection, BarrattEccles, Associative],
)
def test_planar_operad_protocol(operad_cls: Operad, r: int) -> None:
    """Check that arity components satisfy the quasi-planar protocol."""
    assert isinstance(operad_cls(r, QQ), QuasiPlanarProtocol), (
        f"{operad_cls.__name__} should satisfy QuasiPlanarProtocol."
    )


def test_dual_components_reuse_primal_modules() -> None:
    """The dual models are cooperads built on the primal graded modules."""
    assert isinstance(CoAssociative(3, QQ), Associative.Component)
    assert isinstance(SurjectionDual(3, QQ), Surjection.Component)
    assert CoAssociative(3, QQ) is not Associative(3, QQ)
    assert CoAssociative(3, QQ).factory is CoAssociative


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
def test_operad_class_unit_key(operad_cls: Operad, expected_key) -> None:
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
def test_operad_instance_unit_key(operad_cls: Operad, expected_key) -> None:
    """unit_key() on arity-1 component instances returns the same key."""
    component = operad_cls(1, QQ)
    assert component.unit_key() == expected_key, (
        f"{operad_cls.__name__}(1, QQ).unit_key() should return {expected_key!r}."
    )


@pytest.mark.parametrize(
    "operad_cls",
    [Associative, Commutative, Lie, Surjection, BarrattEccles],
)
def test_unit_key_consistent_with_unit(operad_cls: Operad) -> None:
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
def test_cooperad_unit_key(cooperad_cls: Cooperad, expected_key) -> None:
    """unit_key() on cooperad classes returns the counit generator key."""
    assert cooperad_cls.unit_key() == expected_key, (
        f"{cooperad_cls.__name__}.unit_key() should return {expected_key!r}."
    )


@pytest.mark.parametrize(
    "operad_cls",
    [Surjection, Associative, Commutative, Lie, BarrattEccles],
)
def test_bar_construction_unit_key(operad_cls: Operad) -> None:
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
def test_cobar_construction_unit_key(cooperad_cls: Cooperad) -> None:
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
def test_hadamard_unit_key(left_operad_cls: Operad, right_operad_cls: Operad) -> None:
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
def test_shifted_operad_unit_key(operad_cls: Operad, shift: int, expected_key) -> None:
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
def test_shifted_cooperad_unit_key(cooperad_cls: Cooperad, shift: int, expected_key) -> None:
    """ShiftedCooperad unit_key delegates to the underlying cooperad."""
    shifted = ShiftedCooperad(cooperad_cls, shift)
    assert shifted.unit_key() == expected_key, (
        f"ShiftedCooperad({cooperad_cls.__name__}, {shift}).unit_key() should return {expected_key!r}."
    )
    assert shifted(1, QQ).unit_key() == expected_key
