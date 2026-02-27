"""Regression tests for the Surjection operad."""

from collections.abc import Iterable
import math

import pytest

from uconf import Surjection


def _degree_matches(element, d: int) -> bool:
    parent = element.parent()
    return all(parent.degree_on_basis(key) == d for key in element.support())


def _canonical_basis_key(basis):
    if not isinstance(basis, tuple):
        return basis

    canonical = []
    for item in basis:
        if hasattr(item, "tuple"):
            canonical.append(tuple(item.tuple()))
        else:
            canonical.append(item)
    return tuple(canonical)


def _as_dict(x):
    return {_canonical_basis_key(basis): coeff for basis, coeff in x}


def usual_planar_test(rmax: int, dmax: int) -> Iterable[Surjection.Element]:
    for r in range(1, rmax):
        for d in range(1, dmax):
            yield from Surjection(r).planar_basis_it(d)


PLANAR_XSMALL = tuple(usual_planar_test(4, 2))
PLANAR_SMALL = tuple(usual_planar_test(4, 3))
PLANAR_LARGE = tuple(usual_planar_test(6, 3))


def test_surjection_unit() -> None:
    u = Surjection.unit()
    assert _as_dict(u) == {(1,): 1}, "Surjection unit should be (1,)."


@pytest.mark.parametrize("sigma", ([2, 1],))
def test_surjection_permutation_action(sigma: list[int]) -> None:
    s2 = Surjection(2)
    x = s2((1, 2))
    swapped = x.permute(sigma)
    assert _as_dict(swapped) == {(2, 1): 1}, "Expected transposed surjection tuple."
    assert _as_dict(swapped.permute(sigma)) == _as_dict(
        x
    ), "Applying the same transposition twice should return the original element."


def test_surjection_basic_composition() -> None:
    s2 = Surjection(2)
    x = s2((1, 2))
    composed = Surjection.compose(x, 1, x)
    assert _as_dict(composed) == {(1, 2, 3): 1}, "Expected basis element (1,2,3)."


@pytest.mark.parametrize("input_pos", [1, 2, 3])
def test_surjection_operadic_unit_axioms(input_pos: int) -> None:
    s3 = Surjection(3)
    x = s3((1, 2, 3))
    one = Surjection.unit()

    left = Surjection.compose(one, 1, x)
    right = Surjection.compose(x, input_pos, one)

    x_dict = _as_dict(x)
    assert _as_dict(left) == x_dict, "Left unit axiom failed for Surjection."
    assert _as_dict(right) == x_dict, f"Right unit axiom failed at input {input_pos}."


@pytest.mark.parametrize("r", range(1, 5))
@pytest.mark.parametrize("d", range(0, 5))
def test_surjection_basis(r: int, d: int) -> None:
    basis = list(Surjection(r).basis_it(d))
    assert len(basis) == len(set(basis)), "Duplicate elements found in basis_it"
    for el in basis:
        assert isinstance(
            el, Surjection.Element
        ), "Non-Surjection element found in basis_it"
        assert el != el.parent().zero(), "Zero element found in basis_it"
        assert el.arity() == r, "Element with incorrect arity in basis_it"
        assert _degree_matches(
            el, d
        ), "Element with incorrect degree in Surjection basis_it"


@pytest.mark.parametrize("r", range(2, 5))
@pytest.mark.parametrize("d", range(1, 4))
def test_planar_surjection_basis(r: int, d: int) -> None:
    planar_basis = list(Surjection(r).planar_basis_it(d))
    basis = list(Surjection(r).basis_it(d))
    assert set(planar_basis).issubset(
        basis
    ), "Planar basis should be a subset of the full basis."
    assert len(planar_basis) * math.factorial(r) == len(
        basis
    ), "Planar basis size should be full basis size divided by r!."

    for el in planar_basis:
        assert el.is_planar(), "Non-planar element found in planar_basis_it"


@pytest.mark.parametrize("s1", PLANAR_SMALL)
@pytest.mark.parametrize("s2", PLANAR_SMALL)
def test_planar_preserved_under_composition_last_input(
    s1: Surjection.Element, s2: Surjection.Element
) -> None:
    pos = s1.arity()
    composed = Surjection.compose(s1, pos, s2)
    assert (
        composed.is_planar()
    ), f"Composition of {s1} and {s2} at position {pos} is not planar."


@pytest.mark.parametrize("s", PLANAR_LARGE)
def test_section_right_inverse(s: Surjection.Element) -> None:
    sect = s.section()
    nat = sect.table_reduction()
    assert nat == s, f"Section failed for {s}, got {nat} instead."


@pytest.mark.parametrize("s", PLANAR_LARGE)
def test_section_planar(s: Surjection.Element) -> None:
    r = s.arity()
    sect = s.section()
    for key in sect.support():
        first_perm = key[0]
        assert tuple(first_perm.tuple()) == tuple(
            range(1, r + 1)
        ), f"Section of planar surjection {s} is not planar at {key}."


@pytest.mark.xfail
@pytest.mark.parametrize("s1", PLANAR_XSMALL)
@pytest.mark.parametrize("s2", PLANAR_XSMALL)
def test_section_composition(s1: Surjection.Element, s2: Surjection.Element):
    """Check that the section map commutes with composition."""
    composed = Surjection.compose(s1, 1, s2)
    sect_composed = composed.section()

    sect1 = s1.section()
    sect2 = s2.section()
    composed_sect = Surjection.compose(sect1, 1, sect2)
    assert (
        sect_composed == composed_sect
    ), f"Section composition failed for {s1} and {s2} at position 1."


@pytest.mark.xfail
@pytest.mark.parametrize("s", PLANAR_SMALL)
def test_section_boundary(s: Surjection.Element):
    """Check that the section map commutes with the boundary map."""
    sect = s.section()
    sect_boundary = sect.boundary()
    s_boundary = s.boundary()
    sect_of_boundary = s_boundary.section()
    assert sect_boundary == sect_of_boundary, f"Section boundary failed for {s}."
