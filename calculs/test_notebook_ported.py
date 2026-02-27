"""Port of core tests from comch/test.ipynb to current uconf operads.

Notes:
- This file ports the tests that map directly to current APIs.
- Old notebook tests relying on removed helpers/properties (e.g. filtration,
  planarize_surjection, comod_surj) are intentionally omitted.
- Permutations are handled via current Sage permutation objects.
"""

from collections.abc import Iterable

import pytest

from uconf import BarrattEccles, Surjection


def _degree_matches(element, d: int) -> bool:
    parent = element.parent()
    return all(parent.degree_on_basis(key) == d for key in element.support())


def usual_planar_test(rmax: int, dmax: int) -> Iterable[Surjection.Element]:
    for r in range(1, rmax):
        for d in range(1, dmax):
            yield from Surjection(r).planar_basis_it(d)


PLANAR_SMALL = tuple(usual_planar_test(4, 3))
PLANAR_LARGE = tuple(usual_planar_test(6, 3))


@pytest.mark.parametrize("r", range(1, 5))
@pytest.mark.parametrize("d", range(0, 3))
def test_be_basis(r: int, d: int) -> None:
    basis = list(BarrattEccles(r).basis_it(d))
    for el in basis:
        assert isinstance(
            el, BarrattEccles.Element
        ), "Non-BarrattEccles element found in basis_it"
        assert el != el.parent().zero(), "Zero element found in basis_it"
        assert el.arity() == r, "Element with incorrect arity in basis_it"
        assert _degree_matches(
            el, d
        ), "Element with incorrect degree in BarrattEccles basis_it"


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
    basis = list(Surjection(r).planar_basis_it(d))
    for el in basis:
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
@pytest.mark.parametrize("s1", PLANAR_SMALL)
@pytest.mark.parametrize("s2", PLANAR_SMALL)
@pytest.mark.parametrize("pos", range(1, 5))
def test_section_composition(s1: Surjection.Element, s2: Surjection.Element, pos: int):
    """Check that the section map commutes with composition."""
    composed = Surjection.compose(s1, pos, s2)
    sect_composed = composed.section()

    sect1 = s1.section()
    sect2 = s2.section()
    composed_sect = Surjection.compose(sect1, pos, sect2)
    assert (
        sect_composed == composed_sect
    ), f"Section composition failed for {s1} and {s2} at position {pos}."


@pytest.mark.xfail
@pytest.mark.parametrize("s", PLANAR_SMALL)
def test_section_boundary(s: Surjection.Element):
    """Check that the section map commutes with the boundary map."""
    sect = s.section()
    sect_boundary = sect.boundary()
    s_boundary = s.boundary()
    sect_of_boundary = s_boundary.section()
    assert sect_boundary == sect_of_boundary, f"Section boundary failed for {s}."
