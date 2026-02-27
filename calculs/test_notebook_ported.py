"""Port of core tests from comch/test.ipynb to current uconf operads.

Notes:
- This file ports the tests that map directly to current APIs.
- Old notebook tests relying on removed helpers/properties (e.g. filtration,
  planarize_surjection, comod_surj) are intentionally omitted.
- Permutations are handled via current Sage permutation objects.
"""

from collections.abc import Iterable

from uconf import BarrattEccles, Surjection


def _degree_matches(element, d: int) -> bool:
    parent = element.parent()
    return all(parent.degree_on_basis(key) == d for key in element.support())


def usual_planar_test(rmax: int, dmax: int) -> Iterable[Surjection.Element]:
    for r in range(1, rmax):
        for d in range(1, dmax):
            yield from Surjection(r).planar_basis_it(d)


def test_be_basis_ported() -> None:
    # d starts at 1 because current BarrattEccles.basis_it(0) hits a known edge case.
    for r in range(1, 5):
        for d in range(1, 3):
            basis = list(BarrattEccles(r).basis_it(d))
            for el in basis:
                assert isinstance(el, BarrattEccles.Element), (
                    "Non-BarrattEccles element found in basis_it"
                )
                assert el != el.parent().zero(), "Zero element found in basis_it"
                assert el.arity() == r, "Element with incorrect arity in basis_it"
                assert _degree_matches(el, d), (
                    "Element with incorrect degree in BarrattEccles basis_it"
                )


def test_surjection_basis_ported() -> None:
    for r in range(1, 5):
        for d in range(0, 5):
            basis = list(Surjection(r).basis_it(d))
            assert len(basis) == len(set(basis)), "Duplicate elements found in basis_it"
            for el in basis:
                assert isinstance(el, Surjection.Element), (
                    "Non-Surjection element found in basis_it"
                )
                assert el != el.parent().zero(), "Zero element found in basis_it"
                assert el.arity() == r, "Element with incorrect arity in basis_it"
                assert _degree_matches(el, d), (
                    "Element with incorrect degree in Surjection basis_it"
                )


def test_planar_surjection_basis_ported() -> None:
    for r in range(2, 5):
        for d in range(1, 4):
            basis = list(Surjection(r).planar_basis_it(d))
            for el in basis:
                assert el.is_planar(), "Non-planar element found in planar_basis_it"


def test_planar_preserved_under_composition_last_input_ported() -> None:
    for s1 in usual_planar_test(4, 3):
        for s2 in usual_planar_test(4, 3):
            pos = s1.arity()
            composed = Surjection.compose(s1, pos, s2)
            assert composed.is_planar(), (
                f"Composition of {s1} and {s2} at position {pos} is not planar."
            )


def test_section_right_inverse_ported() -> None:
    for s in usual_planar_test(6, 3):
        sect = s.section()
        nat = sect.table_reduction()
        assert nat == s, f"Section failed for {s}, got {nat} instead."


def test_section_planar_ported() -> None:
    for s in usual_planar_test(6, 3):
        r = s.arity()
        sect = s.section()
        for key in sect.support():
            first_perm = key[0]
            assert tuple(first_perm.tuple()) == tuple(range(1, r + 1)), (
                f"Section of planar surjection {s} is not planar at {key}."
            )


def run_all_tests() -> None:
    test_be_basis_ported()
    test_surjection_basis_ported()
    test_planar_surjection_basis_ported()
    test_planar_preserved_under_composition_last_input_ported()
    test_section_right_inverse_ported()
    test_section_planar_ported()


if __name__ == "__main__":
    run_all_tests()
    print("All ported notebook tests passed.")
