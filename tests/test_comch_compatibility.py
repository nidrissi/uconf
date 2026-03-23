import pytest

from sage.all import QQ

from uconf import BarrattEccles, Surjection

pytest.importorskip("comch")
from comch.barratt_eccles.barratt_eccles import BarrattEcclesElement
from comch.surjection.surjection import SurjectionElement


def _canonical_surjection_key(key) -> tuple[int, ...]:
    return tuple(int(v) for v in key)


def _canonical_permutation_key(perm) -> tuple[int, ...]:
    if hasattr(perm, "tuple"):
        return tuple(int(v) for v in perm.tuple())
    return tuple(int(v) for v in perm)


def _canonical_be_key(key) -> tuple[tuple[int, ...], ...]:
    return tuple(_canonical_permutation_key(perm) for perm in key)


def _uconf_surjection_dict(element) -> dict[tuple[int, ...], int]:
    return {_canonical_surjection_key(key): int(coeff) for key, coeff in element if int(coeff) != 0}


def _comch_surjection_dict(element) -> dict[tuple[int, ...], int]:
    return {
        _canonical_surjection_key(key): int(coeff)
        for key, coeff in element.items()
        if int(coeff) != 0
    }


def _uconf_be_dict(element) -> dict[tuple[tuple[int, ...], ...], int]:
    return {_canonical_be_key(key): int(coeff) for key, coeff in element if int(coeff) != 0}


def _comch_be_dict(element) -> dict[tuple[tuple[int, ...], ...], int]:
    return {_canonical_be_key(key): int(coeff) for key, coeff in element.items() if int(coeff) != 0}


def _uconf_surjection_to_comch(element) -> SurjectionElement:
    return SurjectionElement(
        _uconf_surjection_dict(element),
        convention="Berger-Fresse",
    )


def _uconf_be_to_comch(element) -> BarrattEcclesElement:
    return BarrattEcclesElement(_uconf_be_dict(element))


def _uconf_surjection_from_basis(basis: tuple[int, ...]):
    return Surjection(max(basis), QQ)(basis)


def _comch_surjection_from_basis(basis: tuple[int, ...]) -> SurjectionElement:
    return SurjectionElement({basis: 1}, convention="Berger-Fresse")


def _uconf_be_from_basis(basis: tuple[tuple[int, ...], ...]):
    one_line_basis = tuple(list(perm) for perm in basis)
    return BarrattEccles(len(basis[0]), QQ)(one_line_basis)


def _comch_be_from_basis(basis: tuple[tuple[int, ...], ...]) -> BarrattEcclesElement:
    return BarrattEcclesElement({basis: 1})


@pytest.mark.parametrize("r", range(1, 5))
@pytest.mark.parametrize("d", range(0, 3))
def test_surjection_boundary_matches_comch(r: int, d: int) -> None:
    for basis_element in Surjection(r, QQ).basis_iter(d):
        uconf_boundary = basis_element.boundary()
        comch_boundary = _uconf_surjection_to_comch(basis_element).boundary()
        assert _uconf_surjection_dict(uconf_boundary) == _comch_surjection_dict(comch_boundary)


@pytest.mark.parametrize(
    "x_basis,input_pos,y_basis",
    [
        ((1, 2), 1, (1, 2)),
        ((1, 2), 2, (1, 2)),
        ((1, 2, 1), 1, (1, 2)),
        ((1, 2, 1), 2, (1, 2, 1)),
        ((1, 2, 3, 1), 3, (1, 2, 1)),
    ],
)
def test_surjection_compose_matches_comch(
    x_basis: tuple[int, ...], input_pos: int, y_basis: tuple[int, ...]
) -> None:
    uconf_x = _uconf_surjection_from_basis(x_basis)
    uconf_y = _uconf_surjection_from_basis(y_basis)
    comch_x = _comch_surjection_from_basis(x_basis)
    comch_y = _comch_surjection_from_basis(y_basis)

    uconf_composed = Surjection.compose(uconf_x, input_pos, uconf_y)
    comch_composed = comch_x.compose(comch_y, input_pos)
    assert _uconf_surjection_dict(uconf_composed) == _comch_surjection_dict(comch_composed)


@pytest.mark.parametrize(
    "basis",
    [
        (1, 2),
        (1, 2, 1),
        (1, 2, 3, 1),
        (1, 3, 1, 2),
    ],
)
def test_surjection_complexity_matches_comch(basis: tuple[int, ...]) -> None:
    uconf_element = _uconf_surjection_from_basis(basis)
    comch_element = _comch_surjection_from_basis(basis)
    assert uconf_element.complexity() == int(comch_element.complexity)


def test_barratt_eccles_boundary_matches_comch() -> None:
    for r in range(1, 4):
        for d in range(0, 3):
            for basis_element in BarrattEccles(r, QQ).basis_iter(d):
                uconf_boundary = basis_element.boundary()
                comch_boundary = _uconf_be_to_comch(basis_element).boundary()
                assert _uconf_be_dict(uconf_boundary) == _comch_be_dict(comch_boundary)


@pytest.mark.parametrize(
    "x_basis,input_pos,y_basis",
    [
        (((1, 2),), 1, ((1, 2),)),
        (((1, 2), (2, 1)), 1, ((1, 2),)),
        (((1, 2),), 1, ((1, 2), (2, 1))),
        (((1, 2, 3), (2, 1, 3)), 2, ((1, 2),)),
    ],
)
def test_barratt_eccles_compose_matches_comch(
    x_basis: tuple[tuple[int, ...], ...],
    input_pos: int,
    y_basis: tuple[tuple[int, ...], ...],
) -> None:
    uconf_x = _uconf_be_from_basis(x_basis)
    uconf_y = _uconf_be_from_basis(y_basis)
    comch_x = _comch_be_from_basis(x_basis)
    comch_y = _comch_be_from_basis(y_basis)

    uconf_composed = BarrattEccles.compose(uconf_x, input_pos, uconf_y)
    comch_composed = comch_x.compose(comch_y, input_pos)
    assert _uconf_be_dict(uconf_composed) == _comch_be_dict(comch_composed)


def test_table_reduction_matches_comch() -> None:
    for r in range(1, 4):
        for d in range(0, 3):
            for basis_element in BarrattEccles(r, QQ).basis_iter(d):
                uconf_tr = basis_element.table_reduction()
                comch_tr = _uconf_be_to_comch(basis_element).table_reduction()
                assert _uconf_surjection_dict(uconf_tr) == _comch_surjection_dict(
                    SurjectionElement(
                        dict(comch_tr.items()),
                        convention="Berger-Fresse",
                    )
                )
