import itertools
from typing import Hashable, Iterable, Iterator, Literal

from comch import BarrattEcclesElement, SurjectionElement, SymmetricGroupElement


## General utility


def sym(s: tuple[int, ...]) -> SymmetricGroupElement:
    """Convert a tuple to a SymmetricGroupElement.

    Args:
        s: A tuple representing the permutation.
    """
    return SymmetricGroupElement(s)


def sb(
    u: tuple[int, ...],
    convention: Literal["Berger-Fresse", "McClure-Smith"] | None = None,
) -> SurjectionElement:
    """Convert a surjection to a SurjectionElement.

    Args:
        u: A tuple representing the surjection.
    """
    return SurjectionElement({u: 1}, convention=convention)


def be(u: tuple[SymmetricGroupElement, ...]) -> BarrattEcclesElement:
    """Convert a surjection to a BarrattEcclesElement.

    Args:
        u: A tuple representing the surjection.
    """
    return BarrattEcclesElement({u: 1})


def rho(sigma: tuple[SymmetricGroupElement, ...], r: int) -> BarrattEcclesElement:
    """Convert a tuple of permutations to a Barratt-Eccles element.

    Args:
        sigma: A tuple of SymmetricGroupElement representing the permutations.
        r: An integer representing the rank.
    """
    id = SymmetricGroupElement(range(1, r + 1))
    if len(sigma) == 0:
        return BarrattEcclesElement({(id,): 1})
    else:
        ret = [id]
        for perm in reversed(sigma):
            ret.append(perm * ret[-1])
        return BarrattEcclesElement({tuple(ret): 1})


## Bases


### Basis for the Barratt--Eccles operad
def be_basis(r: int, d: int) -> Iterator[BarrattEcclesElement]:
    """Generate the basis elements for the Barratt--Eccles operad.

    Args:
        r: The rank of the operad.
        d: The degree of the operad.
    """

    if d < 0 or r < 0:
        raise ValueError("r and d must be non-negative integers")

    permutations = itertools.permutations(range(1, r + 1))
    for values in itertools.product(permutations, repeat=d + 1):
        if all(values[i] != values[i + 1] for i in range(len(values) - 1)):
            yield be(tuple(sym(v) for v in values))


def planar_be_basis(r: int, d: int) -> Iterator[BarrattEcclesElement]:
    """Generate the basis elements for the Barratt--Eccles operad.

    Args:
        r: The rank of the operad.
        d: The degree of the operad.
    """
    permutations = itertools.permutations(range(1, r + 1))
    id = sym(tuple(range(1, r + 1)))
    for values in itertools.product(permutations, repeat=d):
        if (
            all(values[i] != values[i + 1] for i in range(len(values) - 1))
            and values[0] != id
        ):
            yield be((id,) + tuple(sym(v) for v in values))


### Basis for the surjection operad
def surjection_basis(r: int, d: int) -> Iterator[SurjectionElement]:
    """Generate the basis of Surjection(r, d) : surjections from a set of size r + d to a set of size r with no consecutive repeats.

    Args:
        r: Non-negative integer representing the arity.
        d: Non-negative integer representing the degree.
    """

    if d < 0 or r < 0:
        raise ValueError("r and d must be non-negative integers")

    for values in itertools.product(range(1, r + 1), repeat=r + d):
        # Check if the surjection is valid (no consecutive repeats and hits all values)
        seen = set()
        bad = False
        for i in range(len(values) - 1):
            if values[i] == values[i + 1]:
                bad = True
                break
            seen.add(values[i])
        seen.add(values[-1])
        if not bad and seen == set(range(1, r + 1)):
            yield sb(values)


### Planar basis for the surjection operad
def is_planar(l: tuple[int, ...], r: int) -> bool:
    """Check if the first occurrence of each integer in l are in increasing order up to r.

    Args:
        l: A tuple of integers.
        r: An integer representing the maximum value to check.
    """
    first_occurrences = {}
    for i, val in enumerate(l):
        if val not in first_occurrences:
            first_occurrences[val] = i

    first_indices = [
        first_occurrences[i] for i in range(1, r + 1) if i in first_occurrences
    ]
    return all(
        earlier < later for earlier, later in zip(first_indices, first_indices[1:])
    )


def is_planar_surj(s: SurjectionElement) -> bool:
    """Check if a SurjectionElement is planar, meaning that all its summands are planar.

    Args:
        s: A SurjectionElement to check for planarity.
    """
    r = s.arity
    if r is None:
        raise ValueError(
            "SurjectionElement must have a defined arity to check planarity."
        )

    return all(is_planar(key, r) for key in s.keys())


def planar_surjection_basis(r: int, d: int) -> Iterator[SurjectionElement]:
    """Generate the basis of planar Surjection(r, d)"""
    return filter(is_planar_surj, surjection_basis(r, d))


## Decomposition of a surjection
def planarize_surjection_tuple(
    s: tuple[int, ...],
) -> tuple[SymmetricGroupElement, SurjectionElement]:
    """Return the planarization of a surjection tuple s by finding which permutation makes it planar and returning the permuted tuple and that permutation."""
    r = max(s)
    first_occurrences = {}
    for i, val in enumerate(s):
        if val not in first_occurrences:
            first_occurrences[val] = i

    sorted_values = sorted(range(1, r + 1), key=lambda x: first_occurrences[x])
    planar_s = tuple(sorted_values.index(val) + 1 for val in s)
    return SymmetricGroupElement(tuple(sorted_values)), sb(planar_s)


def planarize_surjection(
    s: SurjectionElement,
) -> dict[SymmetricGroupElement, SurjectionElement]:
    """Return the planarization of a SurjectionElement by returning a dictionary mapping permutations to planar SurjectionElements."""
    if len(s) == 0:
        return {}
    r = s.arity
    if r is None:
        raise ValueError("SurjectionElement must have a defined arity to planarize.")

    result = {}
    for surj, coeff in s.items():
        permutation, planarized = planarize_surjection_tuple(surj)
        if permutation not in result:
            result[permutation] = SurjectionElement()
        result[permutation] += coeff * planarized

    return result


## Section of the table reduction morphism
def caesuras(u: tuple[int, ...]) -> list[int]:
    """Return the list of caesuras in a surjection u,
    i.e. the indices that are NOT the last occurrences of an element.
    Indices are 0-based and returned in increasing order.

    Args:
        u: A tuple representing the surjection.
    """
    seen: set[int] = set()
    res: list[int] = []
    for i in range(len(u) - 1, -1, -1):
        if u[i] in seen:
            res.append(i)
        else:
            seen.add(u[i])
    res.reverse()
    return res


def section_tuple(u: tuple[int, ...]) -> BarrattEcclesElement:
    """Return the section of a surjection tuple u as a tuple of tuples."""
    caesura_indices = caesuras(u)
    sections = [[i for i in range(len(u)) if i not in caesura_indices]]
    for d in reversed(caesura_indices):
        # add a new element to section based on the last element
        # drop the element that maps to u[d]
        # add d to the new element
        new_section = sections[-1][:]
        to_remove = None
        for i, v in enumerate(new_section):
            if u[v] == u[d]:
                to_remove = i
                break
        assert to_remove is not None
        new_section.pop(to_remove)
        new_section.append(d)
        new_section.sort()
        sections.append(new_section)
    sections.reverse()
    return be(tuple(sym(tuple(u[i] for i in section)) for section in sections))


def section(u: SurjectionElement) -> BarrattEcclesElement:
    """Return the section of a SurjectionElement as a BarrattEcclesElement."""
    return sum(
        (coeff * section_tuple(surj) for surj, coeff in u.items()),
        BarrattEcclesElement(),
    )


## Some computation of the comodule structure
def comod_surj(
    u: SurjectionElement,
) -> list[tuple[SurjectionElement, SurjectionElement]]:
    """Compute the E-comodule of a SurjectionElement u, then apply the table reduction morphism."""
    r = u.arity
    assert (
        r is not None
    ), "SurjectionElement must have a defined arity to compute diagonal."
    id = SymmetricGroupElement(tuple(range(1, r + 1)))

    def helper(
        v: SurjectionElement,
    ) -> list[tuple[tuple[SymmetricGroupElement, ...], SurjectionElement]]:
        if len(v) == 0:
            return []
        result: list[tuple[tuple[SymmetricGroupElement, ...], SurjectionElement]] = [
            (tuple(), v)
        ]
        # Calcul de la boundary et planarisation
        boundary = v.boundary()
        planarized = planarize_surjection(boundary)
        for perm, s in planarized.items():
            if perm == id:
                continue  # Donnera zéro
            # Appel récursif
            for sub_perm, sub_s in helper(s):
                # Concatène les permutations
                result.append(((perm,) + sub_perm, sub_s))
        return result

    return [(rho(perms, r).table_reduction(), s) for perms, s in helper(u)]
