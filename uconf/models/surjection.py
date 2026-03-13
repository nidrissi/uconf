"""Surjection operad model on nondegenerate surjective words."""

from __future__ import annotations

import itertools
from itertools import combinations, combinations_with_replacement, pairwise
from typing import TYPE_CHECKING, ClassVar, Iterator

if TYPE_CHECKING:
    from uconf.models.barratt_eccles import BarrattEccles

from sage.all import (
    Integer,
    QQ,
    CombinatorialFreeModule,
    Family,
    GradedModulesWithBasis,
    SymmetricGroup,
    SymmetricGroupAlgebra,
    cached_method,
    tensor,
)


class Surjection(CombinatorialFreeModule):
    """Surjection operad component in fixed arity.

    Basis elements are tuples ``u`` with values in ``{1, ..., n}`` that are
    surjective and have no consecutive equal entries. Degenerate inputs map to
    zero, while malformed inputs raise.
    """

    name: ClassVar[str] = "S"

    def __init__(self, n: int, base_ring=QQ):
        """Initialize ``S_n`` over ``base_ring``."""

        assert n >= 0, f"Arity must be non-negative. Got {n}."
        name = f"{self.name}{n}"
        super().__init__(
            base_ring,
            tuple,
            prefix=name,
            category=GradedModulesWithBasis(base_ring),
        )
        self.rename(name)
        self._arity: int = n
        self._symmetric_group = SymmetricGroup(n)
        self._symmetric_group_algebra = SymmetricGroupAlgebra(base_ring, n)
        self.boundary = self.module_morphism(
            on_basis=self._boundary_on_basis, codomain=self
        )
        self.planarize = self.module_morphism(
            on_basis=self._planarize_on_basis,
            codomain=tensor([self, SymmetricGroupAlgebra(base_ring, n)]),
        )

    def _element_constructor_(self, x: "Surjection.Element | dict | tuple | list"):
        """Build elements from basis tuples or sparse dictionaries.

        Degenerate basis keys, namely non-surjective tuples or tuples with
        consecutive equal entries, map to zero. Invalid keys raise.
        """
        if isinstance(x, dict):
            clean_dict = {}
            for key, coeff in x.items():
                clean_key = self._validate_basis_key(key)
                if clean_key is None:
                    continue
                clean_dict[clean_key] = coeff
            return self.sum_of_terms(clean_dict.items())

        if isinstance(x, (tuple, list)):
            clean_key = self._validate_basis_key(x)
            if clean_key is None:
                return self.zero()
            return self.term(clean_key)

        raise TypeError(
            f"Input must be a dictionary (for linear combinations) or a tuple/list (for basis elements). Got {x} ({type(x)})."
        )

    def _validate_basis_key(self, basis_tuple: "tuple | list") -> tuple | None:
        """Validate a basis tuple.

        Returns ``None`` for degenerate surjections and raises on malformed
        inputs.
        """
        if not isinstance(basis_tuple, (tuple, list)):
            raise TypeError(
                f"Basis key must be a tuple or list, got {type(basis_tuple)}"
            )

        clean_tuple = tuple(basis_tuple)

        for p in clean_tuple:
            if not isinstance(p, (int, Integer)):
                raise TypeError(
                    f"Basis key must be a tuple of integers. Got {p} ({type(p)})."
                )
            if p < 1 or p > self.arity():
                raise ValueError(
                    f"Surjection entries must lie in {{1, ..., {self.arity()}}}. Got {p}."
                )

        if len(clean_tuple) > 0:
            for i in range(len(clean_tuple) - 1):
                if clean_tuple[i] == clean_tuple[i + 1]:
                    return None

        if set(clean_tuple) != set(range(1, self.arity() + 1)):
            return None

        return clean_tuple

    def arity(self):
        """Return the fixed arity of this operad component."""

        return self._arity

    @staticmethod
    def unit() -> "Surjection.Element":
        """Return the operadic unit in arity ``1``."""

        return Surjection(1)((1,))

    def basis_it(self, d: int) -> Iterator[Surjection.Element]:
        """Iterate over basis elements in degree ``d``."""

        assert d >= 0, "d must be a non-negative integer, got d={d}."
        r = self.arity()
        for values in itertools.product(range(1, r + 1), repeat=r + d):
            # Check if the surjection is valid (no consecutive repeats and hits all values)
            res = self(values)
            if res != self.zero():
                yield res

    def planar_basis_it(self, d: int) -> Iterator[Surjection.Element]:
        """Iterate over planar basis elements in degree ``d``."""

        return filter(lambda u: u.is_planar(), self.basis_it(d))

    @cached_method
    def graded_basis(self, d: int) -> Family:
        """Return the ``Family`` of all basis elements in degree ``d``."""
        return Family(tuple(self.basis_it(d)))

    @cached_method
    def graded_planar_basis(self, d: int) -> Family:
        """Return the ``Family`` of planar basis elements in degree ``d``."""
        return Family(tuple(self.planar_basis_it(d)))

    def _planarize_on_basis(self, basis_element: tuple):
        """Split into planar representative and symmetric-group factor."""

        n = self.arity()
        first_occurrence = []
        seen = set()
        for val in basis_element:
            if val not in seen:
                seen.add(val)
                first_occurrence.append(val)
            if len(first_occurrence) == n:
                break
            if len(first_occurrence) == n:
                break
        # first_occurrence is a permutation of (1,...,n)
        sigma = self._symmetric_group(first_occurrence)
        sigma_inv = sigma.inverse()
        # Permute the basis element back into planar form
        planar_basis = tuple(sigma_inv(p) for p in basis_element)
        planar_element = self.term(planar_basis)
        sigma_module = self._symmetric_group_algebra
        return planar_element.tensor(sigma_module(sigma))

    def degree_on_basis(self, basis_element: tuple) -> int:
        """Return homological degree of one basis surjection."""

        return len(basis_element) - self.arity()

    def _boundary_on_basis(self, basis_element: tuple) -> "Surjection.Element":
        """Compute the differential on a basis surjection."""

        # determining the signs of the summands
        signs = {}
        alternating_sign = 1
        for idx, i in enumerate(basis_element):
            if i in basis_element[idx + 1 :]:
                signs[idx] = alternating_sign
                alternating_sign *= -1
            elif i in basis_element[:idx]:
                occurs = (pos for pos, j in enumerate(basis_element[:idx]) if i == j)
                signs[idx] = signs[max(occurs)] * (-1)
            else:
                signs[idx] = 0

        def term_generator():
            for idx in range(0, len(basis_element)):
                bdry_summand = basis_element[:idx] + basis_element[idx + 1 :]
                if (
                    basis_element[idx] in bdry_summand
                    and self._validate_basis_key(bdry_summand) is not None
                ):
                    yield (bdry_summand, signs[idx])

        return self.sum_of_terms(term_generator())

    def _complexity_on_basis(self, basis_element: tuple) -> int:
        """Return pairwise complexity of one basis surjection."""

        result = 0
        for i, j in combinations(range(1, self.arity() + 1), 2):
            seq = [x for x in basis_element if x == i or x == j]
            complexity = len([a for a, b in pairwise(seq) if a != b])
            result = max(result, complexity)
        return result

    @staticmethod
    def compose(
        x: Surjection.Element, i: int, y: Surjection.Element
    ) -> Surjection.Element:
        """Compose surjections by Berger--Fresse insertion at input ``i``."""
        if x.parent().base_ring() != y.parent().base_ring():
            raise TypeError("Both elements must have the same base ring.")
        m = x.arity()
        n = y.arity()
        assert 1 <= i <= m, f"Index i must be between 1 and {m}. Got {i}."
        target = Surjection(m + n - 1, base_ring=x.parent().base_ring())

        def _compose_basis_tuple(x_tuple: tuple[int, ...], y_tuple: tuple[int, ...]):
            def bf_sign(
                p1: tuple[int, ...],
                k1: tuple[int, ...],
                p2: tuple[int, ...],
                k2: tuple[int, ...],
            ) -> int:
                """Sign associated to the Berger-Fresse composition."""

                def caesuras(k: tuple[int, ...]):
                    """Returns the caesuras of a basis element."""
                    caesuras: list[int] = []
                    for idx, i in enumerate(k):
                        if i in k[idx + 1 :]:
                            caesuras.append(idx)
                    return caesuras

                def weights(cae: list[int], p: tuple[int, ...]):
                    """Returns the weights of the splitting knowing the caesuras."""
                    weights: list[int] = []
                    for i, j in pairwise(p):
                        closed_open = len([e for e in cae if i <= e < j])
                        weights.append(closed_open)
                    return [value % 2 for value in weights]

                p1 = (0,) + p1 + (len(k1) - 1,)
                cae1 = caesuras(k1)
                w1 = weights(cae1, p1)
                cae2 = caesuras(k2)
                w2 = weights(cae2, p2)
                sign_exp = 0
                for idx, w in enumerate(w2):
                    if w:
                        sign_exp += sum(w1[idx + 1 :]) % 2
                return (-1) ** sign_exp

            positions = [idx for idx, j in enumerate(x_tuple) if j == i]
            for p in combinations_with_replacement(
                range(len(y_tuple)), len(positions) - 1
            ):
                p = (0,) + p + (len(y_tuple) - 1,)
                split = []
                for a, b in pairwise(p):
                    split.append(tuple(y_tuple[a : b + 1]))
                to_insert = (tuple(j + i - 1 for j in part) for part in split)
                new_k = []
                for j in x_tuple:
                    if j < i:
                        new_k.append(j)
                    elif j == i:
                        new_k += next(to_insert)
                    else:
                        new_k.append(j + n - 1)
                new_k = tuple(new_k)
                yield new_k, bf_sign(tuple(positions), x_tuple, p, y_tuple)

        def term_generator():
            for x_tuple, x_coeff in x:
                for y_tuple, y_coeff in y:
                    for new_k, sign in _compose_basis_tuple(x_tuple, y_tuple):
                        # Validate the new basis key before yielding
                        if target._validate_basis_key(new_k) is None:
                            continue
                        yield (new_k, sign * x_coeff * y_coeff)

        return target.sum_of_terms(term_generator())

    @staticmethod
    def _caesuras(u: tuple[int, ...]) -> list[int]:
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

    class Element(CombinatorialFreeModule.Element):
        """Elements of a fixed-arity surjection component."""

        def boundary(self) -> Surjection.Element:
            """Apply the differential."""

            return self.parent().boundary(self)

        def arity(self) -> int:
            """Return the arity of this element."""

            return self.parent().arity()

        def planarize(self):
            """Project to planar representative tensored with group element."""

            return self.parent().planarize(self)

        def complexity(self) -> int:
            """Return the maximum pairwise complexity on basis support."""

            return max(
                (self.parent()._complexity_on_basis(basis) for basis in self.support()),
                default=0,
            )

        def permute(self, sigma) -> Surjection.Element:
            """
            Permutes the basis elements of self by precomposing with sigma.
            """
            if isinstance(sigma, (list, tuple)):
                sigma = self.parent()._symmetric_group(sigma)
            elif not (
                hasattr(sigma, "parent")
                and sigma.parent() == self.parent()._symmetric_group
            ):
                raise TypeError(
                    f"Permutation must be a list, tuple, or element of S_{self.parent().arity()}. Got {sigma} ({type(sigma)})."
                )

            def permuted_term_generator():
                for u, coeff in self:
                    # Precompose each permutation in the basis tuple with sigma
                    permuted_basis = tuple(sigma(i) for i in u)
                    yield (permuted_basis, coeff)

            return self.parent().sum_of_terms(permuted_term_generator())

        def is_planar(self) -> bool:
            """Return whether each supported basis term satisfies planarity."""
            r = self.arity()

            def _planar(input_list: tuple[int, ...]) -> bool:
                """Check if the first occurrence of each integer in input_list are in increasing order."""
                first_occurrences = {}
                for i, val in enumerate(input_list):
                    if val not in first_occurrences:
                        first_occurrences[val] = i

                first_indices = [
                    first_occurrences[i]
                    for i in range(1, r + 1)
                    if i in first_occurrences
                ]
                return all(
                    earlier < later
                    for earlier, later in zip(first_indices, first_indices[1:])
                )

            return all(_planar(key) for key in self.support())

        def section(self) -> BarrattEccles.Element:
            """Placeholder, replaced at import time by :mod:`uconf.__init__`."""

            raise NotImplementedError("Section is not implemented yet")
