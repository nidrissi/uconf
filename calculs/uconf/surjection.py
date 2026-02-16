import itertools
from itertools import combinations, combinations_with_replacement, pairwise
from typing import ClassVar, Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .barratt_eccles import BarrattEccles

from sage.all import *  # pyright: ignore[reportWildcardImportFromLibrary]


class Surjection(CombinatorialFreeModule):
    name: ClassVar[str] = "S"

    def __init__(self, n: int, base_ring=QQ):
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
        self.boundary = self.module_morphism(
            on_basis=self._boundary_on_basis, codomain=self
        )

    def _element_constructor_(self, x: "Surjection.Element | dict | tuple | list"):
        """
        Intercepts element creation to enforce types.
        x can be:
          - A basis key (tuple of integers)
          - A linear combination (dictionary)
        """
        # Case 1: x is a Dictionary (Linear Combination)
        if isinstance(x, dict):
            # Validate keys before passing to super
            clean_dict = {}
            for key, coeff in x.items():
                clean_key = self._validate_basis_key(key)
                if clean_key is None:
                    # Skip zero terms
                    continue
                clean_dict[clean_key] = coeff
            return super()._element_constructor_(clean_dict)

        # Case 2: x is a Basis Key (Tuple)
        # We try to treat it as a single basis element
        if isinstance(x, (tuple, list)):
            # Check if it looks like a tuple of integers vs a list of terms
            # Simple heuristic: is the first element an integer?
            try:
                clean_key = self._validate_basis_key(x)
                # Check for consecutive identical integers
                if clean_key is None:
                    return self.zero()
                else:
                    # Return the monomial 1 * basis_element
                    return self.term(clean_key)
            except (ValueError, TypeError) as e:
                raise TypeError(
                    f"Item is not a valid element of {self}. Got {x} ({type(x)})"
                ) from e

        raise TypeError(
            f"Input must be a dictionary (for linear combinations) or a tuple/list (for basis elements). Got {x} ({type(x)})."
        )

    def _validate_basis_key(self, basis_tuple: "tuple | list") -> tuple | None:
        """
        Strictly checks that the input is a list of integers which contain all elements from 1 to self.arity().
        """
        if not isinstance(basis_tuple, (tuple, list)):
            raise TypeError(f"Basis key must be a tuple, got {type(basis_tuple)}")

        for p in basis_tuple:
            if not isinstance(p, int):
                raise TypeError(
                    f"Basis key must be a tuple of integers. Got {p} ({type(p)})."
                )

        # Non-surjective maps yield zero
        if set(basis_tuple) != set(range(1, self.arity() + 1)):
            return None

        # Check for consecutive identical integers
        if len(basis_tuple) > 0:
            for i in range(len(basis_tuple) - 1):
                # Consecutive identical integers yield zero
                if basis_tuple[i] == basis_tuple[i + 1]:
                    return None
        return tuple(basis_tuple)

    def arity(self):
        return self._arity

    @staticmethod
    def unit() -> "Surjection.Element":
        return Surjection(1)((1,))

    def basis_it(self, d: int) -> Iterator[Surjection.Element]:
        assert d >= 0, "d must be a non-negative integer, got d={d}."
        r = self.arity()
        for values in itertools.product(range(1, r + 1), repeat=r + d):
            # Check if the surjection is valid (no consecutive repeats and hits all values)
            res = self(values)
            if res != self.zero():
                yield res

    def planar_basis_it(self, d: int) -> Iterator[Surjection.Element]:
        return filter(lambda u: u.is_planar(), self.basis_it(d))

    def degree_on_basis(self, basis_element: tuple) -> int:
        return len(basis_element) - self.arity()

    def _boundary_on_basis(self, basis_element: tuple) -> "Surjection.Element":
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
        result = 0
        for i, j in combinations(range(self.arity()), 2):
            seq = [x for x in basis_element if x == i or x == j]
            complexity = len([k for k, l in pairwise(seq) if k != l])
            result = max(result, complexity)
        return result

    @staticmethod
    def compose(
        x: Surjection.Element, input: int, y: Surjection.Element
    ) -> Surjection.Element:
        """
        Composes x and y by inserting y into the i-th input of x.
        """
        m = x.arity()
        n = y.arity()
        assert 1 <= input <= m, f"Index i must be between 1 and {m}. Got {input}."
        target = Surjection(m + n - 1)

        def _compose_basis_tuple(x_tuple: tuple[int, ...], y_tuple: tuple[int, ...]):
            def bf_sign(
                p1: tuple[int, ...],
                k1: tuple[int, ...],
                p2: tuple[int, ...],
                k2: tuple[int],
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

            positions = [idx for idx, i in enumerate(x_tuple) if i == input]
            for p in combinations_with_replacement(
                range(len(y_tuple)), len(positions) - 1
            ):
                p = (0,) + p + (len(y_tuple) - 1,)
                split = []
                for a, b in pairwise(p):
                    split.append(tuple(y_tuple[a : b + 1]))
                to_insert = (tuple(j + input - 1 for j in part) for part in split)
                new_k = []
                for j in x_tuple:
                    if j < input:
                        new_k.append(j)
                    elif j == input:
                        new_k += next(to_insert)
                    else:
                        new_k.append(j + n - 1)
                new_k = tuple(new_k)
                yield new_k, bf_sign(p, x_tuple, y_tuple, new_k)

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
        def boundary(self):
            return self.parent().boundary(self)

        def arity(self):
            return self.parent().arity()

        def complexity(self) -> int:
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
            """
            Checks if the element is planar by verifying that, for each tuple in
            self.support(), the first occurrences of each integer appear in
            increasing order. Returns True if all tuples satisfy this condition.
            """
            r = self.arity()

            def _planar(l: tuple[int, ...]) -> bool:
                """Check if the first occurrence of each integer in l are in increasing order."""
                first_occurrences = {}
                for i, val in enumerate(l):
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
            raise NotImplementedError("Section is not implemented yet")
