from itertools import combinations, pairwise
from sage.all import *  # pyright: ignore[reportWildcardImportFromLibrary]


class Surjection(CombinatorialFreeModule):
    def __init__(self, n, base_ring=QQ):
        super().__init__(
            base_ring,
            tuple,
            prefix=f"S{n}",
            category=GradedModulesWithBasis(base_ring),
        )
        self.arity = n
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
                    f"Item is not a valid element of S_{self.arity}. Got {x} ({type(x)})"
                ) from e

        raise TypeError(
            f"Input must be a dictionary (for linear combinations) or a tuple/list (for basis elements). Got {x} ({type(x)})."
        )

    def _validate_basis_key(self, basis_tuple: "tuple | list") -> tuple | None:
        """
        Strictly checks that the input is a list of integers which contain all elements from 1 to self.arity.
        """
        if not isinstance(basis_tuple, (tuple, list)):
            raise TypeError(f"Basis key must be a tuple, got {type(basis_tuple)}")

        for p in basis_tuple:
            if not isinstance(p, int):
                raise TypeError(
                    f"Basis key must be a tuple of integers. Got {p} ({type(p)})."
                )

        if set(basis_tuple) != set(range(1, self.arity + 1)):
            raise ValueError(
                f"Basis key must contain all integers from 1 to {self.arity} exactly once. "
                f"Got {basis_tuple}."
            )

        # Check for consecutive identical integers
        if len(basis_tuple) > 0:
            for i in range(len(basis_tuple) - 1):
                if basis_tuple[i] == basis_tuple[i + 1]:
                    # Consecutive identical integers yield zero
                    return None
        return tuple(basis_tuple)

    def degree_on_basis(self, basis_element: tuple) -> int:
        return len(basis_element) - self.arity

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
        for i, j in combinations(range(self.arity), 2):
            seq = [x for x in basis_element if x == i or x == j]
            complexity = len([k for k, l in pairwise(seq) if k != l])
            result = max(result, complexity)
        return result

    class Element(CombinatorialFreeModule.Element):
        def boundary(self):
            return self.parent().boundary(self)

        def arity(self):
            return self.parent().arity

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
                    f"Permutation must be a list, tuple, or element of S_{self.parent().arity}. Got {sigma} ({type(sigma)})."
                )

            def permuted_term_generator():
                for u, coeff in self:
                    # Precompose each permutation in the basis tuple with sigma
                    permuted_basis = tuple(sigma(i) for i in u)
                    yield (permuted_basis, coeff)

            return self.parent().sum_of_terms(permuted_term_generator())
