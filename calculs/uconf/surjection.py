"""Surjection operad model on nondegenerate surjective words."""

import itertools
from functools import reduce
from itertools import combinations, combinations_with_replacement, pairwise, product
from typing import TYPE_CHECKING, ClassVar, Iterator

if TYPE_CHECKING:
    from .barratt_eccles import BarrattEccles
    from .simplicial import CosimplicialCochains, SimplicialChains

from sage.all import (
    QQ,
    CombinatorialFreeModule,
    GradedModulesWithBasis,
    SymmetricGroup,
    SymmetricGroupAlgebra,
    tensor,
)


class Surjection(CombinatorialFreeModule):
    """Surjection operad component in fixed arity.

    Basis elements are tuples ``u`` with values in ``{1, ..., n}`` that are
    surjective and have no consecutive equal entries.
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
        """Build elements from basis tuples or sparse dictionaries."""
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
        """Validate that an input tuple defines a nondegenerate surjection."""
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
        for i, j in combinations(range(self.arity()), 2):
            seq = [x for x in basis_element if x == i or x == j]
            complexity = len([k for k, l in pairwise(seq) if k != l])
            result = max(result, complexity)
        return result

    @staticmethod
    def compose(
        x: Surjection.Element, input: int, y: Surjection.Element
    ) -> Surjection.Element:
        """Compose surjections by Berger--Fresse insertion at input ``i``."""
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
    def act(
        surj: "Surjection.Element",
        chain: "SimplicialChains.Element",
    ) -> "SimplicialChains.Element":
        r"""Action of the surjection operad on normalised simplicial chains.

        Given a surjection `u \in \mathcal{X}(r)_d` and a chain
        `x \in C_n(\Delta)`, the action produces an element
        `\theta_u(x) \in (C^{\otimes r})_{n-d}`.

        **Algorithm** (Berger--Fresse convention, [BF04] §2):

        1. Compute the iterated Alexander--Whitney diagonal
           `\Delta^{r+d-1}(x)`, producing terms in `C^{\otimes(r+d)}`.
        2. For each basis surjection `u = (u_1, \ldots, u_{r+d})` and each
           term `(c_1, \ldots, c_{r+d})` in the diagonal, join the simplices
           `c_j` for which `u_j = i` (for `i = 1, \ldots, r`).  The join of
           adjacent simplices is the concatenation with overlapping endpoints.
        3. Discard degenerate results and accumulate with the BF sign.

        The BF sign consists of two parts:

        * The *ordering (Koszul) sign*: the sign of permuting the factors
          `(c_1, \ldots, c_{r+d})` to group them by value of `u`, respecting
          degrees.
        * The *action sign*: an extra BF-specific sign arising from
          the insertion of degree-1 "operators" between equal consecutive
          values of the sorted surjection, weighted by the accumulated
          degrees to the left.

        Parameters
        ----------
        surj : Surjection.Element
            A homogeneous element of the surjection operad.
        chain : SimplicialChains.Element
            A chain in ``SimplicialChains(1)``.

        Returns
        -------
        SimplicialChains.Element
            The action `\theta_u(x)`, in ``SimplicialChains(r)``.
        """
        from .simplicial import SimplicialChains

        # --- input checks ---
        if not surj or not chain:
            r = surj.arity() if surj else 1
            return SimplicialChains(r=r).zero()

        r = surj.arity()
        surj_support = list(surj.support())
        # Check homogeneity of degree
        degrees = {surj.parent().degree_on_basis(k) for k in surj_support}
        assert len(degrees) == 1, "Surjection must be homogeneous in degree."
        d = degrees.pop()

        target = SimplicialChains(r=r)
        times = r + d - 1  # AW diagonal produces (times+1) = r+d factors

        # pre-compute the iterated diagonal
        pre_diag = chain.iterated_diagonal(times=times, coord=1)

        def _compute_bf_sign(surj_tuple, simplex_factors):
            """Compute the Berger-Fresse sign for a surjection acting on simplex factors.

            Follows [BF04] §2.  The sign has two parts:

            (a) **ordering sign**: Koszul sign of the permutation that sorts the
                factors ``simplex_factors`` by the value ``surj_tuple[j]``,
                stably (preserving original order within each group).

            (b) **action (insertion) sign**: for the sorted sequence, whenever
                two consecutive factors belong to the same group (same value of
                ``surj_tuple``), an implicit degree-1 operator is inserted
                between them.  The sign contribution is ``(-1)`` raised to the
                sum of all degrees to the left of that insertion point.
            """
            weights = [len(s) - 1 for s in simplex_factors]  # dim of each simplex

            # (a) ordering sign --
            # The ordering permutation sends position idx -> sorted position.
            # We need the sign of this permutation acting on graded objects.
            indexed = list(enumerate(surj_tuple))
            # Stable sort by surjection value
            sorted_indexed = sorted(indexed, key=lambda pair: pair[1])
            # The ordering permutation in one-line notation:
            inv_ordering = [pair[0] for pair in sorted_indexed]
            # ordering_perm[i] = position in the original list of the i-th element
            #   in sorted order.
            # Sign = Koszul sign of this permutation with given weights.
            ordering_sign_exp = 0
            ordering_perm = [0] * len(inv_ordering)
            for new_pos, old_pos in enumerate(inv_ordering):
                ordering_perm[old_pos] = new_pos
            # Koszul sign: count (weighted) inversions
            for i in range(len(ordering_perm)):
                for j in range(i + 1, len(ordering_perm)):
                    if ordering_perm[i] > ordering_perm[j]:
                        ordering_sign_exp += weights[i] * weights[j]

            # (b) action sign --
            # After reordering, the surjection values are sorted:
            # [1,..,1, 2,..,2, ... r,..,r].
            # Weights in sorted order:
            sorted_weights = [weights[i] for i in inv_ordering]
            sorted_surj = [surj_tuple[i] for i in inv_ordering]

            action_sign_exp = 0
            for idx in range(len(sorted_surj) - 1):
                if sorted_surj[idx] == sorted_surj[idx + 1]:
                    # Insert a degree-1 operator here; sign = (-1)^(sum of weights to the left)
                    action_sign_exp += sum(sorted_weights[: idx + 1])

            total_sign_exp = (ordering_sign_exp + action_sign_exp) % 2
            return (-1) ** total_sign_exp

        def _join_simplices(simplex_list):
            """Join a list of simplices by concatenation, then drop degenerate terms.

            This follows the simplicial join used in ComCH/BF-style formulas:
            concatenate the vertex tuples and keep the result only if it is
            nondegenerate (strictly increasing).
            """
            if not simplex_list:
                return None
            result = reduce(lambda x, y: x + y, simplex_list)
            # Check non-degeneracy (strictly increasing)
            if any(a >= b for a, b in pairwise(result)):
                return None
            return result

        def term_generator():
            for surj_tuple, surj_coeff in surj:
                for diag_key, diag_coeff in pre_diag:
                    # diag_key is a tuple of r+d simplex-tuples
                    # Join by surjection grouping
                    new_factors = []
                    zero_term = False
                    for i in range(1, r + 1):
                        to_join = [
                            diag_key[idx]
                            for idx in range(len(surj_tuple))
                            if surj_tuple[idx] == i
                        ]
                        joined = _join_simplices(to_join)
                        if joined is None:
                            zero_term = True
                            break
                        new_factors.append(joined)
                    if zero_term:
                        continue
                    new_key = tuple(new_factors)
                    validated = SimplicialChains._validate_basis_key(new_key)
                    if validated is None:
                        continue
                    sign = _compute_bf_sign(surj_tuple, diag_key)
                    yield (validated, sign * surj_coeff * diag_coeff)

        out = target.sum_of_terms(term_generator())
        # Touch native tensor conversion (Sage tensor product parent) so this
        # action stays compatible with tensor-native workflows.
        _ = out.to_native_tensor()
        return out

    @staticmethod
    def coact(
        surj: "Surjection.Element",
        cochains: "tuple[CosimplicialCochains.Element, ...]",
    ) -> "CosimplicialCochains.Element":
        r"""Dual action: cochains side.

        Given `u \in \mathcal{X}(r)_d` and cochains
        `f_1, \ldots, f_r \in C^*(\Delta^N)`, the cochain action is

        .. math::
            \mu_u(f_1 \otimes \cdots \otimes f_r)(x)
            = (f_1 \otimes \cdots \otimes f_r)(\theta_u(x))

        for every chain `x`.  This returns the resulting cochain of degree
        `|f_1| + \cdots + |f_r| + d`.

        Parameters
        ----------
        surj : Surjection.Element
            Homogeneous surjection element.
        cochains : tuple of CosimplicialCochains.Element
            One cochain per arity slot.

        Returns
        -------
        CosimplicialCochains.Element
            The resulting cochain on `\Delta^N`.
        """
        from .simplicial import CosimplicialCochains, SimplicialChains

        r = surj.arity()
        assert len(cochains) == r, f"Expected {r} cochains, got {len(cochains)}."
        N = cochains[0].parent().simplex_dim()
        target = CosimplicialCochains(N=N, r=1)

        # For each chain basis element x of Delta^N, compute (f1⊗…⊗fr)(θ_u(x))
        result_dict: dict[tuple, int] = {}
        # The total cochain degree
        cochain_degrees = [c.degree() for c in cochains]
        total_deg = sum(cochain_degrees)

        # Iterate over basis chains in the appropriate degree
        surj_deg = list({surj.parent().degree_on_basis(k) for k in surj.support()})
        assert len(surj_deg) == 1
        d = surj_deg[0]
        chain_deg = total_deg + d

        # Iterate over all (chain_deg)-dimensional simplices of Delta^N
        for simplex_tuple in combinations(range(N + 1), chain_deg + 1):
            chain_parent = SimplicialChains(r=1)
            x = chain_parent((simplex_tuple,))
            theta = Surjection.act(surj, x)
            # Evaluate f1⊗…⊗fr on theta
            value = 0
            for basis_key, coeff in theta:
                # basis_key = (s_1, ..., s_r)
                contrib = coeff
                for slot in range(r):
                    # f_{slot+1} evaluated on s_{slot+1}
                    f = cochains[slot]
                    simplex_key = (basis_key[slot],)
                    matched = False
                    for f_key, f_coeff in f:
                        if f_key == simplex_key:
                            contrib *= f_coeff
                            matched = True
                            break
                    if not matched:
                        contrib = 0
                        break
                value += contrib
            if value != 0:
                result_dict[(simplex_tuple,)] = (
                    result_dict.get((simplex_tuple,), 0) + value
                )

        # Clean zeros
        result_dict = {k: v for k, v in result_dict.items() if v != 0}
        if not result_dict:
            return target.zero()
        return target(result_dict)

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
            """Placeholder, replaced at import time by :mod:`uconf.__init__`."""

            raise NotImplementedError("Section is not implemented yet")
