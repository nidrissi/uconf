"""Barratt--Eccles operad model on sequences of permutations."""

from __future__ import annotations

import itertools
from itertools import combinations, pairwise, permutations
from typing import TYPE_CHECKING, ClassVar, Iterator

if TYPE_CHECKING:
    from uconf.models.surjection import Surjection

from sage.all import (
    Integer,
    CombinatorialFreeModule,
    Family,
    GradedModulesWithBasis,
    Permutations,
    SymmetricGroup,
    SymmetricGroupAlgebra,
    cached_method,
    tensor,
)
from uconf.core.parented_element import ParentedElementMixin


class BarrattEccles(CombinatorialFreeModule):
    """Barratt--Eccles operad component in fixed arity.

    Basis elements are tuples of permutations in ``S_n`` with no consecutive
    duplicates. Degenerate inputs map to zero, while malformed inputs raise.
    """

    name: ClassVar[str] = "BE"
    connectivity: ClassVar[int] = 0
    """All components live in non-negative degrees."""

    def __init__(self, n, base_ring):
        """Initialize ``E_n`` over ``base_ring``."""
        assert n >= 0, f"Arity must be non-negative. Got {n}."
        name = f"{self.name}{n}"
        super().__init__(
            base_ring,
            tuple,
            prefix=name,
            category=GradedModulesWithBasis(base_ring),
        )
        self.rename(name)
        self._arity = n
        self._symmetric_group = SymmetricGroup(n)
        self._symmetric_group_algebra = SymmetricGroupAlgebra(base_ring, n)
        self.boundary = self.module_morphism(on_basis=self._boundary_on_basis, codomain=self)
        self.planarize = self.module_morphism(
            on_basis=self._planarize_on_basis,
            codomain=tensor([self, SymmetricGroupAlgebra(base_ring, n)]),
        )

    def _element_constructor_(self, x: BarrattEccles.Element | dict | tuple | list):
        """Build elements from a basis key or a sparse coefficient dictionary.

        Tuples with consecutive duplicate permutations map to zero. Invalid
        permutation data raises.
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
        raise TypeError(f"Expected dict or tuple/list; got {type(x).__name__}: {x!r}.")

    def _validate_basis_key(
        self, basis_tuple: tuple | list, keep_dupes=False
    ) -> tuple[SymmetricGroup] | None:
        """Validate a basis tuple and coerce entries to elements of ``S_n``.

        Returns ``None`` for degenerate tuples with consecutive duplicates and
        raises on malformed permutation data.
        """
        if not isinstance(basis_tuple, (tuple, list)):
            raise TypeError(f"Basis key must be a tuple or list, got {type(basis_tuple)}")

        clean_tuple = []
        for i, p in enumerate(basis_tuple):
            if hasattr(p, "parent") and p.parent() == self._symmetric_group:
                clean_tuple.append(p)
                continue

            if hasattr(p, "tuple") and not isinstance(p, (tuple, list)):
                entries = tuple(p.tuple())
                converted = self._symmetric_group(list(entries))
            elif isinstance(p, (tuple, list)):
                entries = tuple(p)
                converted = self._symmetric_group(p)
            else:
                raise TypeError(
                    f"Item {i} in basis tuple must be a permutation or one-line tuple/list. "
                    f"Got {p} ({type(p)})."
                )

            if len(entries) != self.arity():
                raise ValueError(
                    f"Item {i} must be a permutation of {{1, ..., {self.arity()}}}; "
                    f"got length {len(entries)}."
                )
            if any(not isinstance(entry, (int, Integer)) for entry in entries):
                bad_entry = next(
                    entry for entry in entries if not isinstance(entry, (int, Integer))
                )
                raise TypeError(
                    f"Permutation entries must be integers. Got {bad_entry} ({type(bad_entry)})."
                )
            if set(entries) != set(range(1, self.arity() + 1)):
                raise ValueError(
                    f"Item {i} must be a permutation of {{1, ..., {self.arity()}}}. Got {entries}."
                )
            clean_tuple.append(converted)

        if len(clean_tuple) > 0 and not keep_dupes:
            for i in range(len(clean_tuple) - 1):
                if clean_tuple[i] == clean_tuple[i + 1]:
                    return None
        return tuple(clean_tuple)

    def rho(self, data: tuple | list) -> BarrattEccles.Element:
        """Build the normalized element ``(id, sigma_1, sigma_1 sigma_2, ...)``."""
        clean_data = self._validate_basis_key(data, keep_dupes=True)
        if clean_data is None:
            return self.zero()
        id = self._symmetric_group.identity()
        if id in clean_data:
            return self.zero()
        ret = [id]
        for perm in clean_data:
            ret.append(perm * ret[-1])
        return self.term(tuple(ret))

    def arity(self) -> int:
        """Return the fixed arity of this operad component."""
        return self._arity

    @staticmethod
    def unit(base_ring) -> "BarrattEccles.Element":
        """Return the operadic unit in arity ``1``."""
        return BarrattEccles(1, base_ring)(((1,),))

    @staticmethod
    def unit_key() -> tuple:
        """Return the basis key of the unit element in arity ``1``."""
        return (SymmetricGroup(1).identity(),)

    def planar_basis_it(self, d: int) -> Iterator[BarrattEccles.Element]:
        """Iterate over planar basis elements in degree ``d``."""
        assert d >= 0, f"d must be a non-negative integer. Got d={d}."
        perm = permutations(range(1, self._arity + 1))
        u = self._symmetric_group.identity()
        u_tup = u.tuple()
        for values in itertools.product(perm, repeat=d):
            if all(values[i] != values[i + 1] for i in range(len(values) - 1)) and (
                d == 0 or values[0] != u_tup
            ):
                yield self((u,) + tuple(list(v) for v in values))

    def basis_iter(self, d: int) -> Iterator[BarrattEccles.Element]:
        """Iterate over all basis elements in degree ``d``."""
        assert d >= 0, f"d must be a non-negative integer. Got d={d}."
        perm = permutations(range(1, self._arity + 1))
        for sigma, x in itertools.product(perm, self.planar_basis_it(d)):
            yield x.permute(list(sigma))

    @cached_method
    def graded_basis(self, d: int) -> Family:
        """Return the ``Family`` of all basis elements in degree ``d``."""
        return Family(self.basis_iter(d))

    @cached_method
    def graded_planar_basis(self, d: int) -> Family:
        """Return the ``Family`` of planar basis elements in degree ``d``."""
        return Family(self.planar_basis_it(d))

    def _planarize_on_basis(self, basis_element: tuple):
        """Split a basis element into planar representative and group element."""
        perm = basis_element[0]
        perm_inverse = perm.inverse()
        permuted = tuple(perm_inverse * p for p in basis_element)
        return self.term(permuted).tensor(self._symmetric_group_algebra(perm))

    def _boundary_on_basis(self, basis_element: tuple) -> "BarrattEccles.Element":
        """Standard simplicial boundary."""
        res = self.zero()
        if len(basis_element) <= 1:
            return res

        for i in range(len(basis_element)):
            face = basis_element[:i] + basis_element[i + 1 :]
            clean_face = self._validate_basis_key(face)
            if clean_face is None:
                continue
            res += (-1) ** i * self.term(clean_face)
        return res

    # 2. Implement the hook the Category expects
    def degree_on_basis(self, element: tuple) -> int:
        """Return homological degree of a basis element."""
        return len(element) - 1

    @staticmethod
    def compose(x: BarrattEccles, i: int, y: BarrattEccles) -> BarrattEccles:
        """Operadic composition ``x \\circ_i y`` using shuffle/EZ lifting."""
        if x.parent().base_ring() != y.parent().base_ring():
            raise TypeError("Both elements must have the same base ring.")
        m = x.parent().arity()
        n = y.parent().arity()
        assert 1 <= i <= m, f"Index i must be between 1 and {m}. Got {i}."
        target = BarrattEccles(m + n - 1, base_ring=x.parent().base_ring())

        # --- Helper: Composition of single permutations ---
        def _compose_perm_tuple(sigma, idx, tau):
            """
            Composes tuple sigma with tuple tau at index idx.
            Replaces the value 'idx' in sigma with the shifted sequence tau.
            """
            shift = tau.parent().degree() - 1
            res = []
            for val in sigma.tuple():
                if val < idx:
                    res.append(val)
                elif val > idx:
                    res.append(val + shift)
                else:  # val == idx
                    # Insert tau shifted by idx - 1
                    res.extend([t + idx - 1 for t in tau.tuple()])
            # We must return a list, not a tuple!
            # SymmetricGroup(3)([1,2,3]) is the identity permutation
            # SymmetricGroup(3)((1,2,3)) is a cycle of length 3
            return res

        # --- Helper: Eilenberg-Zilber Logic ---
        def term_generator():
            # Iterate over the linear combinations
            for x_basis, x_coeff in x:
                for y_basis, y_coeff in y:
                    # Degrees (length of tuple - 1)
                    p = len(x_basis) - 1
                    q = len(y_basis) - 1

                    # Generate Shuffles of the steps
                    # 0 represents a step in x, 1 represents a step in y
                    multiset = [0] * p + [1] * q
                    shuffles = Permutations(multiset).list()

                    for sh in shuffles:
                        # 1. Calculate the Sign of the shuffle
                        # Formula: (-1)^(sum of 1-based indices of y-steps) - q(q+1)/2
                        # This is equivalent to the sign of the permutation un-shuffling the lists
                        y_indices_sum = sum(k + 1 for k, step in enumerate(sh) if step == 1)
                        sign_exponent = y_indices_sum - (q * (q + 1) // 2)
                        sign = (-1) ** sign_exponent

                        # 2. Build the new basis element (Path lifting)
                        # We start at (x0, y0) and walk the path defined by the shuffle
                        path = []

                        # Current indices in the basis tuples
                        ix, iy = 0, 0

                        # The first vertex is always (x[0] o_i y[0])
                        start_perm = _compose_perm_tuple(x_basis[0], i, y_basis[0])
                        path.append(start_perm)

                        # Walk the steps
                        for step in sh:
                            if step == 0:
                                ix += 1  # Step in X
                            else:
                                iy += 1  # Step in Y

                            # Compose the permutations at the current grid point
                            comp = _compose_perm_tuple(x_basis[ix], i, y_basis[iy])
                            path.append(comp)
                        path = target._validate_basis_key(path)
                        # Yield the constructed basis tuple and the combined coefficient
                        if path is not None:
                            yield (tuple(path), x_coeff * y_coeff * sign)

        return target.sum_of_terms(term_generator())

    def _complexity_on_basis(self, element: tuple) -> int:
        """Return pairwise complexity of a basis element."""
        result = 0
        n = self.arity()

        for i, j in combinations(range(1, n + 1), 2):
            seq = list(
                filter(
                    lambda x: x == i or x == j,
                    (val for perm in element for val in perm.tuple()),
                )
            )
            complexity = len([p for p, q in pairwise(seq) if p != q])
            result = max(result, complexity)
        return result

    class Element(ParentedElementMixin["BarrattEccles"], CombinatorialFreeModule.Element):
        """Elements of the Barratt--Eccles operad component."""

        def arity(self) -> int:
            """Return the arity of this element."""
            return self.parent().arity()

        def planarize(self):
            """Project to planar representative tensored with a group element."""
            parent = self.parent()
            return parent.planarize(self)

        def boundary(self) -> BarrattEccles.Element:
            """Apply the simplicial differential."""
            parent = self.parent()
            return parent.boundary(self)

        def complexity(self) -> int:
            """Return the maximum pairwise complexity on basis support."""
            parent = self.parent()

            return max(
                (parent._complexity_on_basis(basis) for basis in self.support()),
                default=0,
            )

        def permute(self, sigma) -> BarrattEccles.Element:
            """
            Permutes the basis elements of self by precomposing with sigma.
            """
            parent = self.parent()
            if isinstance(sigma, (list, tuple)):
                sigma = parent._symmetric_group(sigma)
            elif not (hasattr(sigma, "parent") and sigma.parent() == parent._symmetric_group):
                raise TypeError(
                    f"Permutation must be a list, tuple, or S_{parent.arity()} element; "
                    f"got {type(sigma).__name__}: {sigma!r}."
                )

            def permuted_term_generator():
                for basis, coeff in self:
                    # Precompose each permutation in the basis tuple with sigma
                    permuted_basis = tuple(sigma * p for p in basis)
                    yield (permuted_basis, coeff)

            return parent.sum_of_terms(permuted_term_generator())

        def diagonal(self):
            """Compute the Alexander--Whitney diagonal ``E -> E \\otimes E``."""
            # 1. Construct the Tensor Product Parent
            # Sage handles this automatically: self.tensor(self) creates the module E (x) E

            parent = self.parent()
            tensor_module = parent.tensor(parent)

            result = tensor_module.zero()

            # 2. Iterate linearly
            for basis_tuple, coeff in self:
                k = len(basis_tuple) - 1  # Degree (length - 1)

                # 3. Apply formula: Sum (front_face) (x) (back_face)
                for i in range(k + 1):
                    # Left side: (sigma_0 ... sigma_i)
                    left_basis = basis_tuple[: i + 1]

                    # Right side: (sigma_i ... sigma_k)
                    right_basis = basis_tuple[i:]

                    # Create the tensor element
                    # In Sage, tensor basis is ((left_key, right_key))
                    tensor_term = tensor_module.term((left_basis, right_basis))

                    result += coeff * tensor_term

            return result

        def table_reduction(self) -> Surjection.Element:
            """Placeholder, replaced at import time by :mod:`uconf.__init__`."""
            raise NotImplementedError("Table reduction is not implemented yet")
