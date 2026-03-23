"""Finite-arity model of the Lie operad in a Hall-type basis."""

import itertools as py_itertools
from typing import Any, ClassVar, Iterator

from sage.all import (
    CombinatorialFreeModule,
    Family,
    GradedModulesWithBasis,
    Permutation,
    SymmetricGroup,
    cached_method,
    matrix,
    vector,
)
from uconf.core.display import latex_linear_combination
from uconf.core.parented_element import ParentedElementMixin


class Lie(CombinatorialFreeModule):
    """Lie operad component in arity ``n``.

    Basis keys are permutations of ``(1, ..., n-1)`` and encode nested brackets
    ``[x_i, -]`` ending in ``x_n``.

    Computational data (PBW matrix, its left-inverse, word/basis-key lists, and
    associative expansions) are cached via :func:`sage.misc.cachefunc.cached_method`.
    Since :class:`~sage.combinat.free_module.CombinatorialFreeModule` inherits from
    :class:`~sage.structure.unique_representation.UniqueRepresentation`, ``Lie(n,
    base_ring)`` always returns the same instance for a given ``(n, base_ring)`` pair,
    so these instance-level caches are automatically shared across all callers—including
    :meth:`compose`, which creates a new ``Lie`` parent on each invocation.
    """

    name: ClassVar[str] = "Lie"
    connectivity: ClassVar[int] = 0
    """All components live in degree 0 (the Lie operad is concentrated in degree 0)."""

    def __init__(self, n, base_ring):
        """Initialize ``Lie(n)`` over ``base_ring``."""
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
        self.boundary = self.module_morphism(on_basis=lambda basis: self.zero(), codomain=self)

    @cached_method
    def _basis_keys(self) -> list[tuple[int, ...]]:
        """Return and cache the list of basis keys in this arity."""
        n = int(self.arity())
        if n == 0:
            return []
        if n == 1:
            return [()]
        return list(py_itertools.permutations(range(1, n), n - 1))

    def _validate_basis_key(self, basis_key: tuple | list) -> tuple[int, ...] | None:
        """Validate and normalize a basis key.

        Returns ``None`` only in arity ``0`` where no nonzero basis elements
        exist.
        """
        if self.arity() == 0:
            return None

        if not isinstance(basis_key, (tuple, list)):
            raise TypeError(f"Basis key must be a tuple/list, got {type(basis_key)}")

        clean = tuple(int(i) for i in basis_key)
        n = int(self.arity())

        if n == 1:
            if clean != ():
                raise ValueError(f"The only basis key in arity 1 is (). Got {clean}.")
            return clean

        if len(clean) != n - 1:
            raise ValueError(f"Basis key in arity {n} must have length {n - 1}. Got {len(clean)}.")

        if set(clean) != set(range(1, n)):
            raise ValueError(f"Basis key must be a permutation of 1..{n - 1}. Got {clean}.")

        return clean

    def _element_constructor_(self, x):
        """Build elements from basis keys or sparse dictionaries."""
        if isinstance(x, dict):
            clean_dict = {}
            for key, coeff in x.items():
                clean_key = self._validate_basis_key(key)
                if clean_key is None:
                    continue
                clean_dict[clean_key] = coeff
            return super()._element_constructor_(clean_dict)

        if isinstance(x, (tuple, list)):
            clean_key = self._validate_basis_key(x)
            if clean_key is None:
                return self.zero()
            return self.term(clean_key)

        raise TypeError(f"Expected dict or tuple/list; got {type(x).__name__}: {x!r}.")

    def _bracket(self, left: dict[tuple[int, ...], Any], right: dict[tuple[int, ...], Any]):
        """Compute commutator ``[left, right]`` on associative expansions."""
        out: dict[tuple[int, ...], Any] = {}
        for wl, cl in left.items():
            for wr, cr in right.items():
                c = cl * cr
                if c == 0:
                    continue

                wlr = wl + wr
                wrl = wr + wl
                out[wlr] = out.get(wlr, self.base_ring().zero()) + c
                out[wrl] = out.get(wrl, self.base_ring().zero()) - c

        return {w: c for w, c in out.items() if c != 0}

    @cached_method
    def _assoc_from_basis_key(self, basis_key: tuple[int, ...]) -> dict[tuple[int, ...], Any]:
        """Expand one Lie basis element into the associative word basis.

        Results are cached via :func:`~sage.misc.cachefunc.cached_method`.
        """
        n = int(self.arity())
        if n == 0:
            return {}
        if n == 1:
            return {(1,): self.base_ring().one()}
        seq = basis_key + (n,)
        current: dict[tuple[int, ...], Any] = {(seq[-1],): self.base_ring().one()}
        for a in reversed(seq[:-1]):
            current = self._bracket({(a,): self.base_ring().one()}, current)
        return current

    @cached_method
    def _word_basis(self) -> list[tuple[int, ...]]:
        """Return and cache the associative word basis in arity ``n``."""
        n = int(self.arity())
        if n == 0:
            return []
        return list(py_itertools.permutations(range(1, n + 1), n))

    @cached_method
    def _pbw_matrix(self):
        """Return the PBW change-of-basis matrix from Lie to words."""
        words = self._word_basis()
        keys = self._basis_keys()

        if self.arity() == 0:
            return matrix(self.base_ring(), 0, 0, [])

        if self.arity() == 1:
            return matrix(self.base_ring(), 1, 1, [1])

        data = []
        for w in words:
            for key in keys:
                data.append(self._assoc_from_basis_key(key).get(w, self.base_ring().zero()))
        return matrix(self.base_ring(), len(words), len(keys), data)

    @cached_method
    def _pbw_left_inverse(self):
        """Return the left-inverse ``L`` of the PBW matrix.

        ``L`` satisfies ``L * M = I`` where ``M`` is the PBW matrix
        (shape ``n! × (n-1)!``).  It is computed once via
        :func:`~sage.misc.cachefunc.cached_method` and applying ``L`` as a
        matrix–vector product replaces the per-call ``solve_right``
        (fresh Gaussian elimination) in :meth:`_element_from_assoc`.

        Over fields where the Gram matrix ``M^T M`` is invertible (e.g. ``QQ``),
        the formula ``L = (M^T M)^{-1} M^T`` is used.  Over fields of small
        characteristic (e.g. ``GF(2)``), ``M^T M`` may be singular even though
        ``M`` has full column rank.  In that case, a set of linearly independent
        rows is found via rank tests and the left-inverse is built from the
        resulting square invertible sub-matrix.
        """
        pbw = self._pbw_matrix()

        if self.arity() <= 1:
            # Square 1×1 or 0×0 — the matrix is already its own inverse.
            return pbw

        pbw_T = pbw.transpose()
        try:
            return (pbw_T * pbw).inverse() * pbw_T
        except (ZeroDivisionError, ArithmeticError):
            return self._pbw_left_inverse_via_pivots(pbw)

    def _pbw_left_inverse_via_pivots(self, pbw):
        """Compute a left-inverse of *pbw* by selecting pivot rows.

        Finds a maximal set of linearly independent rows of *pbw* to form a
        square invertible sub-matrix, then constructs the left-inverse as
        ``sub^{-1} * selector`` where *selector* picks out the pivot rows.
        """
        n_cols = pbw.ncols()
        n_rows = pbw.nrows()
        R = self.base_ring()

        selected: list[int] = []
        for i in range(n_rows):
            candidate = selected + [i]
            sub = pbw.matrix_from_rows(candidate)
            if sub.rank() == len(candidate):
                selected.append(i)
                if len(selected) == n_cols:
                    break

        if len(selected) < n_cols:
            raise ArithmeticError(
                f"PBW matrix does not have full column rank over {R} (arity {self.arity()})"
            )

        sub = pbw.matrix_from_rows(selected)
        sub_inv = sub.inverse()

        # Build the left-inverse: sub_inv * selector,
        # where selector is an n_cols × n_rows matrix picking the pivot rows.
        selector = matrix(R, n_cols, n_rows, sparse=True)
        for i, row_idx in enumerate(selected):
            selector[i, row_idx] = R.one()

        return sub_inv * selector

    def _element_from_assoc(self, assoc: dict[tuple[int, ...], Any]) -> "Lie.Element":
        """Reconstruct a Lie element from associative coefficients.

        Uses the precomputed left-inverse of the PBW matrix (a single
        matrix–vector multiply) instead of a fresh ``solve_right`` call.
        """
        if not assoc:
            return self.zero()

        if self.arity() == 0:
            return self.zero()

        words = self._word_basis()
        vec = vector(self.base_ring(), [assoc.get(w, self.base_ring().zero()) for w in words])

        if self.arity() == 1:
            return vec[0] * self.term(())

        left_inv = self._pbw_left_inverse()
        coords = left_inv * vec
        keys = self._basis_keys()
        return self.sum_of_terms((keys[i], coords[i]) for i in range(len(keys)) if coords[i] != 0)

    def arity(self) -> int:
        """Return the arity of this Lie operad component."""
        return self._arity

    def basis_iter(self, d: int) -> Iterator["Lie.Element"]:
        """Iterate over the canonical Lie basis in this fixed arity and the given degree."""
        if d == 0:
            for key in self._basis_keys():
                yield self.term(key)

    @cached_method
    def graded_basis(self, d: int) -> Family:
        """Return the ``Family`` of all basis elements in degree ``d``."""
        return Family(self.basis_iter(d))

    @staticmethod
    def unit(base_ring):
        """Return the operadic unit in arity ``1``."""
        return Lie(1, base_ring)(())

    @staticmethod
    def unit_key() -> tuple:
        """Return the basis key of the unit element in arity ``1``."""
        return ()

    def degree_on_basis(self, element) -> int:
        """Return homological degree on basis elements (always ``0`` here)."""
        return 0

    def _repr_term(self, basis_element: tuple[int, ...]) -> str:
        """String representation of one basis element as nested brackets."""
        n = self.arity()
        if n == 0:
            return "0"
        if n == 1:
            return "x1"

        expr = f"x{n}"
        for i in reversed(basis_element):
            expr = f"[x{i},{expr}]"
        return expr

    def _latex_term(self, basis_element: tuple[int, ...]) -> str:
        """LaTeX representation of one basis element as nested brackets."""
        n = self.arity()
        if n == 0:
            return "0"
        if n == 1:
            return "x_{1}"

        expr = f"x_{{{n}}}"
        for i in reversed(basis_element):
            expr = f"\\left[x_{{{i}}}, {expr}\\right]"
        return expr

    @staticmethod
    def compose(x: "Lie.Element", i: int, y: "Lie.Element") -> "Lie.Element":
        """Operadic composition ``x \\circ_i y`` in the Lie operad.

        The implementation follows the algorithmic approach described in
        Bremner–Dotsenko *Algebraic Operads: An Algorithmic Companion*:

        1. Expand ``x`` and ``y`` into the associative word basis *once* each,
           combining all terms into single dicts before the composition loop.
           This avoids calling :meth:`_assoc_from_basis_key` redundantly for
           every outer word of ``x`` (the bottleneck in the original code).
        2. Perform variable-substitution and renaming in a single double loop
           over the pre-aggregated word dicts.
        3. Recover the Lie coordinates via a pre-cached left-inverse matrix
           (one matrix–vector multiply) rather than a fresh linear solve.
        """
        x_parent = x.parent()
        y_parent = y.parent()
        m = x_parent.arity()
        n = y_parent.arity()

        assert 1 <= i <= m, f"Index i must be between 1 and {m}. Got {i}."
        if x_parent.base_ring() != y_parent.base_ring():
            raise TypeError("Both elements must have the same base ring.")

        target = Lie(m + n - 1, base_ring=x_parent.base_ring())
        base_ring = target.base_ring()

        # --- Step 1: aggregate the full associative expansion of x ------------
        x_assoc: dict[tuple[int, ...], Any] = {}
        for x_key, x_coeff in x:
            for word, coeff in x_parent._assoc_from_basis_key(x_key).items():
                c = x_coeff * coeff
                x_assoc[word] = x_assoc.get(word, base_ring.zero()) + c
        x_assoc = {w: c for w, c in x_assoc.items() if c != 0}

        # --- Step 2: aggregate the full associative expansion of y, pre-shifted
        #             so that variable k in y becomes variable k + i - 1 ---------
        y_assoc_shifted: dict[tuple[int, ...], Any] = {}
        for y_key, y_coeff in y:
            for word, coeff in y_parent._assoc_from_basis_key(y_key).items():
                shifted = tuple(k + i - 1 for k in word)
                c = y_coeff * coeff
                y_assoc_shifted[shifted] = y_assoc_shifted.get(shifted, base_ring.zero()) + c
        y_assoc_shifted = {w: c for w, c in y_assoc_shifted.items() if c != 0}

        # --- Step 3: substitute position i in each outer word with each inner --
        result_assoc: dict[tuple[int, ...], Any] = {}
        for outer_word, outer_coeff in x_assoc.items():
            pos = outer_word.index(i)
            shifted_left = tuple(j if j < i else j + n - 1 for j in outer_word[:pos])
            shifted_right = tuple(j if j < i else j + n - 1 for j in outer_word[pos + 1 :])

            for shifted_inner, inner_coeff in y_assoc_shifted.items():
                new_word = shifted_left + shifted_inner + shifted_right
                c = outer_coeff * inner_coeff
                result_assoc[new_word] = result_assoc.get(new_word, base_ring.zero()) + c

        result_assoc = {w: c for w, c in result_assoc.items() if c != 0}
        return target._element_from_assoc(result_assoc)

    class Element(ParentedElementMixin["Lie"], CombinatorialFreeModule.Element):
        """Elements of a fixed-arity Lie component."""

        def _repr_latex_(self) -> str:
            return latex_linear_combination(self, lambda basis: self.parent()._latex_term(basis))

        def arity(self) -> int:
            """Return the element arity."""
            return self.parent().arity()

        def boundary(self) -> "Lie.Element":
            """Apply the differential (trivial in this model)."""
            parent = self.parent()
            return parent.boundary(self)

        def permute(self, sigma: Permutation | list | tuple) -> "Lie.Element":
            """Apply the right action of a permutation on generators."""
            parent = self.parent()

            if isinstance(sigma, (list, tuple)):
                sigma_perm: Permutation = parent._symmetric_group(sigma)
            elif not (hasattr(sigma, "parent") and sigma.parent() == parent._symmetric_group):
                raise TypeError(
                    f"Permutation must be a list, tuple, or S_{parent.arity()} element; "
                    f"got {type(sigma).__name__}: {sigma!r}."
                )
            else:
                sigma_perm: Permutation = sigma

            assoc: dict[tuple[int, ...], Any] = {}
            for key, coeff in self:
                key_assoc = parent._assoc_from_basis_key(key)
                for word, word_coeff in key_assoc.items():
                    permuted_word = tuple(sigma_perm(i) for i in word)
                    assoc[permuted_word] = (
                        assoc.get(permuted_word, self.base_ring().zero()) + coeff * word_coeff
                    )

            assoc = {w: c for w, c in assoc.items() if c != 0}
            return parent._element_from_assoc(assoc)
