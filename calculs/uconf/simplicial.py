"""Simplicial chains on the standard simplex and tensor products thereof.

Provides :class:`SimplicialChains`, a :class:`CombinatorialFreeModule` whose
basis elements are tensor products of non-degenerate simplices, and
:class:`SimplicialCochains`, the linear dual over a fixed standard simplex.
"""

from __future__ import annotations

from itertools import combinations, combinations_with_replacement, pairwise, product

from sage.all import (
    QQ,
    CombinatorialFreeModule,
    GradedModulesWithBasis,
    tensor,
    Rational,
)


# ---------------------------------------------------------------------------
# SimplicialChains
# ---------------------------------------------------------------------------


class SimplicialChains(CombinatorialFreeModule):
    r"""Iterated tensor product of normalized simplicial chains on `\Delta^\infty`.

    A basis element is a tuple of *simplices*, where each simplex is itself a
    tuple of strictly-increasing non-negative integers.  For example ::

        ((0, 1, 2), (0, 1))

    represents `[0,1,2] \otimes [0,1] \in C_2 \otimes C_1`.

    Parameters
    ----------
    r : int
        Number of tensor factors (*arity*).
    base_ring : ring, default ``QQ``
        Coefficient ring.
    """

    def __init__(self, r: int = 1, base_ring=QQ):
        assert r >= 1, f"Arity must be >= 1, got {r}."
        name = f"C^⊗{r}" if r > 1 else "C"
        super().__init__(
            base_ring,
            tuple,
            prefix=name,
            category=GradedModulesWithBasis(base_ring),
        )
        self.rename(name)
        self._arity: int = r
        self.boundary = self.module_morphism(
            on_basis=self._boundary_on_basis, codomain=self
        )

    # -- helpers ------------------------------------------------------------

    def arity(self) -> int:
        """Number of tensor factors."""
        return self._arity

    def native_tensor_parent(self):
        """Return the native Sage tensor-product parent for this arity.

        For arity ``r``, this is ``tensor([SimplicialChains(1)] * r)``.
        """
        if not hasattr(self, "_native_tensor_parent"):
            factor = SimplicialChains(r=1, base_ring=self.base_ring())
            self._native_tensor_parent = tensor([factor] * self.arity())
        return self._native_tensor_parent

    # -- element constructor ------------------------------------------------

    def _element_constructor_(self, x):
        """Build elements from basis tuples or sparse dicts.

        Accepts:
        * A *dict* ``{basis_key: coeff, ...}``  – linear combination.
        * A *tuple/list* of tuples  – single basis element.
        """
        if isinstance(x, dict):
            clean = {}
            for key, coeff in x.items():
                k = self._validate_basis_key(key)
                if k is not None:
                    clean[k] = clean.get(k, 0) + coeff
            return super()._element_constructor_(clean)

        if isinstance(x, (tuple, list)):
            k = self._validate_basis_key(x)
            if k is None:
                return self.zero()
            return self.term(k)

        raise TypeError(
            f"Input must be a dict or tuple/list of simplex-tuples, got {type(x)}."
        )

    @staticmethod
    def _validate_basis_key(key) -> "tuple | None":
        """Normalise and validate a basis key.

        A valid key is a tuple of *non-degenerate simplices*, each being a
        strictly increasing tuple of non-negative integers.  Returns ``None``
        (→ zero) for degenerate or empty simplices.
        """
        if not isinstance(key, (tuple, list)):
            return None
        normed = []
        for simplex in key:
            if not isinstance(simplex, (tuple, list)):
                return None
            s = tuple(simplex)
            # Empty simplex → degenerate
            if len(s) == 0:
                return None
            if any((not isinstance(v, int)) or v < 0 for v in s):
                return None
            # Strictly increasing ⇔ non-degenerate
            if any(a >= b for a, b in pairwise(s)):
                return None
            normed.append(s)
        if len(normed) == 0:
            return None
        return tuple(normed)

    # -- grading ------------------------------------------------------------

    def degree_on_basis(self, basis_element: tuple) -> int:
        """Total degree = sum of dimensions of the simplex factors."""
        return sum(len(s) - 1 for s in basis_element)

    # -- boundary -----------------------------------------------------------

    def _boundary_on_basis(self, basis_element: tuple):
        r"""Tensor-product differential on one basis element.

        .. math::
            \partial(x_1 \otimes \cdots \otimes x_r)
            = \sum_i (-1)^{|x_1|+\cdots+|x_{i-1}|}
              x_1 \otimes \cdots \otimes \partial x_i \otimes \cdots \otimes x_r

        where `\partial [v_0,\dots,v_n] = \sum_{j=0}^n (-1)^j [v_0,\dots,\hat v_j,\dots,v_n]`.
        """

        def terms():
            accumulated_dim = 0
            for idx, simplex in enumerate(basis_element):
                dim = len(simplex) - 1
                for j in range(dim + 1):
                    face = simplex[:j] + simplex[j + 1 :]
                    if len(face) == 0:
                        continue
                    # Strictly-increasing check (face of non-deg is non-deg
                    # unless dim-0, which we skip above)
                    new_key = basis_element[:idx] + (face,) + basis_element[idx + 1 :]
                    sign = (-1) ** ((accumulated_dim + j) % 2)
                    validated = self._validate_basis_key(new_key)
                    if validated is not None:
                        yield (validated, sign)
                accumulated_dim += dim

        return self.sum_of_terms(terms())

    # -- standard element ---------------------------------------------------

    @staticmethod
    def standard_element(n: int, times: int = 1):
        r"""The chain `(0, 1, \dots, n)^{\otimes \text{times}}` in `C_n(\Delta^n)`.

        Parameters
        ----------
        n : int  (non-negative)
            Dimension of the standard simplex.
        times : int, default 1
            Number of tensor copies.
        """
        assert n >= 0
        parent = SimplicialChains(r=times)
        simplex = tuple(range(n + 1))
        key = (simplex,) * times
        return parent(key)

    # -- basis enumeration on Delta^N ---------------------------------------

    @staticmethod
    def basis_it(N: int, r: int = 1):
        """Iterate over all non-degenerate basis elements in `C(Delta^N)^{otimes r}`.

        Yields elements of ``SimplicialChains(r)`` whose simplices have
        vertices in ``{0, ..., N}``.
        """
        parent = SimplicialChains(r=r)
        # All non-empty strictly-increasing subsequences of 0..N
        simplices = []
        for k in range(1, N + 2):
            simplices.extend(combinations(range(N + 1), k))
        for factors in product(simplices, repeat=r):
            yield parent(tuple(factors))

    # -----------------------------------------------------------------------
    # Element class
    # -----------------------------------------------------------------------

    class Element(CombinatorialFreeModule.Element):
        """Elements of a simplicial chains component."""

        def boundary(self):
            """Apply the tensor-product differential."""
            return self.parent().boundary(self)

        def arity(self) -> int:
            """Number of tensor factors."""
            return self.parent().arity()

        def to_native_tensor(self):
            """Convert to the corresponding native Sage tensor-product element."""
            parent = self.parent()
            tensor_parent = parent.native_tensor_parent()
            if parent.arity() == 1:
                return self

            factor = SimplicialChains(r=1, base_ring=parent.base_ring())
            result = tensor_parent.zero()
            for key, coeff in self:
                factors = [factor((simplex,)) for simplex in key]
                term = factors[0]
                for nxt in factors[1:]:
                    term = term.tensor(nxt)
                result += coeff * term
            return result

        def iterated_diagonal(self, times: int = 1, coord: int = 1):
            r"""Iterated Alexander-Whitney diagonal applied at tensor factor *coord*.

            The AW diagonal is the chain map `\Delta: C \to C \otimes C`
            given by

            .. math::
                \Delta((v_0,\dots,v_n))
                = \sum_{i=0}^n (v_0,\dots,v_i) \otimes (v_i,\dots,v_n).

            This method computes `\Delta^{k}` (applying `k = \text{times}`
            copies of the diagonal, producing `k+1` factors from the chosen
            coordinate) and inserts the result back into the tensor product.

            Parameters
            ----------
            times : int
                Number of *extra* factors produced (``times=1`` gives two
                factors, ``times=k`` gives ``k+1`` factors).
            coord : int (1-based)
                Which tensor factor to diagonalize.

            Returns
            -------
            SimplicialChains.Element
                Element in ``SimplicialChains(self.arity() + times)``.
            """
            if times == 0:
                return self
            r = self.arity()
            assert 1 <= coord <= r, f"coord={coord} out of range [1, {r}]"
            target = SimplicialChains(r=r + times)

            def terms():
                for basis_key, coeff in self:
                    left = basis_key[: coord - 1]
                    simplex = basis_key[coord - 1]
                    right = basis_key[coord:]
                    dim = len(simplex) - 1
                    # All ways to split simplex into (times+1) overlapping parts
                    for p in combinations_with_replacement(range(dim + 1), times):
                        p = (0,) + p + (dim,)
                        pieces = []
                        for a, b in pairwise(p):
                            pieces.append(simplex[a : b + 1])
                        new_key = left + tuple(pieces) + right
                        validated = SimplicialChains._validate_basis_key(new_key)
                        if validated is not None:
                            yield (validated, coeff)

            return target.sum_of_terms(terms())


# ---------------------------------------------------------------------------
# SimplicialCochains
# ---------------------------------------------------------------------------


class SimplicialCochains(CombinatorialFreeModule):
    r"""Normalized cochains on `\Delta^N`, with tensor products.

    The dual of :class:`SimplicialChains` restricted to simplices with
    vertices in `\{0, \dots, N\}`.  A basis cochain is identified with the
    *same* tuple-of-tuples key as ``SimplicialChains``; `f` evaluates to
    `\delta_{f, \sigma}` on a chain basis element `\sigma`.

    Parameters
    ----------
    N : int
        Dimension of the ambient standard simplex.
    r : int
        Number of tensor factors.
    base_ring
        Coefficient ring (default ``QQ``).
    """

    def __init__(self, N: int, r: int = 1, base_ring=QQ):
        assert N >= 0 and r >= 1
        name = f"C*({N})^⊗{r}" if r > 1 else f"C*({N})"
        super().__init__(
            base_ring,
            tuple,
            prefix=name,
            category=GradedModulesWithBasis(base_ring),
        )
        self.rename(name)
        self._N = N
        self._arity: int = r
        self.coboundary = self.module_morphism(
            on_basis=self._coboundary_on_basis, codomain=self
        )

    def arity(self) -> int:
        return self._arity

    def simplex_dim(self) -> int:
        return self._N

    # -- element constructor ------------------------------------------------

    def _element_constructor_(self, x):
        if isinstance(x, dict):
            clean = {}
            for key, coeff in x.items():
                k = SimplicialChains._validate_basis_key(key)
                if k is not None:
                    clean[k] = clean.get(k, 0) + coeff
            return super()._element_constructor_(clean)
        if isinstance(x, (tuple, list)):
            k = SimplicialChains._validate_basis_key(x)
            if k is None:
                return self.zero()
            return self.term(k)
        raise TypeError(f"Expected dict or tuple, got {type(x)}.")

    # -- grading (cohomological = negative of chain degree) -----------------

    def degree_on_basis(self, basis_element: tuple) -> int:
        """Cohomological degree = sum of dimensions."""
        return sum(len(s) - 1 for s in basis_element)

    # -- coboundary ---------------------------------------------------------

    def _coboundary_on_basis(self, basis_element: tuple):
        r"""Coboundary: the transpose of the chain boundary.

        `(\delta f)(\sigma) = f(\partial \sigma)`.  We compute the
        *transpose* of the boundary matrix restricted to `\Delta^N`.
        """
        # Build all chains whose boundary contains this basis_element
        # Equivalently: insert a vertex into each simplex factor at every
        # valid position to produce a (degree+1) cochain.

        def terms():
            accumulated_dim = 0
            for idx, simplex in enumerate(basis_element):
                dim = len(simplex) - 1
                # simplex = (v_0,...,v_dim) strictly increasing in {0..N}
                # A co-face inserts a vertex w at position j in the simplex
                # This corresponds to the j-th face map d_j hitting the
                # augmented simplex and producing our simplex.
                # We need to find all (dim+2)-simplices in Delta^N whose
                # j-th face is our simplex.
                for j in range(dim + 2):
                    # Determine the inserted vertex
                    if j == 0:
                        candidates = range(0, simplex[0])
                    elif j == dim + 1:
                        candidates = range(simplex[-1] + 1, self._N + 1)
                    else:
                        candidates = range(simplex[j - 1] + 1, simplex[j])
                    for w in candidates:
                        augmented = simplex[:j] + (w,) + simplex[j:]
                        new_key = (
                            basis_element[:idx]
                            + (augmented,)
                            + basis_element[idx + 1 :]
                        )
                        sign = (-1) ** ((accumulated_dim + j) % 2)
                        yield (new_key, sign)
                accumulated_dim += dim

        return self.sum_of_terms(terms())

    # -- evaluation pairing -------------------------------------------------

    @staticmethod
    def evaluate(cochain, chain) -> "int | Rational":
        r"""Kronecker pairing `\langle f, \sigma \rangle`.

        Both *cochain* and *chain* must be expressed in the same basis
        (tuples of simplex tuples).  Returns `\sum_k f_k \sigma_k` where
        the sum runs over matching basis keys.
        """
        result = 0
        for k_f, v_f in cochain:
            for k_c, v_c in chain:
                if k_f == k_c:
                    result += v_f * v_c
        return result

    # -- standard dual element ----------------------------------------------

    @staticmethod
    def dual_basis_it(N: int, r: int = 1):
        """Iterate over the dual basis cochains on `\\Delta^N` in arity *r*."""
        parent = SimplicialCochains(N=N, r=r)
        simplices = []
        for k in range(1, N + 2):
            simplices.extend(combinations(range(N + 1), k))
        for factors in product(simplices, repeat=r):
            yield parent(tuple(factors))

    # -- Element class ------------------------------------------------------

    class Element(CombinatorialFreeModule.Element):
        def coboundary(self):
            return self.parent().coboundary(self)

        def arity(self) -> int:
            return self.parent().arity()
