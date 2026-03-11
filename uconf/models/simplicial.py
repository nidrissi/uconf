"""Simplicial chains on the standard simplex.

Provides :class:`SimplicialChains`, a :class:`CombinatorialFreeModule` whose
basis elements are **single** non-degenerate simplices (strictly-increasing
tuples of non-negative integers), and :class:`SimplicialCochains`, the
linear dual over a fixed standard simplex ``Δ^N``.

For tensor products of chain modules use the native Sage ``tensor()``
function::

    T = tensor([SimplicialChains(), SimplicialChains()])

The :meth:`SimplicialChains.tensor_boundary` static method computes the
Koszul tensor-product differential on such elements.
"""

from __future__ import annotations

from itertools import combinations, combinations_with_replacement, pairwise

from sage.all import (
    QQ,
    CombinatorialFreeModule,
    GradedModulesWithBasis,
    Rational,
    tensor,
)


# ---------------------------------------------------------------------------
# SimplicialChains
# ---------------------------------------------------------------------------


class SimplicialChains(CombinatorialFreeModule):
    r"""Normalized simplicial chains on `\Delta^\infty`.

    A basis element is a **single** non-degenerate simplex, i.e. a
    strictly-increasing tuple of non-negative integers.  For example::

        (0, 1, 2)   # the 2-simplex [0,1,2]
        (3,)        # the vertex 3

    For tensor products of chain modules use ``tensor([SimplicialChains()]*r)``.

    Parameters
    ----------
    base_ring : ring, default ``QQ``
        Coefficient ring.
    """

    def __init__(self, base_ring=QQ):
        name = "C"
        super().__init__(
            base_ring,
            tuple,
            prefix=name,
            category=GradedModulesWithBasis(base_ring),
        )
        self.rename(name)
        self.boundary = self.module_morphism(
            on_basis=self._boundary_on_basis, codomain=self
        )

    # -- helpers ------------------------------------------------------------

    def as_surjection_coalgebra(self):
        """Return the canonical ``SurjectionDual``-coalgebra wrapper."""
        from uconf.algebraic.simplicial import SurjectionSimplicialChainCoalgebra

        return SurjectionSimplicialChainCoalgebra(self)

    # -- element constructor ------------------------------------------------

    def _element_constructor_(self, x):
        """Build an element from a simplex tuple or a sparse dict.

        Accepts:

        * A *dict* ``{simplex: coeff, ...}`` – linear combination.
        * A *tuple/list* of integers – single simplex.
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

        raise TypeError(f"Input must be a dict or a simplex tuple, got {type(x)}.")

    @staticmethod
    def _validate_basis_key(key) -> "tuple | None":
        """Validate a single simplex as a basis key.

        A valid simplex is a strictly-increasing non-empty tuple of
        non-negative integers.  Returns ``None`` for invalid or degenerate
        inputs.
        """
        if not isinstance(key, (tuple, list)):
            return None
        s = tuple(key)
        if len(s) == 0:
            return None
        if any((not isinstance(v, int)) or v < 0 for v in s):
            return None
        if any(a >= b for a, b in pairwise(s)):
            return None
        return s

    # -- grading ------------------------------------------------------------

    def degree_on_basis(self, simplex: tuple) -> int:
        """Degree = dimension of the simplex = ``len(simplex) - 1``."""
        return len(simplex) - 1

    # -- boundary -----------------------------------------------------------

    def _boundary_on_basis(self, simplex: tuple):
        r"""Simplicial boundary of one basis element.

        .. math::
            \partial [v_0, \dots, v_n]
            = \sum_{j=0}^n (-1)^j [v_0, \dots, \hat{v}_j, \dots, v_n].
        """
        result = self.zero()
        for j in range(len(simplex)):
            face = simplex[:j] + simplex[j + 1 :]
            k = self._validate_basis_key(face)
            if k is not None:
                sign = (-1) ** j
                result += sign * self.term(k)
        return result

    # -- standard element ---------------------------------------------------

    @staticmethod
    def fundamental_chain(n: int) -> "SimplicialChains.Element":
        r"""The chain `[0, 1, \dots, n]` in `C_n(\Delta^n)`.

        Parameters
        ----------
        n : int  (non-negative)
            Dimension of the standard simplex ``Δ^n`` (so the chain lies in degree ``n``).
        """
        assert n >= 0
        return SimplicialChains().term(tuple(range(n + 1)))

    # -- basis enumeration on Delta^N ----------------------------------------

    @staticmethod
    def basis_it(N: int):
        """Iterate over all non-degenerate basis elements in ``C(Δ^N)``.

        Yields :class:`SimplicialChains` elements with vertices in
        ``{0, ..., N}``.
        """
        parent = SimplicialChains()
        for k in range(1, N + 2):
            for simplex in combinations(range(N + 1), k):
                yield parent.term(simplex)

    # -- tensor-product boundary --------------------------------------------

    @staticmethod
    def tensor_boundary(x):
        r"""Koszul tensor-product differential on ``tensor([SC]*r)`` elements.

        For `x = \sum x_1 \otimes \dots \otimes x_r` computes

        .. math::
            \partial x = \sum_i (-1)^{\deg x_1 + \dots + \deg x_{i-1}}
                x_1 \otimes \dots \otimes \partial x_i \otimes \dots \otimes x_r.

        Also works on plain :class:`SimplicialChains` elements (delegates
        to ``parent().boundary()``).

        Parameters
        ----------
        x : element of ``tensor([SimplicialChains()]*r)`` or
            :class:`SimplicialChains`
        """
        parent = x.parent()
        if isinstance(parent, SimplicialChains):
            return parent.boundary(x)
        # Native tensor product: parent._sets = (SC, SC, ...) – r factors.
        # _sets is the standard Sage attribute for tensor product factor modules.
        factors = parent._sets
        result = parent.zero()
        for tensor_key, coeff in x:
            # tensor_key = (s_1, s_2, ..., s_r) where each s_i is a simplex tuple.
            deg_acc = 0
            for i, (SC_i, simplex) in enumerate(zip(factors, tensor_key)):
                sign = (-1) ** (deg_acc % 2)
                for new_simplex, bcoeff in SC_i.boundary(SC_i.term(simplex)):
                    new_key = tensor_key[:i] + (new_simplex,) + tensor_key[i + 1 :]
                    result += sign * coeff * bcoeff * parent.term(new_key)
                deg_acc += len(simplex) - 1
        return result

    # -----------------------------------------------------------------------
    # Element class
    # -----------------------------------------------------------------------

    class Element(CombinatorialFreeModule.Element):
        """Elements of :class:`SimplicialChains`."""

        def boundary(self) -> "SimplicialChains.Element":
            """Apply the simplicial boundary ∂."""
            return self.parent().boundary(self)

        def iterated_diagonal(self, times: int = 1):
            r"""Iterated Alexander-Whitney diagonal.

            The AW diagonal is the chain map `\Delta: C \to C \otimes C`

            .. math::
                \Delta([v_0,\dots,v_n])
                = \sum_{i=0}^n [v_0,\dots,v_i] \otimes [v_i,\dots,v_n].

            This method applies the diagonal ``times`` times, producing
            ``times + 1`` tensor factors.  Returns an element of
            ``tensor([SimplicialChains()]*( times + 1))``.

            Parameters
            ----------
            times : int
                Number of extra factors (``times=1`` → 2-fold tensor,
                ``times=k`` → ``(k+1)``-fold tensor).
            """
            SC = self.parent()
            if times == 0:
                return self
            target = tensor([SC] * (times + 1))

            def terms():
                for simplex, coeff in self:
                    dim = len(simplex) - 1
                    # All ways to split simplex into (times+1) overlapping parts.
                    for split in combinations_with_replacement(range(dim + 1), times):
                        p = (0,) + split + (dim,)
                        pieces = tuple(simplex[a : b + 1] for a, b in pairwise(p))
                        if any(len(pc) < 1 for pc in pieces):
                            continue
                        yield (pieces, coeff)

            return target.sum_of_terms(terms())


# ---------------------------------------------------------------------------
# SimplicialCochains
# ---------------------------------------------------------------------------


class SimplicialCochains(CombinatorialFreeModule):
    r"""Normalized cochains on `\Delta^N`.

    The linear dual of :class:`SimplicialChains` restricted to simplices
    with vertices in `\{0, \dots, N\}`.  Basis elements are the same
    simplex tuples as in :class:`SimplicialChains`; a basis cochain
    `f_\sigma` evaluates to `\delta_{\sigma, \tau}` on a chain `\tau`.

    Parameters
    ----------
    N : int
        Dimension of the ambient standard simplex.
    base_ring
        Coefficient ring (default ``QQ``).
    """

    def __init__(self, N: int, base_ring=QQ):
        assert N >= 0
        name = f"C*({N})"
        super().__init__(
            base_ring,
            tuple,
            prefix=name,
            category=GradedModulesWithBasis(base_ring),
        )
        self.rename(name)
        self._N = N
        self.coboundary = self.module_morphism(
            on_basis=self._coboundary_on_basis, codomain=self
        )

    def simplex_dim(self) -> int:
        """Ambient simplex dimension N."""
        return self._N

    def as_surjection_algebra(self):
        """Return the canonical ``Surjection``-algebra wrapper."""
        from uconf.algebraic.simplicial import SurjectionSimplicialCochainAlgebra

        return SurjectionSimplicialCochainAlgebra(self)

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
        raise TypeError(f"Expected dict or simplex tuple, got {type(x)}.")

    # -- grading ------------------------------------------------------------

    def degree_on_basis(self, simplex: tuple) -> int:
        """Homological degree convention: ``-(len(simplex) - 1)``."""
        return -(len(simplex) - 1)

    # -- coboundary ---------------------------------------------------------

    def _coboundary_on_basis(self, simplex: tuple):
        r"""Coboundary: the transpose of the simplicial boundary.

        `(\delta f)(\sigma) = f(\partial \sigma)`.  Inserts one vertex
        at each valid position (which lowers homological degree by 1).
        """

        def terms():
            dim = len(simplex) - 1
            for j in range(dim + 2):
                if j == 0:
                    candidates = range(0, simplex[0])
                elif j == dim + 1:
                    candidates = range(simplex[-1] + 1, self._N + 1)
                else:
                    candidates = range(simplex[j - 1] + 1, simplex[j])
                for w in candidates:
                    augmented = simplex[:j] + (w,) + simplex[j:]
                    sign = (-1) ** (j % 2)
                    yield (augmented, sign)

        return self.sum_of_terms(terms())

    # -- volume form ---------------------------------------------------------
    @staticmethod
    def volume_form(N: int) -> "SimplicialCochains.Element":
        """The cochain evaluating to 1 on the fundamental chain of `Δ^N`."""
        return SimplicialCochains(N).term(tuple(range(N + 1)))

    # -- evaluation pairing -------------------------------------------------

    @staticmethod
    def evaluate(cochain, chain) -> "int | Rational":
        r"""Kronecker pairing `\langle f, \sigma \rangle`.

        Both *cochain* and *chain* share the same simplex-tuple basis.
        Returns `\sum_k f_k \sigma_k` over matching basis keys.
        """
        result = 0
        for k_f, v_f in cochain:
            for k_c, v_c in chain:
                if k_f == k_c:
                    result += v_f * v_c
        return result

    # -- basis enumeration --------------------------------------------------

    @staticmethod
    def dual_basis_it(N: int):
        """Iterate over all dual-basis cochains on ``Δ^N``."""
        parent = SimplicialCochains(N=N)
        for k in range(1, N + 2):
            for simplex in combinations(range(N + 1), k):
                yield parent.term(simplex)

    # -- Element class ------------------------------------------------------

    class Element(CombinatorialFreeModule.Element):
        """Elements of :class:`SimplicialCochains`."""

        def coboundary(self) -> "SimplicialCochains.Element":
            """Apply the coboundary δ."""
            return self.parent().coboundary(self)
