"""Cofree conilpotent C-coalgebra on a dg-module M.

The cofree conilpotent C-coalgebra on a dg-module M is

    T^c_C(M) = \u2295_{n\u22651} C(n) \u2297_{S_n} M^{\u2297n}

with:
- Degree: deg(c_key, m_tuple) = deg_C(c_key) + \u03a3_i deg_M(m_i).
- Differential d = d_C + d_M from the Koszul sign rule (Leibniz rule).
- C-coalgebra costructure: the infinitesimal cocomposition \u0394^{i;m,n} applies
  the cooperad\'s cocomposition to the C-decoration and splits the M-labels.

**Quasi-planar requirement**: C must be a quasi-planar cooperad, i.e. each
component ``C(n)`` exposes a ``planarize`` linear map.  Non-planar C-keys are
normalised by permuting m_tuple accordingly.

The basis keys are pairs ``(c_key, m_tuple)`` where:

- ``c_key`` is a **planar** basis key of ``C(n)`` for ``n = len(m_tuple) >= 1``.
- ``m_tuple`` is a tuple of ``n`` values.

The coprojection pi: T^c_C(M) -> M kills all elements with n >= 2 and maps
``(id_key, (m,)) |-> m``.

Reference: Loday-Vallette "Algebraic Operads", Section 5.8.
"""

from __future__ import annotations

from typing import Any, ClassVar, Iterator

from sage.all import (
    CombinatorialFreeModule,
    Family,
    GradedModulesWithBasis,
    SymmetricGroup,
    cached_method,
    tensor,
)

from uconf.algebraic.coalgebra import CooperadCoalgebra
from uconf.algebraic.tree_module import _module_basis_keys_in_degree, _tuples_in_degree
from uconf.core.signs import sign_from_exponent
from uconf.core.vertex_decoration import QuasiPlanarLike


class CofreeCoalgebraModule(CombinatorialFreeModule):
    """Underlying dg-module of the cofree conilpotent C-coalgebra ``T^c_C(M)``.

    Basis keys are ``(c_key, m_tuple)`` pairs where ``c_key`` is a **planar**
    basis key of ``C(n)`` and ``m_tuple`` has length ``n``.  The differential
    is the Leibniz rule ``d = d_C + d_M`` with Koszul signs.  Non-planar keys
    are automatically normalised via ``planarize`` (permuting the m_tuple).

    The cooperad ``C`` must be quasi-planar: each component ``C(n)`` must
    expose a ``planarize`` linear map.

    This class is normally not instantiated directly; use
    :class:`CofreeConilpotentCoalgebra` instead.
    """

    name: ClassVar[str] = "T^c"

    def __init__(
        self,
        cooperad_cls: QuasiPlanarLike,
        inner_module: CombinatorialFreeModule,
        base_ring,
        *,
        name: str | None = None,
    ):
        """Initialize the cofree conilpotent C-coalgebra module ``T^c_C(M)``.

        Args:
            cooperad_cls: Arity-indexed **quasi-planar** cooperad provider.
            inner_module: Cogenerating dg-module M.
            base_ring: Coefficient ring.
            name: Display name override.  Defaults to ``T^c_C(M)``.

        Raises:
            TypeError: If the cooperad is not quasi-planar (no ``planarize``).

        """
        if name is None:
            name = f"T^c_{cooperad_cls.name}({inner_module})"
        self._cooperad_cls = cooperad_cls
        self._inner_module = inner_module
        # Runtime check: cooperad must be quasi-planar (free S_n-action)
        _comp2 = cooperad_cls(2, base_ring)
        if not callable(getattr(_comp2, "planarize", None)):
            raise TypeError(
                f"Cooperad {cooperad_cls.name!r} is not quasi-planar: "
                f"its arity-2 component does not expose 'planarize'.  "
                "Only quasi-planar cooperads (CoAssociative, SurjectionDual, "
                "and their wrappers) are accepted."
            )
        super().__init__(
            base_ring,
            tuple,
            prefix=name,
            category=GradedModulesWithBasis(base_ring),
        )
        self.rename(name)
        self.boundary = self.module_morphism(on_basis=self._boundary_on_basis, codomain=self)

    # ------------------------------------------------------------------
    # Normalisation helper
    # ------------------------------------------------------------------

    def _normalized_corolla_sum(self, c_elem, m_tuple) -> "CofreeCoalgebraModule.Element":
        """Return the sum of canonical corollas for ``c_elem x m_tuple``.

        For each basis term ``(c_key, coeff)`` of ``c_elem in C(n)``, applies
        ``planarize`` to obtain ``(c_planar_key x sigma) * cf`` and returns::

            sum coeff * cf * self.term((c_planar_key, sigma . m_tuple))

        where ``sigma . m_tuple = (m_tuple[sigma^{-1}(1)-1], ..., m_tuple[sigma^{-1}(n)-1])``.

        Args:
            c_elem: An element of ``C(n)`` (any arity, may be non-planar).
            m_tuple: A tuple of ``n`` objects.

        Returns:
            An element of ``self`` with planar C-keys.

        """
        n = len(m_tuple)
        comp = c_elem.parent()
        S_n = SymmetricGroup(n)
        result = self.zero()
        for c_key, c_coeff in c_elem:
            planarized = comp.planarize(comp.term(c_key))
            for (c_planar_key, sigma_key), pl_coeff in planarized:
                sigma = S_n(sigma_key)
                sigma_inv = sigma.inverse()
                permuted_m = tuple(m_tuple[sigma_inv(i) - 1] for i in range(1, n + 1))
                result += c_coeff * pl_coeff * self.term((c_planar_key, permuted_m))
        return result

    # ------------------------------------------------------------------
    # Basis key validation
    # ------------------------------------------------------------------

    def _validate_basis_key(self, key) -> tuple | None:
        """Validate a ``(c_key, m_tuple)`` basis key (structural check only).

        Returns the normalised key, or ``None`` if structurally invalid.
        Does **not** planarize; use :meth:`_normalized_corolla_sum` for that.
        """
        if not isinstance(key, (tuple, list)) or len(key) != 2:
            return None
        c_key_raw, m_tuple_raw = key[0], key[1]
        if not isinstance(m_tuple_raw, (tuple, list)):
            return None
        n = len(m_tuple_raw)
        if n == 0:
            return None

        # Validate c_key against C(n)
        # NOTE: Component construction C(n, R) must never fail—all cooperads
        # support every non-negative arity.  Any exception here is a real bug
        # and should propagate, not be silently ignored.
        comp = self._cooperad_cls(n, self.base_ring())
        if hasattr(comp, "_validate_basis_key"):
            try:
                c_key = comp._validate_basis_key(c_key_raw)
            except (TypeError, ValueError):
                # The c_key is invalid for arity n (wrong type, wrong length,
                # out-of-range values, etc.); treat the composite key as invalid.
                return None
            if c_key is None:
                return None
        else:
            c_key = c_key_raw

        return (c_key, tuple(m_tuple_raw))

    def _element_constructor_(self, x):
        if isinstance(x, dict):
            result = self.zero()
            for key, coeff in x.items():
                k = self._validate_basis_key(key)
                if k is not None:
                    c_key_raw, m_tuple_raw = k
                    n = len(m_tuple_raw)
                    comp = self._cooperad_cls(n, self.base_ring())
                    result += coeff * self._normalized_corolla_sum(
                        comp.term(c_key_raw), m_tuple_raw
                    )
            return result

        if isinstance(x, (tuple, list)) and len(x) == 2:
            k = self._validate_basis_key(x)
            if k is None:
                return self.zero()
            c_key_raw, m_tuple_raw = k
            n = len(m_tuple_raw)
            comp = self._cooperad_cls(n, self.base_ring())
            return self._normalized_corolla_sum(comp.term(c_key_raw), m_tuple_raw)

        return super()._element_constructor_(x)

    # ------------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------------

    @property
    def connectivity(self) -> int:
        """Minimum degree of any basis element.

        The arity-1 term ``(id, (m,))`` has degree ``deg_M(m)``, so the
        connectivity equals that of the inner module.
        """
        return int(getattr(self._inner_module, "connectivity", 0))

    # ------------------------------------------------------------------
    # Degree
    # ------------------------------------------------------------------

    def degree_on_basis(self, key) -> int:
        """Degree = deg_C(c_key) + sum_i deg_M(m_i)."""
        c_key, m_tuple = key
        n = len(m_tuple)
        comp = self._cooperad_cls(n, self.base_ring())
        c_deg = comp.degree_on_basis(c_key)
        m_deg = sum(self._inner_module.degree_on_basis(mk) for mk in m_tuple)
        return c_deg + m_deg

    # ------------------------------------------------------------------
    # Differential
    # ------------------------------------------------------------------

    def _boundary_on_basis(self, key) -> Any:
        """Leibniz rule: d(c x m_1 x...x m_n) = d_C(c) x m_... + sum_i (-1)^{...} c x...x d_M(m_i) x....

        Koszul sign at leaf i: ``(-1)^{deg_C(c_key) + sum_{j<i} deg_M(m_j)}``.
        Non-planar keys produced by ``d_C`` are normalised via planarize.
        """
        c_key, m_tuple = key
        n = len(m_tuple)
        comp = self._cooperad_cls(n, self.base_ring())
        result = self.zero()

        # d_C term: normalise via planarize since boundary may produce non-planar keys
        dc_elem = comp.boundary(comp.term(c_key))
        if dc_elem:
            result += self._normalized_corolla_sum(dc_elem, m_tuple)

        # d_M terms with Koszul signs
        c_deg = comp.degree_on_basis(c_key)
        cumulative = c_deg
        for i, mk in enumerate(m_tuple):
            sign = sign_from_exponent(cumulative)
            m_elem = self._inner_module.term(mk)
            for new_mk, m_coeff in self._inner_module.boundary(m_elem):
                new_m = m_tuple[:i] + (new_mk,) + m_tuple[i + 1 :]
                result += sign * m_coeff * self.term((c_key, new_m))
            cumulative += self._inner_module.degree_on_basis(mk)

        return result

    # ------------------------------------------------------------------
    # Basis iteration
    # ------------------------------------------------------------------

    def basis_it(self, d: int) -> Iterator[Any]:
        """Iterate over basis elements of total degree ``d``.

        Uses the isomorphism ``C(n) x_{S_n} M^{xn} ~= C_pl(n) x M^{xn}``
        and enumerates only planar C(n)-decorations.

        Raises:
            ValueError: when arity is unbounded.
        """
        M = self._inner_module
        C = self._cooperad_cls
        R = self.base_ring()

        m_keys_by_deg: dict[int, list] = {}
        for d_m in range(d + 1):
            keys = list(_module_basis_keys_in_degree(M, d_m))
            if keys:
                m_keys_by_deg[d_m] = keys

        # n = 1
        comp_1 = C(1, R)
        comp_1_list = list(comp_1.basis_it(0))
        assert len(comp_1_list) == 1, (
            f"C(1) must have exactly one basis element. Got {len(comp_1_list)}."
        )
        c_key_1 = C.unit_key()
        for mk in m_keys_by_deg.get(d, []):
            yield self.term((c_key_1, (mk,)))

        if not m_keys_by_deg:
            return

        min_m_deg = min(m_keys_by_deg.keys())
        connectivity = getattr(C, "connectivity", 0)

        if d < connectivity:
            return

        if min_m_deg > 0:
            max_n = d // min_m_deg
        elif connectivity > 0:
            max_n = d // connectivity + 1
        else:
            raise ValueError(
                "Cannot exhaustively enumerate basis_it(d): both C and M admit "
                "degree-0 generators (connectivity=0, min_m_deg=0)."
            )

        for n in range(2, max_n + 1):
            comp_n = C(n, R)
            for d_c in range(d + 1):
                d_m_needed = d - d_c
                if d_m_needed < 0:
                    continue
                c_elems = list(comp_n.planar_basis_it(d_c))
                if not c_elems:
                    continue
                m_tuples = list(_tuples_in_degree(m_keys_by_deg, n, d_m_needed))
                if not m_tuples:
                    continue
                for c_elem in c_elems:
                    for c_key in c_elem.support():
                        for m_tuple in m_tuples:
                            yield self.term((c_key, m_tuple))

    @cached_method
    def graded_basis(self, d: int):
        return Family(self.basis_it(d))

    # ------------------------------------------------------------------
    # Element class
    # ------------------------------------------------------------------

    class Element(CombinatorialFreeModule.Element):
        """An element of the cofree C-coalgebra module ``T^c_C(M)``."""

        def boundary(self) -> "CofreeCoalgebraModule.Element":
            """Apply the differential d = d_C + d_M."""
            return self.parent().boundary(self)


class CofreeConilpotentCoalgebra(CooperadCoalgebra):
    """Cofree conilpotent C-coalgebra on a dg-module M.

    Constructs ``T^c_C(M) = sum_{n>=1} C(n) x_{S_n} M^{xn}`` as a
    :class:`CofreeCoalgebraModule` and equips it with the canonical
    C-coalgebra coaction.

    **Quasi-planar requirement**: C must be quasi-planar.

    Args:
        cooperad_cls: Quasi-planar cooperad provider C.
        inner_module: The cogenerating dg-module M.
        base_ring: Coefficient ring.

    The coprojection ``pi: T^c_C(M) -> M`` is given by ``project()``.

    Examples::

        cofree_coass = CofreeConilpotentCoalgebra(CoAssociative, module_M, QQ)
        elem = cofree_coass.module.term(((1, 2), (m1, m2)))
        cofree_coass.coact(elem, 2)

    """

    def __init__(self, cooperad_cls: QuasiPlanarLike, inner_module, base_ring):
        cofree_module = CofreeCoalgebraModule(cooperad_cls, inner_module, base_ring)
        super().__init__(cofree_module, cooperad_cls, self._coact_impl)
        self._inner_module = inner_module
        self._base_ring = base_ring

    def _coact_impl(self, v_element, n: int):
        """C-coalgebra coaction delta_n on ``T^c_C(M)``.

        For each basis element ``(c_key, (m_1, ..., m_k))`` with ``k == n``:

            delta_n((c_key, (m_1, ..., m_n))) =
                c_key x (id_key, (m_1,)) x ... x (id_key, (m_n,))

        where ``id_key`` is the unique basis key of ``C(1)`` (the counit).
        For ``k != n``: delta_n = 0.

        Returns an element of ``C(n) x T^c_C(M)^{xn}``.
        """
        base_ring = self._base_ring
        coop_parent = self.cooperad_cls(n, base_ring)
        cofree_mod = self.module

        # Get identity key of C(1)
        comp_1 = self.cooperad_cls(1, base_ring)
        comp_1_list = list(comp_1.basis_it(0))
        assert len(comp_1_list) == 1, (
            f"C(1) must have exactly one basis element. Got {len(comp_1_list)}."
        )
        id_key = self.cooperad_cls.unit_key()

        right_factors = [cofree_mod] * n
        target = tensor([coop_parent] + right_factors)
        result = target.zero()

        for (c_key, m_tuple), v_coeff in v_element:
            k = len(m_tuple)
            if k != n:
                continue  # arity mismatch

            # Build flat tensor: c_key x (id_key,(m_1,)) x ... x (id_key,(m_n,))
            coop_elem = coop_parent.term(c_key)
            leaf_elems = [cofree_mod.term((id_key, (mk,))) for mk in m_tuple]
            term = tensor([coop_elem] + leaf_elems)
            result += v_coeff * term

        return result

    def infinitesimal_cocompose(self, x, i: int, m: int, n: int):
        """Delta^{i;m,n}: T^c_C(M)(m+n-1) -> T^c_C(M)(m) x T^c_C(M)(n).

        For each basis element ``(c_key, (mu_1, ..., mu_{m+n-1}))``,
        applies the cooperad's infinitesimal cocomposition and distributes
        M-labels:

        - Left factor (arity m):  ``(c_L_key, (mu_1,...,mu_{i-1}, mu_{i+n-1},...,mu_{m+n-1}))``
          with mu_i replaced by a placeholder at position i, corresponding to the
          right subtree occupying positions i..i+n-1.
        - Right factor (arity n): ``(c_R_key, (mu_i,...,mu_{i+n-1}))``

        Both factors are normalised via planarize.

        Args:
            x: An element of the cofree module.
            i: Starting position of the right factor (1-indexed).
            m: Arity of the left factor.
            n: Arity of the right factor.

        Returns:
            An element of ``T^c_C(M)(m) x T^c_C(M)(n)``.

        """
        if m <= 0 or n <= 0:
            raise ValueError(f"Arities must be positive. Got m={m}, n={n}.")
        if not (1 <= i <= m):
            raise ValueError(f"Index i must satisfy 1 <= i <= {m}. Got i={i}.")

        cofree_mod = self.module
        target = tensor([cofree_mod, cofree_mod])
        result = target.zero()
        base_ring = self._base_ring

        for (c_key, m_tuple), coeff in x:
            k = len(m_tuple)
            if k != m + n - 1:
                continue

            comp = self.cooperad_cls(k, base_ring)
            c_elem = comp.term(c_key)
            cocomp = comp.infinitesimal_cocompose(c_elem, i, m, n)

            # Split M-labels (1-indexed positions):
            # Left (arity m): mu_1..mu_i, mu_{i+n}..mu_{m+n-1}
            left_m = m_tuple[:i] + m_tuple[i + n - 1 :]
            # Right (arity n): mu_i..mu_{i+n-1}
            right_m = m_tuple[i - 1 : i + n - 1]

            for (c_L_key, c_R_key), tensor_coeff in cocomp:
                # Normalise each factor via planarize
                left_comp = self.cooperad_cls(m, base_ring)
                right_comp = self.cooperad_cls(n, base_ring)
                left_elem = cofree_mod._normalized_corolla_sum(left_comp.term(c_L_key), left_m)
                right_elem = cofree_mod._normalized_corolla_sum(right_comp.term(c_R_key), right_m)
                result += coeff * tensor_coeff * left_elem.tensor(right_elem)

        return result

    def project(self, x):
        """Coprojection pi: T^c_C(M) -> M, projecting onto arity-1 generators.

        Returns the image in the inner module M.  Non-zero only for elements
        of the form ``(id_key, (m_key,))`` where ``id_key in C(1)``.

        Args:
            x: An element of the cofree module.

        Returns:
            An element of the inner module M.

        """
        inner = self._inner_module
        result = inner.zero()

        # Get the unique C(1) identity key
        id_key = self.cooperad_cls.unit_key()

        for (c_key, m_tuple), coeff in x:
            if len(m_tuple) == 1 and c_key == id_key:
                result += coeff * inner.term(m_tuple[0])
        return result
