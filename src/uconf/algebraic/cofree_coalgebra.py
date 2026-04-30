"""Cofree conilpotent C-coalgebra on a dg-module M.

The cofree conilpotent C-coalgebra on a dg-module M is

    T^c_C(M) = ⊕_{n≥1} C(n) ⊗_{S_n} M^{⊗n}

with:
- Degree: deg(c_key, m_tuple) = deg_C(c_key) + Σ_i deg_M(m_i).
- Differential d = d_C + d_M from the Koszul sign rule (Leibniz rule).
- C-coalgebra costructure: the infinitesimal cocomposition Δ^{i;m,n} applies
  the cooperad's cocomposition to the C-decoration and splits the M-labels.

The basis keys are pairs ``(c_key, m_tuple)`` where:

- ``c_key`` is a **planar** basis key of ``C(n)`` for ``n = len(m_tuple) >= 1``.
- ``m_tuple`` is a tuple of ``n`` values.

**Quasi-planar requirement**: C must be a quasi-planar cooperad, i.e. each
component ``C(n)`` satisfies ``C(n) ≅ C_pl(n) ⊗ k[S_n]`` and exposes a
``planarize`` linear map.  The basis uses only planar C-keys, so two corollas
``(c·σ, m_tuple)`` and ``(c, σ·m_tuple)`` are identified as the same element
(coinvariant quotient).

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
    cached_method,
    tensor,
)

from uconf.algebraic.coalgebra import CooperadCoalgebra
from uconf.algebraic.tree_module import (
    _inner_weight_on_key,
    _module_basis_keys_in_degree,
    _module_basis_keys_in_weight_and_degree,
    _tuples_in_degree,
    _tuples_in_degree_and_weight,
)
from uconf.core.display import latex_linear_combination
from uconf.core.signs import sign_from_exponent
from uconf.core.signs import koszul_sign_of_permutation
from uconf.core.vertex_decoration import QuasiPlanarLike


class CofreeCoalgebraModule(CombinatorialFreeModule):
    """Underlying dg-module of the cofree conilpotent C-coalgebra ``T^c_C(M)``.

    Basis keys are ``(c_key, m_tuple)`` pairs where ``c_key`` is a **planar**
    basis key of ``C(n)`` and ``m_tuple`` has length ``n``.  The differential
    is the Leibniz rule ``d = d_C + d_M`` with Koszul signs.  Non-planar keys
    are automatically normalised via ``planarize`` (permuting the m_tuple).

    The cooperad ``C`` must be quasi-planar: each component ``C(n)`` must
    expose a ``planarize`` linear map decomposing elements into planar
    representative ⊗ symmetric group element.

    This class is normally not instantiated directly; use
    :class:`CofreeConilpotentCoalgebra` instead.
    """

    name: ClassVar[str] = "T^c"

    def __init__(
        self,
        cooperad_cls: QuasiPlanarLike,
        inner_module: CombinatorialFreeModule,
        *,
        name: str | None = None,
    ):
        """Initialize the cofree conilpotent C-coalgebra module ``T^c_C(M)``.

        Args:
            cooperad_cls: Arity-indexed **quasi-planar** cooperad provider.
            inner_module: Cogenerating dg-module M.
            name: Display name override.  Defaults to ``T^c_C(M)``.

        Raises:
            TypeError: If the cooperad is not quasi-planar (no ``planarize``).

        """
        if name is None:
            name = f"T^c_{cooperad_cls.name}({inner_module})"
        self._cooperad_cls = cooperad_cls
        self._inner_module = inner_module
        base_ring = inner_module.base_ring()
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
        """Return the sum of canonical corollas for ``c_elem ⊗ m_tuple``.

        For each basis term ``(c_key, coeff)`` of ``c_elem ∈ C(n)``, applies
        ``planarize`` to obtain ``Σ (c_planar_key ⊗ σ) * c`` and returns::

            Σ coeff * c * ε(leaf_perm; degrees) * self.term((c_planar_key, permute(m_tuple)))

        Here ``permute(m_tuple)`` is the leaf-tensor reordering induced by
        ``planarize``.  By default it is the usual left action of ``σ``:

            ``σ · (m_1, ..., m_n) = (m_{σ^{-1}(1)}, ..., m_{σ^{-1}(n)})``

        but cooperad components may override this convention via
        ``_leaf_tensor_permutation_from_planarize`` when their planarization
        stores the global leaf relabeling differently.  ``ε(leaf_perm;
        degrees)`` is the Koszul sign for the resulting graded leaf action.

        This ensures all stored C-keys are in the planar basis.

        Args:
            c_elem: An element of ``C(n)`` (any arity, may be non-planar).
            m_tuple: A tuple of ``n`` objects (basis keys or module elements).

        Returns:
            An element of ``self`` with planar C-keys.
        """
        n = len(m_tuple)
        comp = c_elem.parent()
        M = self._inner_module
        identity_tuple = tuple(range(1, n + 1))
        if n > 1:
            degrees = [M.degree_on_basis(m_tuple[j]) for j in range(n)]
        result = self.zero()
        for c_key, c_coeff in c_elem:
            planarized = comp.planarize(comp.term(c_key))
            for (c_planar_key, sigma_key), pl_coeff in planarized:
                sigma_tuple = tuple(int(s) for s in sigma_key)
                permute_leaf_tuple = getattr(comp, "_leaf_tensor_permutation_from_planarize", None)
                if callable(permute_leaf_tuple):
                    leaf_perm = tuple(int(s) for s in permute_leaf_tuple(sigma_tuple))
                else:
                    sigma_inv = [0] * n
                    for pos, value in enumerate(sigma_tuple, start=1):
                        sigma_inv[value - 1] = pos
                    leaf_perm = tuple(sigma_inv)
                permuted_m = tuple(m_tuple[leaf_perm[i] - 1] for i in range(n))
                # Koszul sign for permuting graded leaf-module elements.
                if n > 1 and leaf_perm != identity_tuple:
                    perm_0idx = [s - 1 for s in leaf_perm]
                    koszul = koszul_sign_of_permutation(perm_0idx, degrees)
                else:
                    koszul = 1
                result += c_coeff * pl_coeff * koszul * self.term((c_planar_key, permuted_m))
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
            raise TypeError(
                f"Basis key must be a tuple/list of length 2: (c_key, m_tuple). Got {key!r}."
            )
        c_key_raw, m_tuple_raw = key[0], key[1]
        if not isinstance(m_tuple_raw, (tuple, list)):
            raise TypeError(f"m_tuple must be a tuple/list of leaf keys. Got {m_tuple_raw!r}.")
        n = len(m_tuple_raw)
        if n == 0:
            return None

        # Validate c_key against C(n)
        # NOTE: Component construction C(n, R) must never fail—all cooperads
        # support every non-negative arity.  Any exception here is a real bug
        # and should propagate, not be silently ignored.
        comp = self._cooperad_cls(n, self.base_ring())
        validate_fn = getattr(comp, "_validate_basis_key", None)
        if validate_fn is not None:
            c_key = validate_fn(c_key_raw)
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

        raise TypeError(
            f"Cannot construct element from {x!r}. Expected a dict of basis keys to coefficients, or a single basis key tuple (c_key, m_tuple)."
        )

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

    def _repr_term(self, basis_element) -> str:
        """Readable basis-term notation for ``T^c_C(M)`` corollas."""
        c_key, m_tuple = basis_element
        n = len(m_tuple)
        c_parent = self._cooperad_cls(n, self.base_ring())

        c_repr = getattr(c_parent, "_repr_term", None)
        m_repr = getattr(self._inner_module, "_repr_term", None)

        c_str = c_repr(c_key) if callable(c_repr) else str(c_key)
        leaves = [m_repr(mk) if callable(m_repr) else str(mk) for mk in m_tuple]
        return f"<{c_str}; {', '.join(leaves)}>"

    def _latex_term(self, basis_element) -> str:
        """LaTeX basis-term notation for ``T^c_C(M)`` corollas."""
        c_key, m_tuple = basis_element
        n = len(m_tuple)
        c_parent = self._cooperad_cls(n, self.base_ring())

        c_repr = getattr(c_parent, "_latex_term", None)
        m_repr = getattr(self._inner_module, "_latex_term", None)

        c_ltx = c_repr(c_key) if callable(c_repr) else str(c_key)
        leaves = [m_repr(mk) if callable(m_repr) else str(mk) for mk in m_tuple]
        return f"\\langle {c_ltx}; {', '.join(leaves)} \\rangle"

    # ------------------------------------------------------------------
    # Differential
    # ------------------------------------------------------------------

    @cached_method
    def _boundary_on_basis(self, key) -> Any:
        r"""Leibniz rule: d(c ⊗ m_1 ⊗ … ⊗ m_n) = d_C(c) ⊗ m_… + Σ_i (−1)^{…} c ⊗ … ⊗ d_M(m_i) ⊗ ….

        Koszul sign at leaf i: ``(-1)^{deg_C(c_key) + sum_{j<i} deg_M(m_j)}``.

        The d_C output may contain non-planar cooperad keys; these are
        normalised to planar keys via ``_normalized_corolla_sum`` (coinvariant
        quotient).  The d_M terms keep the (already planar) cooperad key
        unchanged.
        """
        c_key, m_tuple = key
        n = len(m_tuple)
        comp = self._cooperad_cls(n, self.base_ring())
        result = self.zero()

        # d_C term: normalise non-planar keys via _normalized_corolla_sum.
        dc_on_basis = getattr(comp, "_boundary_on_basis", None)
        if dc_on_basis is not None:
            dc_elem = dc_on_basis(c_key)
        else:
            dc_elem = comp.boundary(comp.term(c_key))
        result += self._normalized_corolla_sum(dc_elem, m_tuple)

        # d_M terms with Koszul signs.
        # The cooperad key c_key is already planar (from basis iteration),
        # so no normalisation is needed — use self.term() directly.
        # Bypass morphism overhead: call the inner module's boundary
        # on_basis function directly instead of going through
        # morphism.__call__ + linear_combination.
        c_deg = comp.degree_on_basis(c_key)
        cumulative = c_deg
        M = self._inner_module
        m_bdry = M.boundary
        m_bdry_on_key = getattr(m_bdry, "on_basis", None)
        if m_bdry_on_key is not None:
            m_bdry_on_key = m_bdry_on_key()
        for i, mk in enumerate(m_tuple):
            sign = sign_from_exponent(cumulative)
            if m_bdry_on_key is not None:
                dm = m_bdry_on_key(mk)
            else:
                dm = m_bdry(M.term(mk))
            for new_mk, m_coeff in dm:
                new_m = m_tuple[:i] + (new_mk,) + m_tuple[i + 1 :]
                result += sign * m_coeff * self.term((c_key, new_m))
            cumulative += M.degree_on_basis(mk)

        return result

    # ------------------------------------------------------------------
    # Basis iteration
    # ------------------------------------------------------------------

    def basis_iter(self, d: int) -> Iterator[Any]:
        """Iterate over planar basis elements of total degree ``d``.

        Each yielded element is a single planar term ``self.term((c_pl, m))``
        indexed by a **planar** cooperad key ``c_pl`` and m-tuple ``m``.

        Uses the isomorphism ``C(n) ⊗_{S_n} M^{⊗n} ≅ C_pl(n) ⊗ M^{⊗n}``
        and enumerates only planar C(n)-decorations.

        Raises:
            ValueError: when arity is unbounded.
        """
        M = self._inner_module
        C = self._cooperad_cls
        R = self.base_ring()

        connectivity = int(getattr(C, "connectivity", 0))

        # n = 1: C(1) is the unit, degree 0 → need module keys at degree d
        c_key_1 = C.unit_key()
        for mk in _module_basis_keys_in_degree(M, d):
            yield self.term((c_key_1, (mk,)))

        # Determine max arity for n >= 2.
        # Need: d_c + d_m = d with d_c >= connectivity*(n-1), d_m >= 0
        # So d >= connectivity*(n-1), giving n <= d/connectivity + 1 (if connectivity > 0)
        # or n <= d - connectivity*(n-1) / min_m_deg (general case).
        # We need a finite bound — either positive cooperad connectivity or positive
        # minimum module degree provides one.
        m_conn = int(getattr(M, "connectivity", 0))
        if m_conn > 0:
            # Each leaf contributes at least m_conn to the total module degree.
            # d_m >= n * m_conn, so n <= (d - min_d_c) / m_conn.
            # For n >= 2, min_d_c = connectivity*(n-1).
            # Solving: n <= d / (m_conn + max(0, -connectivity)) + 2 (generous bound).
            max_n = max(1, (d + 2 * abs(connectivity)) // max(1, m_conn) + 2)
        elif connectivity > 0:
            max_n = d // connectivity + 1
        else:
            raise ValueError(
                "Cannot exhaustively enumerate basis_iter(d): both C and M admit "
                "degree-0 generators (connectivity=0, min_m_deg=0)."
            )

        # Collect inner-module keys at degrees 0..max_possible_d_m.
        # d_m = d - d_c, and d_c can be as low as connectivity*(n-1) < 0 for
        # negative-connectivity cooperads, so max_d_m can exceed d.
        if connectivity < 0:
            max_d_m = d - connectivity * max(1, max_n - 1)
        else:
            max_d_m = d

        m_keys_by_deg: dict[int, list] = {}
        for d_m in range(max(0, max_d_m) + 1):
            keys = list(_module_basis_keys_in_degree(M, d_m))
            if keys:
                m_keys_by_deg[d_m] = keys

        if not m_keys_by_deg:
            return

        for n in range(2, max_n + 1):
            comp_n = C(n, R)
            min_d_c = connectivity * (n - 1) if connectivity < 0 else 0
            for d_c in range(min_d_c, d + 1):
                d_m_needed = d - d_c
                if d_m_needed < 0:
                    continue
                c_elems = list(
                    getattr(comp_n, "graded_planar_basis", comp_n.planar_basis_iter)(d_c)
                )
                if not c_elems:
                    continue
                m_tuples = list(_tuples_in_degree(m_keys_by_deg, n, d_m_needed))
                if not m_tuples:
                    continue
                for c_elem in c_elems:
                    for c_key in c_elem.support():
                        for m_tuple in m_tuples:
                            yield self.term((c_key, m_tuple))

    def _weight_on_basis(self, key) -> int:
        """Weight of a basis key ``(c_key, m_tuple)``.

        The weight is the sum of weights of the leaf-module elements in
        ``m_tuple``.  Each leaf element's weight is obtained from the inner
        module's ``_weight_on_basis``; modules without this attribute
        default to weight 1 per key.
        """
        _c_key, m_tuple = key
        return sum(_inner_weight_on_key(self._inner_module, m) for m in m_tuple)

    def basis_weight_iter(self, d: int, w: int) -> Iterator[Any]:
        """Iterate over planar basis elements of degree ``d`` and weight ``w``.

        Each yielded element is a single planar term ``self.term((c_pl, m))``.
        Weight is additive over leaf-module elements.

        Unlike :meth:`basis_iter`, this method is always finite (``w``
        bounds the arity).
        """
        if w < 1:
            return

        M = self._inner_module
        C = self._cooperad_cls
        R = self.base_ring()

        connectivity = int(getattr(C, "connectivity", 0))

        # Max arity: each leaf has weight >= 1, so n <= w.
        max_n = w

        # Maximum module degree: d_m = d - d_c, and d_c can be as low as
        # connectivity*(n-1) for the largest arity n.
        min_d_c = connectivity * max(1, max_n - 1) if connectivity < 0 else 0
        max_d_m = max(0, d - min_d_c)

        # Collect inner-module keys by (degree, weight)
        keys_by_dw: dict[tuple, list] = {}
        for d_m in range(max_d_m + 1):
            for w_m in range(1, w + 1):
                keys = list(_module_basis_keys_in_weight_and_degree(M, d_m, w_m))
                if keys:
                    keys_by_dw[(d_m, w_m)] = keys

        # n = 1: single leaf
        id_key = C.unit_key()
        for mk in keys_by_dw.get((d, w), []):
            yield self.term((id_key, (mk,)))

        # n >= 2
        for n in range(2, max_n + 1):
            comp_n = C(n, R)
            min_dc_n = connectivity * (n - 1) if connectivity < 0 else 0
            for d_c in range(min_dc_n, d + 1):
                d_m_needed = d - d_c
                if d_m_needed < 0:
                    continue
                c_elems = list(
                    getattr(comp_n, "graded_planar_basis", comp_n.planar_basis_iter)(d_c)
                )
                if not c_elems:
                    continue
                m_tuples = list(_tuples_in_degree_and_weight(keys_by_dw, n, d_m_needed, w))
                if not m_tuples:
                    continue
                for c_elem in c_elems:
                    for c_key in c_elem.support():
                        for m_tuple in m_tuples:
                            yield self.term((c_key, m_tuple))

    @cached_method
    def graded_basis(self, d: int):
        return Family(self.basis_iter(d))

    @cached_method
    def graded_basis_by_weight(self, d: int, w: int):
        """Return a :class:`Family` of basis elements at degree ``d`` and weight ``w``."""
        return Family(self.basis_weight_iter(d, w))

    # ------------------------------------------------------------------
    # Element class
    # ------------------------------------------------------------------

    class Element(CombinatorialFreeModule.Element):
        """An element of the cofree C-coalgebra module ``T^c_C(M)``."""

        def _repr_latex_(self) -> str:
            return latex_linear_combination(self, lambda basis: self.parent()._latex_term(basis))

        def boundary(self) -> "CofreeCoalgebraModule.Element":
            """Apply the differential d = d_C + d_M."""
            return self.parent().boundary(self)


class CofreeConilpotentCoalgebra(CooperadCoalgebra):
    """Cofree conilpotent C-coalgebra on a dg-module M.

    Constructs ``T^c_C(M) = sum_{n>=1} C(n) x M^{xn}`` as a
    :class:`CofreeCoalgebraModule` and equips it with the canonical
    C-coalgebra coaction.

    Args:
        cooperad_cls: Quasi-planar cooperad provider C.
        inner_module: The cogenerating dg-module M.

    The coprojection ``pi: T^c_C(M) -> M`` is given by ``project()``.

    Examples::

        cofree_coass = CofreeConilpotentCoalgebra(CoAssociative, module_M)
        elem = cofree_coass.modul(((1, 2), (m1, m2)))
        cofree_coass.coact(elem, 2)

    """

    def __init__(self, cooperad_cls: QuasiPlanarLike, inner_module):
        cofree_module = CofreeCoalgebraModule(cooperad_cls, inner_module)
        super().__init__(cofree_module, cooperad_cls, self._coact_impl)
        self._inner_module = inner_module
        self._base_ring = inner_module.base_ring()

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
        comp_1_list = list(comp_1.basis_iter(0))
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
            coop_elem = coop_parent(c_key)
            leaf_elems = [cofree_mod((id_key, (mk,))) for mk in m_tuple]
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

        Both factors are constructed directly (no coinvariant planarization).

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
            c_elem = comp(c_key)
            cocomp = comp.infinitesimal_cocompose(c_elem, i, m, n)

            # Split M-labels (1-indexed positions):
            # Left (arity m): mu_1..mu_i, mu_{i+n}..mu_{m+n-1}
            left_m = m_tuple[:i] + m_tuple[i + n - 1 :]
            # Right (arity n): mu_i..mu_{i+n-1}
            right_m = m_tuple[i - 1 : i + n - 1]

            for (c_L_key, c_R_key), tensor_coeff in cocomp:
                left_elem = cofree_mod((c_L_key, left_m))
                right_elem = cofree_mod((c_R_key, right_m))
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
                result += coeff * inner(m_tuple[0])
        return result
