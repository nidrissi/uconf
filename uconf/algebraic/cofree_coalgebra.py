"""Cofree conilpotent C-coalgebra on a dg-module M.

The cofree conilpotent C-coalgebra on a dg-module M is

    T^c_C(M) = ⊕_{n≥1} C(n) ⊗_{S_n} M^{⊗n}

with:
- Degree: deg(c_key, m_tuple) = deg_C(c_key) + Σ_i deg_M(m_i).
- Differential d = d_C + d_M from the Koszul sign rule (Leibniz rule).
- C-coalgebra costructure: the infinitesimal cocomposition Δ^{i;m,n} applies
  the cooperad's cocomposition to the C-decoration and splits the M-labels.

The basis keys are pairs ``(c_key, m_tuple)`` where:

- ``c_key`` is a basis key of ``C(n)`` for ``n = len(m_tuple) ≥ 1``.
- ``m_tuple`` is a tuple of ``n`` basis keys of the inner module M.

The arity ``n`` is determined implicitly as ``len(m_tuple)``.  For ``n = 1``
the unique C(1)-key is the counit/identity.

The coprojection π: T^c_C(M) → M kills all elements with n ≥ 2 (i.e.
``len(m_tuple) ≥ 2``) and maps ``(id_key, (m,)) ↦ m``.

Reference: Loday-Vallette "Algebraic Operads", Section 5.8.
"""

from __future__ import annotations

from typing import Any, ClassVar, Iterator

from sage.all import CombinatorialFreeModule, Family, GradedModulesWithBasis, cached_method, tensor

from uconf.algebraic.coalgebra import CooperadCoalgebra
from uconf.algebraic.tree_module import _module_basis_keys_in_degree, _tuples_in_degree
from uconf.core.cooperad import CooperadLike
from uconf.core.signs import sign_from_exponent


class CofreeCoalgebraModule(CombinatorialFreeModule):
    """Underlying dg-module of the cofree conilpotent C-coalgebra ``T^c_C(M)``.

    Basis keys are ``(c_key, m_tuple)`` pairs, where ``c_key`` is a basis
    key of ``C(n)`` and ``m_tuple`` is an ``n``-tuple of M-basis keys with
    ``n = len(m_tuple)``.  The differential is the Leibniz rule
    ``d = d_C + d_M`` with Koszul signs.

    This class is normally not instantiated directly; use
    :class:`CofreeConilpotentCoalgebra` instead.
    """

    name: ClassVar[str] = "T^c"

    def __init__(
        self,
        cooperad_cls: CooperadLike,
        inner_module: CombinatorialFreeModule,
        base_ring,
        *,
        name: str | None = None,
    ):
        """Initialize the cofree conilpotent C-coalgebra module ``T^c_C(M)``.

        Args:
            cooperad_cls: Arity-indexed cooperad (or cooperad-like) provider.
            inner_module: Cogenerating dg-module M (a ``CombinatorialFreeModule``).
            base_ring: Coefficient ring.
            name: Display name override.  Defaults to ``T^c_C(M)``.

        """
        if name is None:
            name = f"T^c_{cooperad_cls.name}({inner_module})"
        self._cooperad_cls = cooperad_cls
        self._inner_module = inner_module
        super().__init__(
            base_ring,
            tuple,
            prefix=name,
            category=GradedModulesWithBasis(base_ring),
        )
        self.rename(name)
        self.boundary = self.module_morphism(on_basis=self._boundary_on_basis, codomain=self)

    # ------------------------------------------------------------------
    # Basis key validation
    # ------------------------------------------------------------------

    def _validate_basis_key(self, key) -> tuple | None:
        """Validate and normalise a ``(c_key, m_tuple)`` basis key.

        Returns the normalised key, or ``None`` if invalid.
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
        try:
            comp = self._cooperad_cls(n, self.base_ring())
            if hasattr(comp, "_validate_basis_key"):
                c_key = comp._validate_basis_key(c_key_raw)
                if c_key is None:
                    return None
            else:
                c_key = c_key_raw
        except (TypeError, ValueError, AttributeError):
            return None

        # Validate each m-key
        new_m = []
        for mk in m_tuple_raw:
            if hasattr(self._inner_module, "_validate_basis_key"):
                vk = self._inner_module._validate_basis_key(mk)
                if vk is None:
                    return None
                new_m.append(vk)
            else:
                new_m.append(mk)

        return (c_key, tuple(new_m))

    def _element_constructor_(self, x):
        if isinstance(x, dict):
            clean: dict = {}
            for key, coeff in x.items():
                k = self._validate_basis_key(key)
                if k is not None:
                    clean[k] = clean.get(k, 0) + coeff
            return self.sum_of_terms(clean.items())

        if isinstance(x, (tuple, list)) and len(x) == 2:
            k = self._validate_basis_key(x)
            if k is None:
                return self.zero()
            return self.term(k)

        return super()._element_constructor_(x)

    # ------------------------------------------------------------------
    # Degree
    # ------------------------------------------------------------------

    def degree_on_basis(self, key) -> int:
        """Degree = deg_C(c_key) + Σ_i deg_M(m_i)."""
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
        """Leibniz rule: d(c ⊗ m_1 ⊗…⊗ m_n) = d_C(c) ⊗ m_… + Σ_i (−1)^{…} c ⊗…⊗ d_M(m_i) ⊗….

        Koszul sign at leaf i: ``(−1)^{deg_C(c_key) + Σ_{j<i} deg_M(m_j)}``.
        """
        c_key, m_tuple = key
        n = len(m_tuple)
        comp = self._cooperad_cls(n, self.base_ring())
        result = self.zero()

        # d_C term
        c_elem = comp.term(c_key)
        for new_c_key, coeff in comp.boundary(c_elem):
            result += coeff * self.term((new_c_key, m_tuple))

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

        For a quasi-planar cooperad C, uses the isomorphism
        ``C(n) ⊗_{S_n} M^{⊗n} ≅ C_pl(n) ⊗ M^{⊗n}`` and enumerates only
        planar C(n)-decorations for ``n ≥ 2``.

        Raises:
            NotImplementedError: when C does not expose ``planar_basis_it``.
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
        c_key_1 = comp_1_list[0].support()[0]
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

        try:
            _use_planar = hasattr(C(2, R), "planar_basis_it")
        except (TypeError, ValueError, NotImplementedError, AttributeError):
            _use_planar = False

        if not _use_planar:
            raise NotImplementedError(
                f"basis_it() requires the cooperad {C.name!r} to support "
                "planar_basis_it() on its arity-2 component."
            )

        for n in range(2, max_n + 1):
            try:
                comp_n = C(n, R)
            except (TypeError, ValueError, AttributeError):
                continue
            for d_c in range(d + 1):
                d_m_needed = d - d_c
                if d_m_needed < 0:
                    continue
                try:
                    c_elems = list(comp_n.planar_basis_it(d_c))
                except (TypeError, ValueError, NotImplementedError, AttributeError):
                    continue
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

    Constructs ``T^c_C(M) = ⊕_{n≥1} C(n) ⊗_{S_n} M^{⊗n}`` as a
    :class:`CofreeCoalgebraModule` and equips it with the canonical
    C-coalgebra coaction and infinitesimal cocomposition.

    Args:
        cooperad_cls: Cooperad provider C (class or wrapper instance).
        inner_module: The cogenerating dg-module M.
        base_ring: Coefficient ring.

    The coprojection ``π: T^c_C(M) → M`` is given by ``project()``.

    The coaction ``δ_k`` is implemented via ``coact(v_elem, k)`` which returns
    an element of ``C(k) ⊗ T^c_C(M)^{⊗k}``.  The infinitesimal cocomposition
    ``Δ^{i;m,n}`` is accessible via ``infinitesimal_cocompose(x, i, m, n)``.

    Examples::

        cofree_coass = CofreeConilpotentCoalgebra(CoAssociative, module_M, QQ)
        # Coaction on a corolla element:
        elem = cofree_coass.module.term(((1, 2), (m1, m2)))
        cofree_coass.coact(elem, 2)   # splits at root

    """

    def __init__(self, cooperad_cls: CooperadLike, inner_module, base_ring):
        cofree_module = CofreeCoalgebraModule(cooperad_cls, inner_module, base_ring)
        super().__init__(cofree_module, cooperad_cls, self._coact_impl)
        self._inner_module = inner_module
        self._base_ring = base_ring

    def _coact_impl(self, v_element, n: int):
        """C-coalgebra coaction δ_n on ``T^c_C(M)``.

        For each basis element ``(c_key, (m_1, ..., m_k))`` with ``k == n``:

            δ_n((c_key, (m_1, ..., m_n))) =
                c_key ⊗ (id_key, (m_1,)) ⊗ … ⊗ (id_key, (m_n,))

        where ``id_key`` is the unique basis key of ``C(1)`` (the counit).

        For ``k ≠ n``: δ_n = 0.

        Returns an element of ``C(n) ⊗ T^c_C(M)^{⊗n}`` (a Sage tensor module element).
        """
        base_ring = self._base_ring
        coop_parent = self.cooperad_cls(n, base_ring)
        cofree_mod = self.module

        # Get identity key of C(1)
        comp_1 = self.cooperad_cls(1, base_ring)
        id_keys = list(comp_1.basis_it(0))
        assert len(id_keys) == 1, f"C(1) must have exactly one basis element. Got {len(id_keys)}."
        # C(1) has exactly one basis element
        id_key = id_keys[0].support()[0]

        right_factors = [cofree_mod] * n
        target = tensor([coop_parent] + right_factors)
        result = target.zero()

        for (c_key, m_tuple), v_coeff in v_element:
            k = len(m_tuple)
            if k != n:
                continue  # arity mismatch

            # Build tensor: c_key ⊗ (id_key,(m_1,)) ⊗ ... ⊗ (id_key,(m_n,))
            coop_elem = coop_parent(c_key)
            terms = [coop_elem]
            for mk in m_tuple:
                leaf_elem = cofree_mod((id_key, (mk,)))
                terms.append(leaf_elem)
            term = tensor(terms)

            result += v_coeff * term

        return result

    def infinitesimal_cocompose(self, x, i: int, m: int, n: int):
        """Δ^{i;m,n}: T^c_C(M)(m+n-1) → T^c_C(M)(m) ⊗ T^c_C(M)(n).

        For each basis element ``(c_key, (μ_1, ..., μ_{m+n-1}))`` with
        ``len(m_tuple) == m+n-1``, applies the cooperad's infinitesimal
        cocomposition ``C(m+n-1).infinitesimal_cocompose(c, i, m, n)`` to
        split the C-decoration and distributes the M-labels:

        - Left factor (arity m):  ``(c_L_key, (μ_1,...,μ_{i}, μ_{i+n},...,μ_{m+n-1}))``
        - Right factor (arity n): ``(c_R_key, (μ_i,...,μ_{i+n-1}))``

        Note that μ_i appears in both factors (the left factor's position i
        "points to" the right factor's first leaf, consistent with the
        placeholder-min-leaf convention for tree splits).

        Args:
            x: An element of the cofree module.
            i: Starting position of the right factor's leaves (1-indexed).
            m: Arity of the left factor.
            n: Arity of the right factor.

        Returns:
            An element of ``T^c_C(M)(m) ⊗ T^c_C(M)(n)``.

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

            # Left M-tuple: μ_1,...,μ_i (first i), then μ_{i+n},...,μ_{m+n-1}
            # (μ_i is included in the left at position i, using the same
            # placeholder convention as tree-based splits)
            left_m = m_tuple[:i] + m_tuple[i + n - 1 :]
            # Right M-tuple: μ_i,...,μ_{i+n-1}
            right_m = m_tuple[i - 1 : i + n - 1]

            for (c_L_key, c_R_key), tensor_coeff in cocomp:
                left_key = (c_L_key, left_m)
                right_key = (c_R_key, right_m)
                if cofree_mod._validate_basis_key(left_key) is None:
                    continue
                if cofree_mod._validate_basis_key(right_key) is None:
                    continue
                result += (
                    coeff
                    * tensor_coeff
                    * cofree_mod.term(left_key).tensor(cofree_mod.term(right_key))
                )

        return result

    def project(self, x):
        """Coprojection π: T^c_C(M) → M, projecting onto arity-1 generators.

        Returns the image in the inner module M.  Non-zero only for elements
        of the form ``(id_key, (m_key,))`` where ``id_key ∈ C(1)``.

        Args:
            x: An element of the cofree module.

        Returns:
            An element of the inner module M.

        """
        inner = self._inner_module
        result = inner.zero()

        for (c_key, m_tuple), coeff in x:
            if len(m_tuple) == 1:
                result += coeff * inner.term(m_tuple[0])
        return result
