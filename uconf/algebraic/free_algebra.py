"""Free P-algebra on a dg-module M.

The free P-algebra on a dg-module M is the composite product

    Free_P(M) = P ∘ M = ⊕_{n≥1} P(n) ⊗_{S_n} M^{⊗n}

with:
- Degree: deg(p_key, m_tuple) = deg_P(p_key) + Σ_i deg_M(m_i).
- Differential d = d_P + d_M from the Koszul sign rule (Leibniz rule).
- P-algebra structure γ: P(k) ⊗ Free_P(M)^⊗k → Free_P(M) given by the
  full operad substitution on the P-decorations and concatenation of
  M-tuples.

The basis keys are pairs ``(p_key, m_tuple)`` where:

- ``p_key`` is a basis key of ``P(n)`` for ``n = len(m_tuple) ≥ 1``.
- ``m_tuple`` is a tuple of ``n`` basis keys of the inner module M.

The arity ``n`` is determined implicitly as ``len(m_tuple)``.  For ``n = 1``
the unique P(1)-basis key is the identity (connected operad convention).

The inclusion η: M → Free_P(M) sends a basis key m to (id_key, (m,)) where
``id_key`` is the unique basis key of P(1).

Reference: Loday-Vallette "Algebraic Operads", Section 5.2.
"""

from __future__ import annotations

import itertools
from typing import Any, ClassVar, Iterator

from sage.all import CombinatorialFreeModule, Family, GradedModulesWithBasis, cached_method

from uconf.algebraic.algebra import OperadAlgebra
from uconf.algebraic.tree_module import _module_basis_keys_in_degree, _tuples_in_degree
from uconf.core.operad import OperadLike
from uconf.core.signs import sign_from_exponent


class FreeAlgebraModule(CombinatorialFreeModule):
    """Underlying dg-module of the free P-algebra ``P ∘ M``.

    Basis keys are ``(p_key, m_tuple)`` pairs, where ``p_key`` is a basis
    key of ``P(n)`` and ``m_tuple`` is an ``n``-tuple of M-basis keys with
    ``n = len(m_tuple)``.  The differential is the Leibniz rule
    ``d = d_P + d_M`` with Koszul signs.

    This class is normally not instantiated directly; use
    :class:`FreeOperadAlgebra` instead.
    """

    name: ClassVar[str] = "Free"

    def __init__(
        self,
        operad_cls: OperadLike,
        inner_module: CombinatorialFreeModule,
        base_ring,
        *,
        name: str | None = None,
    ):
        """Initialize the free P-algebra module ``P ∘ M``.

        Args:
            operad_cls: Arity-indexed operad (or operad-like) provider.
            inner_module: Generating dg-module M (a ``CombinatorialFreeModule``).
            base_ring: Coefficient ring.
            name: Display name override.  Defaults to ``P ∘ M``.

        """
        if name is None:
            name = f"{operad_cls.name} ∘ {inner_module}"
        self._operad_cls = operad_cls
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
        """Validate and normalise a ``(p_key, m_tuple)`` basis key.

        Returns the normalised key, or ``None`` if invalid.
        """
        if not isinstance(key, (tuple, list)) or len(key) != 2:
            return None
        p_key_raw, m_tuple_raw = key[0], key[1]
        if not isinstance(m_tuple_raw, (tuple, list)):
            return None
        n = len(m_tuple_raw)
        if n == 0:
            return None

        # Validate p_key against P(n)
        try:
            comp = self._operad_cls(n, self.base_ring())
            if hasattr(comp, "_validate_basis_key"):
                p_key = comp._validate_basis_key(p_key_raw)
                if p_key is None:
                    return None
            else:
                p_key = p_key_raw
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

        return (p_key, tuple(new_m))

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
        """Degree = deg_P(p_key) + Σ_i deg_M(m_i)."""
        p_key, m_tuple = key
        n = len(m_tuple)
        comp = self._operad_cls(n, self.base_ring())
        p_deg = comp.degree_on_basis(p_key)
        m_deg = sum(self._inner_module.degree_on_basis(mk) for mk in m_tuple)
        return p_deg + m_deg

    # ------------------------------------------------------------------
    # Differential
    # ------------------------------------------------------------------

    def _boundary_on_basis(self, key) -> Any:
        """Leibniz rule: d(p ⊗ m_1 ⊗…⊗ m_n) = d_P(p) ⊗ m_… + Σ_i (−1)^{…} p ⊗…⊗ d_M(m_i) ⊗….

        Koszul sign at leaf i: ``(−1)^{deg_P(p_key) + Σ_{j<i} deg_M(m_j)}``.
        """
        p_key, m_tuple = key
        n = len(m_tuple)
        comp = self._operad_cls(n, self.base_ring())
        result = self.zero()

        # d_P term
        p_elem = comp.term(p_key)
        for new_p_key, coeff in comp.boundary(p_elem):
            result += coeff * self.term((new_p_key, m_tuple))

        # d_M terms with Koszul signs
        p_deg = comp.degree_on_basis(p_key)
        cumulative = p_deg
        for i, mk in enumerate(m_tuple):
            sign = sign_from_exponent(cumulative)
            m_elem = self._inner_module.term(mk)
            for new_mk, m_coeff in self._inner_module.boundary(m_elem):
                new_m = m_tuple[:i] + (new_mk,) + m_tuple[i + 1 :]
                result += sign * m_coeff * self.term((p_key, new_m))
            cumulative += self._inner_module.degree_on_basis(mk)

        return result

    # ------------------------------------------------------------------
    # Basis iteration
    # ------------------------------------------------------------------

    def basis_it(self, d: int) -> Iterator[Any]:
        """Iterate over basis elements of total degree ``d``.

        For a quasi-planar operad P, uses the isomorphism
        ``P(n) ⊗_{S_n} M^{⊗n} ≅ P_pl(n) ⊗ M^{⊗n}`` and enumerates only
        planar P(n)-decorations for ``n ≥ 2``.

        Raises:
            NotImplementedError: when P does not expose ``planar_basis_it``
                (e.g. ``Commutative``, ``Lie``).
            ValueError: when arity is unbounded (P and M both admit degree-0
                generators).
        """
        M = self._inner_module
        P = self._operad_cls
        R = self.base_ring()

        m_keys_by_deg: dict[int, list] = {}
        for d_m in range(d + 1):
            keys = list(_module_basis_keys_in_degree(M, d_m))
            if keys:
                m_keys_by_deg[d_m] = keys

        # n = 1: exactly one P(1)-key (identity); yields M-generators in degree d
        unit_key = P.unit(R).support()[0]
        for mk in m_keys_by_deg.get(d, []):
            yield self.term((unit_key, (mk,)))

        if not m_keys_by_deg:
            return

        min_m_deg = min(m_keys_by_deg.keys())
        connectivity = getattr(P, "connectivity", 0)
        min_corolla_deg = connectivity  # minimum deg of a P(2)-element

        if d < min_corolla_deg:
            return

        # Determine maximum arity
        if min_m_deg > 0:
            max_n = d // min_m_deg
        elif connectivity > 0:
            max_n = (d - 0) // connectivity + 1
        else:
            raise ValueError(
                "Cannot exhaustively enumerate basis_it(d): both P and M admit "
                "degree-0 generators (connectivity=0, min_m_deg=0), so arity is "
                "unbounded in fixed degree."
            )

        # Require quasi-planar support for n ≥ 2
        try:
            _use_planar = hasattr(P(2, R), "planar_basis_it")
        except (TypeError, ValueError, NotImplementedError, AttributeError):
            _use_planar = False

        if not _use_planar:
            raise NotImplementedError(
                f"basis_it() requires the operad {P.name!r} to support "
                "planar_basis_it() on its arity-2 component.  "
                "Supported quasi-planar operads include Associative, Surjection, "
                "BarrattEccles, and ShiftedOperad wrapping any of these.  "
                "For non-quasi-planar operads (e.g. Commutative, Lie) the basis "
                "of the composite product P ∘ M cannot be enumerated this way."
            )

        for n in range(2, max_n + 1):
            try:
                comp_n = P(n, R)
            except (TypeError, ValueError, AttributeError):
                continue
            for d_p in range(d + 1):
                d_m_needed = d - d_p
                if d_m_needed < 0:
                    continue
                # Enumerate planar P(n)-keys in degree d_p
                try:
                    p_elems = list(comp_n.planar_basis_it(d_p))
                except (TypeError, ValueError, NotImplementedError, AttributeError):
                    continue
                if not p_elems:
                    continue
                # M^n tuples in total degree d_m_needed
                m_tuples = list(_tuples_in_degree(m_keys_by_deg, n, d_m_needed))
                if not m_tuples:
                    continue
                for p_elem in p_elems:
                    for p_key in p_elem.support():
                        for m_tuple in m_tuples:
                            yield self.term((p_key, m_tuple))

    @cached_method
    def graded_basis(self, d: int):
        return Family(self.basis_it(d))

    # ------------------------------------------------------------------
    # Element class
    # ------------------------------------------------------------------

    class Element(CombinatorialFreeModule.Element):
        """An element of the free P-algebra module ``P ∘ M``."""

        def boundary(self) -> "FreeAlgebraModule.Element":
            """Apply the differential d = d_P + d_M."""
            return self.parent().boundary(self)


class FreeOperadAlgebra(OperadAlgebra):
    """Free P-algebra on a dg-module M.

    Constructs the composite product ``P ∘ M`` as a
    :class:`FreeAlgebraModule` and equips it with the canonical P-algebra
    structure.  The action ``γ(q; a_1, ..., a_k)`` for ``a_i = (p_i_key,
    x_i_tuple)`` applies the full operad substitution ``q(p_1, ..., p_k)``
    and concatenates the M-tuples.

    Args:
        operad_cls: Operad provider P (class or wrapper instance).
        inner_module: The generating dg-module M.
        base_ring: Coefficient ring.

    The inclusion ``η: M → Free_P(M)`` is::

        m_key  ↦  free_algebra.module.term((id_key, (m_key,)))

    where ``id_key`` is the unique basis key of ``P(1)``.

    Examples::

        free_lie = FreeOperadAlgebra(Lie, module_M, R)
        # Apply the Lie bracket to two generators:
        x = free_lie.include('x')
        y = free_lie.include('y')
        bracket = free_lie.act(Lie(2, R).term((1,)), [x, y])

    """

    def __init__(self, operad_cls: OperadLike, inner_module: CombinatorialFreeModule, base_ring):
        free_module = FreeAlgebraModule(operad_cls, inner_module, base_ring)
        super().__init__(free_module, operad_cls, self._act_impl)
        self._inner_module = inner_module
        self._base_ring = base_ring

    def _act_impl(self, p_element, algebra_elements):
        """P-algebra action γ(q; a_1, ..., a_k) via full operad substitution.

        For each input ``a_i = (p_i_key, x_i_tuple)`` (a corolla in
        ``Free_P(M)``), computes the full operad substitution
        ``q(p_1, ..., p_k) ∈ P(n_1 + ... + n_k)`` and returns the
        corresponding corolla ``(result_key, x_1_tuple + ... + x_k_tuple)``.

        Args:
            p_element: An element of ``operad_cls(k)`` for some arity ``k``.
            algebra_elements: A list of ``k`` elements of ``free_module``.

        Returns:
            An element of ``free_module``.

        """
        k = p_element.arity()
        inputs = list(algebra_elements)
        if len(inputs) != k:
            raise ValueError(f"Expected {k} inputs for P({k}) action, got {len(inputs)}.")

        P = self.operad_cls
        R = self._base_ring
        result = self.module.zero()
        input_term_lists = [list(x) for x in inputs]

        for q_key, q_coeff in p_element:
            for term_combo in itertools.product(*input_term_lists):
                input_keys = [bk for (bk, _) in term_combo]
                coeff = q_coeff
                for _, c in term_combo:
                    coeff = coeff * c

                # Full substitution: compose q with each p_i from right to left
                # q(p_1,...,p_k) = (...((q ∘_k p_k) ∘_{k-1} p_{k-1}) ...) ∘_1 p_1
                n_list = [len(ik[1]) for ik in input_keys]
                # Build the composed P-element iteratively
                # Start with q_elem in P(k)
                composed_arity = k
                composed_elem = P(k, R).term(q_key)

                for j in range(k - 1, -1, -1):
                    p_j_key, m_j_tuple = input_keys[j]
                    n_j = n_list[j]
                    p_j_elem = P(n_j, R).term(p_j_key)
                    pos = j + 1  # 1-indexed position
                    # Before composing at pos j+1, the composed element has
                    # arity = k - (k-1-j) + Σ_{l>j} (n_l - 1)
                    # = j+1 + Σ_{l>j} (n_l - 1) -- this is auto-tracked by P.compose
                    composed_elem = P.compose(composed_elem, pos, p_j_elem)

                # composed_elem is now in P(n_1 + ... + n_k)
                # Concatenate the M-tuples
                m_concat = tuple(mk for ik in input_keys for mk in ik[1])

                for res_key, res_coeff in composed_elem:
                    result += coeff * res_coeff * self.module.term((res_key, m_concat))

        return result

    def include(self, m_key):
        """Return the image of basis key ``m_key`` under the inclusion η: M → P ∘ M.

        Args:
            m_key: A basis key of the inner module M.

        Returns:
            The element ``module.term((id_key, (m_key,)))`` where ``id_key``
            is the unique basis key of ``P(1)``.

        """
        R = self._base_ring
        id_key = self.operad_cls.unit(R).support()[0]
        return self.module.term((id_key, (m_key,)))
