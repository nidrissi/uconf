r"""Free P-algebra on a dg-module M.

The free P-algebra on a dg-module M is the composite product

.. math::

    \operatorname{Free}_P(M) = P \circ M
    = \bigoplus_{n\geq 1} P(n) \otimes_{S_n} M^{\otimes n}

with:

- Degree: deg(p_key, m_tuple) = deg_P(p_key) + Σ_i deg_M(m_i).
- Differential d = d_P + d_M from the Koszul sign rule (Leibniz rule).
- P-algebra structure γ: P(k) ⊗ Free_P(M)^⊗k → Free_P(M) given by the
  full operad substitution on the P-decorations and concatenation of
  M-tuples.

**Quasi-planar requirement**: P must be a quasi-planar operad, i.e. each
component ``P(n)`` satisfies ``P(n) ≅ P_pl(n) ⊗ k[S_n]`` and exposes a
``planarize`` linear map.  The basis uses only planar P-keys, so two corollas
``(p·σ, m_tuple)`` and ``(p, σ·m_tuple)`` are identified as the same element.

The basis keys are pairs ``(p_key, m_tuple)`` where:

- ``p_key`` is a **planar** basis key of ``P(n)`` for ``n = len(m_tuple) ≥ 1``.
- ``m_tuple`` is a tuple of ``n`` values, one per input.

The inclusion η: M → Free_P(M) sends a basis key m to (id_key, (m,)) where
``id_key`` is the unique basis key of P(1).

Reference: Loday-Vallette "Algebraic Operads", Section 5.2.
"""

from __future__ import annotations

import itertools
from typing import Any, ClassVar, Iterator

from sage.all import (
    CombinatorialFreeModule,
    Family,
    GradedModulesWithBasis,
    cached_method,
)

from uconf.algebraic.algebra import OperadAlgebra
from uconf.algebraic.tree_module import _module_basis_keys_in_degree, _tuples_in_degree
from uconf.algebraic.tree_module import _basis_support_keys
from uconf.core.display import latex_linear_combination
from uconf.core.signs import get_on_basis, koszul_sign_of_permutation, sign_from_exponent
from uconf.core.vertex_decoration import QuasiPlanarLike


class FreeAlgebraModule(CombinatorialFreeModule):
    """Underlying dg-module of the free P-algebra ``P ∘ M``.

    Basis keys are ``(p_key, m_tuple)`` pairs where ``p_key`` is a **planar**
    basis key of ``P(n)`` and ``m_tuple`` has length ``n``.  The differential
    is the Leibniz rule ``d = d_P + d_M`` with Koszul signs.  Non-planar keys
    are automatically normalised via ``planarize`` (permuting the m_tuple).

    The operad ``P`` must be quasi-planar: each component ``P(n)`` must expose
    a ``planarize`` linear map decomposing elements into planar representative
    ⊗ symmetric group element.

    This class is normally not instantiated directly; use
    :class:`FreeOperadAlgebra` instead.
    """

    name: ClassVar[str] = "Free"

    def __init__(
        self,
        operad_cls: QuasiPlanarLike,
        inner_module: CombinatorialFreeModule,
        *,
        name: str | None = None,
    ):
        """Initialize the free P-algebra module ``P ∘ M``.

        Args:
            operad_cls: Arity-indexed **quasi-planar** operad provider.  Each
                component ``operad_cls(n, base_ring)`` must expose ``planarize``.
            inner_module: Generating dg-module M (a ``CombinatorialFreeModule``).
            name: Display name override.  Defaults to ``P ∘ M``.

        Raises:
            TypeError: If the operad is not quasi-planar (no ``planarize``).

        """
        if name is None:
            name = f"{operad_cls.name} ∘ {inner_module}"
        self._operad_cls = operad_cls
        self._inner_module = inner_module
        base_ring = inner_module.base_ring()
        # Runtime check: operad must be quasi-planar (free S_n-action)
        _comp2 = operad_cls(2, base_ring)
        if not callable(getattr(_comp2, "planarize", None)):
            raise TypeError(
                f"Operad {operad_cls.name!r} is not quasi-planar: "
                f"its arity-2 component does not expose 'planarize'.  "
                "Only quasi-planar operads (Associative, Surjection, BarrattEccles, "
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

    def _normalized_corolla_sum(self, p_elem, m_tuple) -> "FreeAlgebraModule.Element":
        """Return the sum of canonical corollas for ``p_elem ⊗ m_tuple``.

        For each basis term ``(p_key, coeff)`` of ``p_elem ∈ P(n)``, applies
        ``planarize`` to obtain ``Σ (p_planar_key ⊗ σ) * c`` and returns::

            Σ coeff * c * ε(σ; degrees) * self.term((p_planar_key, σ · m_tuple))

        where ``σ · m_tuple`` is the left action on the leaf tensor:

            ``σ · (m_1, ..., m_n) = (m_{σ^{-1}(1)}, ..., m_{σ^{-1}(n)})``

        and ``ε(σ; degrees)`` is the Koszul sign for that graded left action.

        This ensures all stored P-keys are in the planar basis.

        Args:
            p_elem: An element of ``P(n)`` (any arity, may be non-planar).
            m_tuple: A tuple of ``n`` objects (basis keys or module elements).

        Returns:
            An element of ``self`` with planar P-keys.

        """
        result_dict = {}
        R = self.base_ring()
        zero = R.zero()
        for p_key, p_coeff in p_elem:
            for norm_coeff, norm_key in self._normalize_key((p_key, m_tuple)):
                combined = R(p_coeff * norm_coeff)
                result_dict[norm_key] = result_dict.get(norm_key, zero) + combined
        return self._from_dict(result_dict, remove_zeros=True)

    @cached_method
    def _operad_planarize_on_basis(self, n: int):
        """Return the arity-``n`` fast planarization hook or ``None``.

        When present, the returned callable takes a basis key of ``P(n)`` and
        returns the planarization expansion as ``((planar_key, sigma_key), coeff)``
        terms.
        """
        return getattr(self._operad_cls(n, self.base_ring()), "_planarize_on_basis", None)

    @cached_method
    def _operad_boundary_on_basis(self, n: int):
        """Return the arity-``n`` boundary-on-basis hook or ``None``.

        When present, the returned callable takes a basis key of ``P(n)`` and
        returns the boundary expansion as ``(basis_key, coeff)`` terms.
        """
        return get_on_basis(self._operad_cls(n, self.base_ring()).boundary)

    @cached_method
    def _inner_boundary_on_basis(self):
        """Return the inner-module boundary-on-basis hook or ``None``.

        When present, the returned callable takes an inner-module basis key and
        returns the boundary expansion as ``(basis_key, coeff)`` terms.
        """
        return get_on_basis(self._inner_module.boundary)

    @cached_method
    def _normalize_key(self, key):
        """Normalize a single ``(p_key, m_tuple)`` to planar free-algebra keys.

        Returns a list of ``(coeff, normalized_key)`` pairs.
        """
        p_key, m_tuple = key
        n = len(m_tuple)
        comp = self._operad_cls(n, self.base_ring())
        planarize_on_basis = self._operad_planarize_on_basis(n)
        if planarize_on_basis is not None:
            planarized = planarize_on_basis(p_key)
        else:
            planarized = comp.planarize(comp.term(p_key))

        R = self.base_ring()
        if n == 1:
            return [
                (R(pl_coeff), (p_planar_key, m_tuple))
                for (p_planar_key, _sigma_key), pl_coeff in planarized
            ]

        identity_tuple = tuple(range(1, n + 1))
        degrees = [self._inner_module.degree_on_basis(m_key) for m_key in m_tuple]
        result = []
        for (p_planar_key, sigma_key), pl_coeff in planarized:
            sigma_tuple = sigma_key if isinstance(sigma_key, tuple) else tuple(sigma_key)
            if sigma_tuple == identity_tuple:
                result.append((R(pl_coeff), (p_planar_key, m_tuple)))
                continue
            sigma_inv = [0] * n
            for pos, value in enumerate(sigma_tuple, start=1):
                sigma_inv[value - 1] = pos
            sigma_inv_tuple = tuple(sigma_inv)
            permuted_m = tuple(m_tuple[sigma_inv_tuple[i] - 1] for i in range(n))
            perm_0idx = [s - 1 for s in sigma_inv_tuple]
            koszul = koszul_sign_of_permutation(perm_0idx, degrees)
            result.append((R(pl_coeff * koszul), (p_planar_key, permuted_m)))
        return result

    def _normalize_known_planar_key(self, key):
        """Normalize a key whose operad part is already a stored planar basis key.

        Every key stored in this module has already been normalized, so when the
        differential only changes the inner-module tuple we can preserve the
        operad basis key unchanged and avoid a second planarization pass.

        Returns a one-term tuple ``((coeff, normalized_key),)``.
        """
        return ((self.base_ring().one(), key),)

    # ------------------------------------------------------------------
    # Basis key validation
    # ------------------------------------------------------------------

    def _validate_basis_key(self, key) -> tuple | None:
        """Validate a ``(p_key, m_tuple)`` basis key (structural check only).

        Returns the normalised key, or ``None`` if structurally invalid.
        Does **not** planarize; use :meth:`_normalized_corolla_sum` for that.
        """
        if not isinstance(key, (tuple, list)) or len(key) != 2:
            raise ValueError(f"Invalid basis key {key!r}: expected a pair (p_key, m_tuple).")
        p_key_raw, m_tuple_raw = key[0], key[1]
        if not isinstance(m_tuple_raw, (tuple, list)):
            raise ValueError(
                f"Invalid basis key {key!r}: m_tuple must be a tuple/list of leaf keys."
            )
        n = len(m_tuple_raw)
        if n == 0:
            raise ValueError(
                f"Invalid basis key {key!r}: m_tuple cannot be empty (arity must be ≥ 1)."
            )

        # Validate p_key against P(n)
        # NOTE: Component construction P(n, R) must never fail—all operads
        # support every non-negative arity.  Any exception here is a real bug
        # and should propagate, not be silently ignored.
        comp = self._operad_cls(n, self.base_ring())
        validate_fn = getattr(comp, "_validate_basis_key", None)
        if validate_fn is not None:
            p_key = validate_fn(p_key_raw)
            if p_key is None:
                return None
        else:
            p_key = p_key_raw

        return (p_key, tuple(m_tuple_raw))

    def _element_constructor_(self, x):
        if isinstance(x, dict):
            clean_dict = {}
            R = self.base_ring()
            zero = R.zero()
            for key, coeff in x.items():
                k = self._validate_basis_key(key)
                if k is not None:
                    p_key_raw, m_tuple_raw = k
                    for norm_coeff, norm_key in self._normalize_key((p_key_raw, m_tuple_raw)):
                        clean_dict[norm_key] = clean_dict.get(norm_key, zero) + R(
                            coeff * norm_coeff
                        )
            return self._from_dict(clean_dict, remove_zeros=True)

        if isinstance(x, (tuple, list)) and len(x) == 2:
            k = self._validate_basis_key(x)
            if k is None:
                return self.zero()
            p_key_raw, m_tuple_raw = k
            clean_dict = {}
            R = self.base_ring()
            zero = R.zero()
            for norm_coeff, norm_key in self._normalize_key((p_key_raw, m_tuple_raw)):
                clean_dict[norm_key] = clean_dict.get(norm_key, zero) + norm_coeff
            return self._from_dict(clean_dict, remove_zeros=True)

        raise TypeError(
            f"Cannot construct element from {x!r}. Expected a dict of basis keys to coefficients, or a single basis key tuple (p_key, m_tuple)."
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
        """Degree = deg_P(p_key) + Σ_i deg_M(m_i)."""
        p_key, m_tuple = key
        n = len(m_tuple)
        comp = self._operad_cls(n, self.base_ring())
        p_deg = comp.degree_on_basis(p_key)
        m_deg = sum(self._inner_module.degree_on_basis(mk) for mk in m_tuple)
        return p_deg + m_deg

    def _repr_term(self, basis_element) -> str:
        """Readable basis-term notation for ``P ∘ M`` corollas."""
        p_key, m_tuple = basis_element
        n = len(m_tuple)
        p_parent = self._operad_cls(n, self.base_ring())

        p_repr = getattr(p_parent, "_repr_term", None)
        m_repr = getattr(self._inner_module, "_repr_term", None)

        p_str = p_repr(p_key) if callable(p_repr) else str(p_key)
        leaves = [m_repr(mk) if callable(m_repr) else str(mk) for mk in m_tuple]
        return f"<{p_str}; {', '.join(leaves)}>"

    def _latex_term(self, basis_element) -> str:
        """LaTeX basis-term notation for ``P ∘ M`` corollas."""
        p_key, m_tuple = basis_element
        n = len(m_tuple)
        p_parent = self._operad_cls(n, self.base_ring())

        p_repr = getattr(p_parent, "_latex_term", None)
        m_repr = getattr(self._inner_module, "_latex_term", None)

        p_ltx = p_repr(p_key) if callable(p_repr) else str(p_key)
        leaves = [m_repr(mk) if callable(m_repr) else str(mk) for mk in m_tuple]
        return f"\\langle {p_ltx}; {', '.join(leaves)} \\rangle"

    # ------------------------------------------------------------------
    # Differential
    # ------------------------------------------------------------------

    @cached_method
    def _boundary_on_basis(self, key) -> Any:
        """Leibniz rule: d(p ⊗ m_1 ⊗…⊗ m_n) = d_P(p) ⊗ m_… + Σ_i (−1)^{…} p ⊗…⊗ d_M(m_i) ⊗….

        Koszul sign at leaf i: ``(−1)^{deg_P(p_key) + Σ_{j<i} deg_M(m_j)}``.
        """
        p_key, m_tuple = key
        n = len(m_tuple)
        comp = self._operad_cls(n, self.base_ring())
        R = self.base_ring()
        zero = R.zero()
        result_dict = {}
        boundary_on_basis = self._operad_boundary_on_basis(n)
        inner_boundary_on_basis = self._inner_boundary_on_basis()

        # d_P term: keep raw operad keys (may be non-planar)
        dp_elem = (
            boundary_on_basis(p_key)
            if boundary_on_basis is not None
            else comp.boundary(comp.term(p_key))
        )
        for dp_key, dp_coeff in dp_elem:
            for norm_coeff, norm_key in self._normalize_key((dp_key, m_tuple)):
                combined = R(dp_coeff * norm_coeff)
                result_dict[norm_key] = result_dict.get(norm_key, zero) + combined

        # d_M terms with Koszul signs
        p_deg = comp.degree_on_basis(p_key)
        cumulative = p_deg
        for i, mk in enumerate(m_tuple):
            sign = sign_from_exponent(cumulative)
            m_boundary = (
                inner_boundary_on_basis(mk)
                if inner_boundary_on_basis is not None
                else self._inner_module.boundary(self._inner_module.term(mk))
            )
            for new_mk, m_coeff in m_boundary:
                new_m = m_tuple[:i] + (new_mk,) + m_tuple[i + 1 :]
                for norm_coeff, norm_key in self._normalize_known_planar_key((p_key, new_m)):
                    combined = R(sign * m_coeff * norm_coeff)
                    result_dict[norm_key] = result_dict.get(norm_key, zero) + combined
            cumulative += self._inner_module.degree_on_basis(mk)

        return self._from_dict(result_dict, remove_zeros=True)

    # ------------------------------------------------------------------
    # Basis iteration
    # ------------------------------------------------------------------

    def basis_iter(self, d: int) -> Iterator[Any]:
        """Iterate over basis elements of total degree ``d``.

        Uses the isomorphism ``P(n) ⊗_{S_n} M^{⊗n} ≅ P_pl(n) ⊗ M^{⊗n}``
        and enumerates only planar P(n)-decorations.

        Raises:
            ValueError: when arity is unbounded (P and M both admit degree-0
                generators).
        """
        M = self._inner_module
        P = self._operad_cls
        R = self.base_ring()

        # Collect inner-module keys by degree.  The operad may contribute
        # negative degrees, so the inner module may need to compensate with
        # degrees higher than d.  The maximum needed m_tuple degree is
        # d - min_operad_deg_at_arity_2; use connectivity as a lower bound.
        connectivity = int(getattr(P, "connectivity", 0))
        max_m_total = d - min(connectivity, 0)  # could exceed d

        m_keys_by_deg: dict[int, list] = {}
        for d_m in range(max_m_total + 1):
            keys = list(_module_basis_keys_in_degree(M, d_m))
            if keys:
                m_keys_by_deg[d_m] = keys

        # n = 1: exactly one P(1)-key (identity); yields M-generators in degree d
        unit_key = P.unit_key()
        for mk in m_keys_by_deg.get(d, []):
            yield self.term((unit_key, (mk,)))

        if not m_keys_by_deg:
            return

        min_m_deg = min(m_keys_by_deg.keys())
        min_corolla_deg = connectivity

        if d < min_corolla_deg:
            return

        # Compute the maximum arity n for which degree-d elements can exist.
        # At arity n the minimum total degree is:
        #   connectivity*(n−1) + n*min_m_deg = n*(connectivity + min_m_deg) − connectivity
        # Solving n*(connectivity + min_m_deg) ≤ d + connectivity:
        step = connectivity + min_m_deg
        if step > 0:
            max_n = (d + connectivity) // step
        elif step == 0:
            if d + connectivity >= 0:
                # Unbounded arity: every n is feasible
                raise ValueError(
                    "Cannot exhaustively enumerate basis_iter(d): both P and M admit "
                    "degree-0 generators (connectivity + min_m_deg = 0), so arity is "
                    "unbounded in fixed degree."
                )
            else:
                max_n = 1  # no n >= 2 is feasible
        else:
            # step < 0: higher arity gives lower minimum degree; all n are feasible.
            # This shouldn't happen for well-behaved operads.
            raise ValueError(
                "Cannot exhaustively enumerate basis_iter(d): "
                "connectivity + min_m_deg < 0, so arity is unbounded in fixed degree."
            )

        for n in range(2, max_n + 1):
            comp_n = P(n, R)
            # The operad degree can be negative (connectivity < 0), so
            # d_p ranges from min(connectivity*(n-1), 0) up to d - n*min_m_deg.
            min_d_p = min(connectivity * (n - 1), 0)
            max_d_p = d - n * min_m_deg if min_m_deg >= 0 else d
            for d_p in range(min_d_p, max_d_p + 1):
                d_m_needed = d - d_p
                if d_m_needed < 0:
                    continue
                p_elems = list(
                    getattr(comp_n, "graded_planar_basis", comp_n.planar_basis_iter)(d_p)
                )
                if not p_elems:
                    continue
                m_tuples = list(_tuples_in_degree(m_keys_by_deg, n, d_m_needed))
                if not m_tuples:
                    continue
                for p_elem in p_elems:
                    for p_key in p_elem.support():
                        for m_tuple in m_tuples:
                            yield self.term((p_key, m_tuple))

    @cached_method
    def graded_basis(self, d: int):
        return Family(self.basis_iter(d))

    # ------------------------------------------------------------------
    # Weight
    # ------------------------------------------------------------------

    def _weight_on_basis(self, key) -> int:
        """Weight of a single basis key ``(p_key, m_tuple)``.

        The weight equals the arity ``n = len(m_tuple)``.
        """
        _p_key, m_tuple = key
        return len(m_tuple)

    def basis_weight_iter(self, d: int, w: int) -> Iterator[Any]:
        """Iterate over basis elements of total degree ``d`` and weight ``w``.

        The weight of ``(p_key, m_tuple)`` is the arity ``len(m_tuple)``.
        Only basis elements with exactly ``w`` inputs (arity ``w``) are
        returned.

        Args:
            d: Homological degree.
            w: Weight (arity) to enumerate.

        Yields:
            Elements of this module with degree ``d`` and weight ``w``.
        """
        if w < 1:
            return

        M = self._inner_module
        P = self._operad_cls
        R = self.base_ring()

        if w == 1:
            # n=1: identity key, inner-module generators in degree d
            unit_key = P.unit_key()
            for mk in _module_basis_keys_in_degree(M, d):
                yield self.term((unit_key, (mk,)))
            return

        comp_n = P(w, R)
        connectivity = int(getattr(P, "connectivity", 0))

        # Collect inner-module keys by degree
        max_m_deg = d - min(connectivity * (w - 1), 0)
        m_keys_by_deg: dict[int, list] = {}
        for d_m in range(max_m_deg + 1):
            keys = list(_module_basis_keys_in_degree(M, d_m))
            if keys:
                m_keys_by_deg[d_m] = keys

        if not m_keys_by_deg:
            return

        min_m_deg = min(m_keys_by_deg.keys())
        min_d_p = min(connectivity * (w - 1), 0)
        max_d_p = d - w * min_m_deg if min_m_deg >= 0 else d

        for d_p in range(min_d_p, max_d_p + 1):
            d_m_needed = d - d_p
            if d_m_needed < 0:
                continue
            p_keys = tuple(
                _basis_support_keys(
                    getattr(comp_n, "graded_planar_basis", comp_n.planar_basis_iter)(d_p)
                )
            )
            if not p_keys:
                continue
            for m_tuple in _tuples_in_degree(m_keys_by_deg, w, d_m_needed):
                for p_key in p_keys:
                    yield self.term((p_key, m_tuple))

    @cached_method
    def graded_basis_by_weight(self, d: int, w: int):
        """Cached family of basis elements of degree ``d`` and weight ``w``."""
        return Family(self.basis_weight_iter(d, w))

    # ------------------------------------------------------------------
    # Element class
    # ------------------------------------------------------------------

    class Element(CombinatorialFreeModule.Element):
        """An element of the free P-algebra module ``P ∘ M``."""

        def _repr_latex_(self) -> str:
            return latex_linear_combination(self, lambda basis: self.parent()._latex_term(basis))

        def boundary(self) -> "FreeAlgebraModule.Element":
            """Apply the differential d = d_P + d_M."""
            return self.parent().boundary(self)


class FreeOperadAlgebra(OperadAlgebra):
    """Free P-algebra on a dg-module M.

    Constructs the composite product ``P ∘ M`` as a :class:`FreeAlgebraModule`
    and equips it with the canonical P-algebra structure.

    The action ``γ(q; a_1, ..., a_k)`` applies the full operad substitution
    ``q(p_1, ..., p_k) ∈ P(Σn_i)`` and concatenates the M-tuples.  The result
    is normalised via ``planarize``: if the composed P-element has a non-planar
    key ``r·σ``, the output carries ``r`` (planar) and ``σ · m_concat``
    (permuted M-entries).

    **Quasi-planar requirement**: P must be quasi-planar (expose ``planarize``
    on each component).  Non-quasi-planar operads such as Commutative or Lie
    are **not** accepted.

    Args:
        operad_cls: Quasi-planar operad provider P (class or factory instance).
        inner_module: The generating dg-module M.

    The inclusion ``η: M → Free_P(M)`` is::

        m_key  ↦  free_algebra.module.term((id_key, (m_key,)))

    where ``id_key`` is the unique basis key of ``P(1)``.

    Examples::

        from uconf import Associative
        from sage.all import QQ, CombinatorialFreeModule, GradedModulesWithBasis

        M = CombinatorialFreeModule(QQ, ['a', 'b'],
                                    category=GradedModulesWithBasis(QQ))
        M.degree_on_basis = lambda _: 1
        M.boundary_on_basis = lambda _: M.zero()
        F = FreeOperadAlgebra(Associative, M)
        a = F.include('a')
        b = F.include('b')
        # Non-planar Ass(2) key: result is normalised to planar key with swapped M-tuple
        result = F.act(Associative(2, QQ)((2, 1)), [a, b])
        # → corolla ((1,2), (B['b'], B['a']))

    """

    def __init__(self, operad_cls: QuasiPlanarLike, inner_module: CombinatorialFreeModule):
        free_module = FreeAlgebraModule(operad_cls, inner_module)
        super().__init__(free_module, operad_cls, self._act_impl)
        self._inner_module = inner_module
        self._base_ring = inner_module.base_ring()

    @cached_method
    def _act_on_basis_tuple(self, q_key: Any, input_keys: tuple):
        """Return ``γ(q; a_1, ..., a_k)`` for basis inputs."""
        k = len(input_keys)
        P = self.operad_cls
        R = self._base_ring

        # Full substitution: compose q with each p_i from left to right.
        n_list = [len(input_key[1]) for input_key in input_keys]
        composed_elem = P(k, R).term(q_key)
        cumulative_shift = 0
        for j, input_key in enumerate(input_keys):
            p_j_key, _m_j_tuple = input_key
            n_j = n_list[j]
            p_j_elem = P(n_j, R).term(p_j_key)
            pos = j + 1 + cumulative_shift
            composed_elem = P.compose(composed_elem, pos, p_j_elem)
            cumulative_shift += n_j - 1

        # Koszul sign from rearranging
        #   q ⊗ (p₁⊗m₁) ⊗ … ⊗ (pₖ⊗mₖ)
        # into
        #   (q ⊗ p₁ ⊗ … ⊗ pₖ) ⊗ (m₁ ⊗ … ⊗ mₖ).
        inner_mod = self._inner_module
        sign_exp = 0
        for j in range(k - 1):
            _p_j_key, m_j_tuple = input_keys[j]
            m_deg_j = sum(inner_mod.degree_on_basis(mk) for mk in m_j_tuple)
            for l in range(j + 1, k):
                p_l_key = input_keys[l][0]
                p_l_deg = P(n_list[l], R).degree_on_basis(p_l_key)
                sign_exp += m_deg_j * p_l_deg
        koszul = sign_from_exponent(sign_exp)

        m_concat = tuple(mk for input_key in input_keys for mk in input_key[1])
        return koszul * self.module._normalized_corolla_sum(composed_elem, m_concat)

    @cached_method
    def _act_on_basis_inputs(self, p_terms: tuple, input_keys: tuple):
        """Apply the free-algebra action to basis inputs and cache the result."""
        R = self._base_ring
        zero = R.zero()
        result_dict: dict = {}
        for q_key, q_coeff in p_terms:
            basis_result = self._act_on_basis_tuple(q_key, input_keys)
            for out_key, out_coeff in basis_result:
                combined = R(q_coeff * out_coeff)
                result_dict[out_key] = result_dict.get(out_key, zero) + combined
        return self.module._from_dict(result_dict, remove_zeros=True)

    def _act_impl(self, p_element, algebra_elements):
        """P-algebra action γ(q; a_1, ..., a_k) via full operad substitution.

        For each input ``a_i = (p_i_key, x_i_tuple)`` (a corolla in
        ``Free_P(M)``), computes the full operad substitution
        ``q(p_1, ..., p_k) ∈ P(n_1 + ... + n_k)`` and normalises the result
        via ``planarize``, permuting the concatenated M-tuple accordingly.

        Args:
            p_element: An element of ``operad_cls(k)`` for some arity ``k``.
            algebra_elements: A list of ``k`` elements of ``free_module``.

        Returns:
            An element of ``free_module`` with planar P-keys.

        """
        k = p_element.arity()
        inputs = list(algebra_elements)
        if len(inputs) != k:
            raise ValueError(f"Expected {k} inputs for P({k}) action, got {len(inputs)}.")

        R = self._base_ring
        input_term_lists = [list(x) for x in inputs]
        result_dict: dict = {}
        zero = R.zero()

        for q_key, q_coeff in p_element:
            for term_combo in itertools.product(*input_term_lists):
                input_keys = [bk for (bk, _) in term_combo]
                coeff = q_coeff
                for _, c in term_combo:
                    coeff = coeff * c
                if coeff == zero:
                    continue
                basis_result = self._act_on_basis_tuple(q_key, tuple(input_keys))
                for out_key, out_coeff in basis_result:
                    combined = R(coeff * out_coeff)
                    result_dict[out_key] = result_dict.get(out_key, zero) + combined

        return self.module._from_dict(result_dict, remove_zeros=True)

    def include(self, m_key):
        """Return the image of ``m_key`` under the inclusion η: M → P ∘ M.

        Args:
            m_key: A basis key of the inner module M (or any object stored as
                an M-label in the free algebra).

        Returns:
            The element ``module.term((id_key, (m_key,)))`` where ``id_key``
            is the unique basis key of ``P(1)``.

        """
        id_key = self.operad_cls.unit_key()
        return self.module.term((id_key, (m_key,)))
