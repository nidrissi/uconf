"""E-comodule morphism Δ: Ω(C) → E ⊗ Ω(C).

For a quasi-planar cooperad C, the Le Grignou–Roca i Lucio E-comodule structure gives
a chain map :math:`\\nu: C \\to E \\otimes C` at the cooperad level, where
E = BarrattEccles is the Barratt–Eccles operad.  This is computed by
:func:`e_comodule_on_generator`.

The extension to an operad morphism :math:`\\Omega(C) \\to E \\otimes \\Omega(C)`
uses the universal property of the free operad :math:`\\Omega(C) = T(s^{-1}\\bar C)`:
on each generator (single-vertex cobar tree), the cooperad element is mapped
via :math:`\\nu`, embedded into :math:`E \\otimes \\Omega(C)` via the canonical
inclusion :math:`\\iota: C \\hookrightarrow \\Omega(C)`, and then composed
operadically with child images.

The cooperad :math:`\\mathcal{C}` must be quasi-planar: its arity-n components
must inherit from :class:`~uconf.core.quasi_planar.QuasiPlanarMixin` and expose
``planarize``, ``boundary``, and ``d_sigma``.  The canonical case is
:class:`~uconf.constructions.bar_construction.BarConstruction` applied to a
Hadamard product :math:`P \\otimes \\mathcal{E}` or :math:`P \\otimes \\Surj`.
"""

from __future__ import annotations

from typing import Any

from sage.all import SymmetricGroup, cached_function, tensor

from uconf.constructions.cobar_construction import CobarConstruction
from uconf.core.cooperad import CooperadComponent, CooperadLike
from uconf.core.morphism import OperadMorphism
from uconf.core.signs import sign_from_exponent
from uconf.core.quasi_planar import QuasiPlanarMixin
from uconf.core.trees import RootedTree, children, decoration, is_leaf, vertex_arity
from uconf.models.barratt_eccles import BarrattEccles
from uconf.wrappers.hadamard_operad import HadamardProduct


# Module-level caches for hot paths in _nu_on_planar.  Keyed on
# (arity, elem_key) so entries are shared across distinct top-level calls
# that encounter the same cooperad sub-elements.
_decompose_cache: dict = {}
_rho_cache: dict = {}


def _sigma_as_tuple(sigma: Any) -> tuple[int, ...]:
    """Return a permutation as a one-line tuple."""
    if isinstance(sigma, tuple):
        return sigma
    if isinstance(sigma, list):
        return tuple(sigma)
    return tuple(sigma.tuple())


def _element_cache_key(element: Any) -> tuple:
    """Return a hashable cache key for a free-module element."""
    return tuple(element)


def _permute_component_element_direct(
    component: Any,
    element: Any,
    sigma: Any,
    sigma_tuple: tuple[int, ...] | None = None,
) -> Any:
    """Permute an element via ``_permute_on_basis`` when available.

    This avoids repeated ``result += ...`` element construction inside the
    element-level ``permute`` wrappers, which is a hot path in the E-comodule
    recursion.
    """
    permute_on_basis = getattr(component, "_permute_on_basis", None)
    if permute_on_basis is None:
        return element.permute(sigma)
    if sigma_tuple is None:
        sigma_tuple = _sigma_as_tuple(sigma)

    result_dict: dict = {}
    R = component.base_ring()
    for basis_key, coeff in element:
        permuted = permute_on_basis(basis_key, sigma_tuple)
        for permuted_key, permuted_coeff in permuted:
            combined = R(coeff * permuted_coeff)
            if permuted_key in result_dict:
                result_dict[permuted_key] += combined
            else:
                result_dict[permuted_key] = combined
    return component._from_dict(result_dict, remove_zeros=True)


def _permute_component_basis_direct(
    component: Any,
    basis_key: Any,
    sigma: Any,
    sigma_tuple: tuple[int, ...] | None = None,
) -> Any:
    """Permute a single basis key via ``_permute_on_basis`` when available."""
    permute_on_basis = getattr(component, "_permute_on_basis", None)
    if permute_on_basis is not None:
        if sigma_tuple is None:
            sigma_tuple = _sigma_as_tuple(sigma)
        return permute_on_basis(basis_key, sigma_tuple)
    return component.term(basis_key).permute(sigma)


@cached_function
def _non_identity_perms(n: int) -> list:
    """Return the list of non-identity elements of ``S_n`` in a fixed order.

    Cached so that ``SymmetricGroup(n).__iter__`` (which internally invokes
    GAP's ``StabChain``/``strong_generating_system``) is called only once
    per arity ``n`` for the lifetime of the process.
    """
    S = SymmetricGroup(n)
    identity = S.identity()
    return [s for s in S if s != identity]


@cached_function
def e_comodule_on_generator(dec_elem: Any) -> Any:
    r"""Compute the Le Grignou–Roca i Lucio E-comodule map on a cooperad element.

    Given an element *dec_elem* :math:`c \in \mathcal{C}(n)` (planar or not),
    returns the E-comodule structure map

    .. math::

        \nu(c) \in \mathcal{E}(n) \otimes \mathcal{C}(n).

    This is a **cooperad-level** chain map: it satisfies

    .. math::

        \nu(\partial_C\, c)
          = (d_E \otimes 1 + 1 \otimes \partial_C)(\nu(c))

    where :math:`\partial_C` is the cooperad differential.

    For planar input the formula is the direct recursive expansion using
    :math:`d_\sigma` components.  Non-planar input is handled via
    equivariance:
    :math:`\nu(c \cdot \sigma) = \nu(c) \cdot \sigma`,
    where the diagonal :math:`S_n` action acts on both the
    Barratt–Eccles and cooperad factors.

    Parameters
    ----------
    dec_elem :
        Element of a quasi-planar cooperad component (planar or not).

    Returns
    -------
    Element of ``tensor([be_component, cooperad_component])``.

    Notes
    -----
    Sequences :math:`\underline\sigma` that contain the identity permutation
    give :math:`\rho(\underline\sigma) = 0` (degenerate Barratt–Eccles
    element), so they are skipped.  Zero branches of :math:`d_\sigma` are
    pruned early.  The recursion terminates naturally because each
    :math:`d_\sigma` reduces the degree by 1 and the cooperad is bounded below.
    """
    cooperad_component: CooperadComponent = dec_elem.parent()
    assert isinstance(cooperad_component, QuasiPlanarMixin), (
        "Expected a quasi-planar cooperad component."
    )

    n = cooperad_component.arity()
    base_ring = cooperad_component.base_ring()
    be_component = BarrattEccles(n, base_ring)

    S_n = SymmetricGroup(n)
    identity_n = S_n.identity()

    target = tensor([be_component, cooperad_component])

    if not dec_elem:
        return target.zero()

    # -----------------------------------------------------------------------
    # Planarize the input: dec_elem = Σ pl_coeff * c_pl ⊗ σ.
    # Compute ν on each planar component and apply equivariance.
    # -----------------------------------------------------------------------
    planarized = cooperad_component.planarize(dec_elem)

    total_result = target.zero()

    for (planar_key, sigma_key), pl_coeff in planarized:
        sigma = S_n(sigma_key)
        planar_elem = cooperad_component(planar_key)

        # Compute ν on the planar element via the recursive formula.
        planar_result = _nu_on_planar(
            planar_elem,
            cooperad_component,
            be_component,
            S_n,
            identity_n,
            target,
        )

        # Apply equivariance: ν(c·σ) = ν(c)·σ.
        # The diagonal S_n RIGHT action acts on both factors:
        #   (e ⊗ c)·σ = (e·σ) ⊗ (c·σ)
        # Bar construction .permute(σ) already applies the RIGHT action (leaf relabeling).
        # BE .permute(σ) applies LEFT action (σ·p), so for the RIGHT action e·σ
        # we right-multiply each permutation in the BE basis: (p₀,...,pₖ)·σ = (p₀σ,...,pₖσ).
        if sigma == identity_n:
            total_result += pl_coeff * planar_result
        else:
            # Accumulate acted result as dict to avoid repeated tensor construction
            acted_dict: dict = {}
            sigma_tuple = _sigma_as_tuple(sigma)
            for (be_key, coop_key), t_coeff in planar_result:
                # RIGHT action on BE: right-multiply each simplex element by σ.
                be_right = tuple(p * sigma for p in be_key)
                coop_perm = _permute_component_basis_direct(
                    cooperad_component,
                    coop_key,
                    sigma,
                    sigma_tuple,
                )
                for cp_k, cp_c in coop_perm:
                    pair = (be_right, cp_k)
                    combined = t_coeff * cp_c
                    if pair in acted_dict:
                        acted_dict[pair] += combined
                    else:
                        acted_dict[pair] = combined
            total_result += pl_coeff * target._from_dict(acted_dict, remove_zeros=True)

    return total_result


def _nu_on_planar(
    planar_elem: Any,
    cooperad_component: Any,
    be_component: Any,
    S_n: Any,
    identity_n: Any,
    target: Any,
) -> Any:
    """Core recursive computation of ν on a **planar** cooperad element.

    Returns an element of ``tensor([be_component, cooperad_component])``.

    Parameters
    ----------
    planar_elem :
        A planar element of the cooperad component.
    cooperad_component :
        The arity-n cooperad component.
    be_component :
        The arity-n Barratt–Eccles component.
    S_n :
        The symmetric group ``S_n``.
    identity_n :
        The identity element of ``S_n``.
    target :
        The tensor product ``tensor([be_component, cooperad_component])``.

    Returns
    -------
    Element of *target*.
    """
    n = cooperad_component.arity()
    # Pre-warm the non-identity permutation list so that the first call to
    # d_sigma_decompose doesn't trigger GAP's strong_generating_system.
    _non_identity_perms(n)
    # Cache the k=0 BE basis key (identity simplex) to avoid repeated
    # construction on every recursion entry.
    be_id_key = (identity_n,)

    # Accumulate as {(be_key, coop_key): coeff} dict to avoid
    # repeated tensor element construction overhead.
    result_dict: dict[tuple[Any, Any], Any] = {}
    cooperad_factor_cache: dict[tuple[tuple[Any, ...], tuple[int, ...]], Any] = {}
    # Use module-level caches so entries are shared across distinct top-level
    # calls that encounter the same cooperad sub-elements.  Both are keyed on
    # (arity, elem_key) to avoid cross-arity collisions.
    _n_for_cache = n  # capture for use inside nested closures
    be_id = id(be_component)

    def make_cooperad_factor(pl_elem, sigma_prod=None, sigma_prod_tuple=None):
        """Build the cooperad factor for the formula.

        For k=0 (no permutations applied), returns the element as-is.
        For k≥1, applies the cumulative permutation σ_prod to the
        cooperad element via the S_n action: c ↦ c·σ_prod.
        """
        if sigma_prod is not None and sigma_prod != identity_n:
            elem_key = _element_cache_key(pl_elem)
            cache_key = (elem_key, sigma_prod_tuple)
            cached = cooperad_factor_cache.get(cache_key)
            if cached is not None:
                return cached
            coop_factor = _permute_component_element_direct(
                cooperad_component,
                pl_elem,
                sigma_prod,
                sigma_prod_tuple,
            )
            cooperad_factor_cache[cache_key] = coop_factor
            return coop_factor
        return pl_elem

    def recurse(current_d_elem, sigma_bar, sigma_prod, sigma_prod_tuple):
        """Accumulate contributions, pruning zero branches."""
        if not current_d_elem:
            return

        k = len(sigma_bar)

        if k == 0:
            # be_id_key ⊗ current_d_elem
            for coop_key, coop_coeff in current_d_elem:
                pair = (be_id_key, coop_key)
                if pair in result_dict:
                    result_dict[pair] += coop_coeff
                else:
                    result_dict[pair] = coop_coeff
        else:
            sigma_bar_key = tuple(_sigma_as_tuple(sigma) for sigma in sigma_bar)
            rho_global_key = (be_id, sigma_bar_key)
            be_elem = _rho_cache.get(rho_global_key)
            if be_elem is None:
                be_elem = be_component.rho(list(sigma_bar))
                _rho_cache[rho_global_key] = be_elem

            if be_elem:
                coop_factor = make_cooperad_factor(current_d_elem, sigma_prod, sigma_prod_tuple)
                for be_key, be_coeff in be_elem:
                    for coop_key, coop_coeff in coop_factor:
                        pair = (be_key, coop_key)
                        combined = be_coeff * coop_coeff
                        if pair in result_dict:
                            result_dict[pair] += combined
                        else:
                            result_dict[pair] = combined

        # Compute all d_sigma components at once (one boundary+planarize
        # call) instead of once per permutation.
        elem_key = _element_cache_key(current_d_elem)
        decomp_global_key = (_n_for_cache, elem_key)
        decomp = _decompose_cache.get(decomp_global_key)
        if decomp is None:
            decomp = cooperad_component.d_sigma_decompose(current_d_elem)
            _decompose_cache[decomp_global_key] = decomp
        for sigma, next_d in decomp.items():
            if sigma != identity_n:
                next_sigma_prod = sigma * sigma_prod
                recurse(
                    next_d,
                    sigma_bar + [sigma],
                    next_sigma_prod,
                    _sigma_as_tuple(next_sigma_prod),
                )

    recurse(planar_elem, [], identity_n, _sigma_as_tuple(identity_n))

    # Build the tensor element from accumulated dict, bypassing
    # the dict→items→dict round-trip of sum_of_terms(distinct=True).
    return target._from_dict(result_dict, remove_zeros=True)


def make_e_comodule_morphism(
    cooperad_cls: CooperadLike,
) -> OperadMorphism:
    """Build the operad morphism Δ: Ω(C) → E ⊗ Ω(C).

    The morphism is defined on generators of the free operad
    Ω(C) = T(s⁻¹C̄) by composing the cooperad-level E-comodule map
    :func:`e_comodule_on_generator` (which gives ν: C → E ⊗ C) with
    the canonical inclusion ι: C ↪ Ω(C), then extending via operadic
    composition.

    Parameters
    ----------
    cooperad_cls : CooperadLike
        A quasi-planar cooperad (class or factory) for which the cobar
        construction Ω(C) is defined.

    Returns
    -------
    OperadMorphism
        The morphism Δ whose source is ``CobarConstruction(cooperad_cls)``
        and whose target is ``HadamardProduct(BarrattEccles, CobarConstruction(cooperad_cls))``.
    """
    cobar = CobarConstruction(cooperad_cls)
    target_factory = HadamardProduct(BarrattEccles, cobar)
    root_image_cache: dict = {}
    tree_image_cache: dict = {}

    def _root_image_for_generator(dec: Any, k: int, base_ring: Any) -> Any:
        cache_key = (dec, k, base_ring)
        cached = root_image_cache.get(cache_key)
        if cached is not None:
            return cached

        cooperad_parent = cooperad_cls(k, base_ring)
        gen_elem = cooperad_parent(dec)
        root_tensor = e_comodule_on_generator(gen_elem)

        cobar_k = cobar(k, base_ring)
        target_k = target_factory(k, base_ring)
        be_component = BarrattEccles(k, base_ring)

        root_dict: dict = {}
        R = base_ring
        for (be_key, coop_key), t_coeff in root_tensor:
            cobar_tree = RootedTree(coop_key, *range(1, k + 1))
            cobar_elem = cobar_k._from_validated_tree(cobar_tree)
            koszul = sign_from_exponent(be_component.degree_on_basis(be_key))
            for cobar_key, c_coeff in cobar_elem:
                combined_key = (be_key, cobar_key)
                combined_coeff = R(koszul * t_coeff * c_coeff)
                if combined_key in root_dict:
                    root_dict[combined_key] += combined_coeff
                else:
                    root_dict[combined_key] = combined_coeff

        root_image = target_k._from_dict(root_dict, remove_zeros=True)
        root_image_cache[cache_key] = root_image
        return root_image

    def _extend_tree(tree: Any, base_ring: Any) -> Any:
        """Extend the morphism to a single tree by the free operad universal property."""
        if is_leaf(tree):
            return target_factory.unit(base_ring)
        # RootedTree is immutable and hashable, so subtree memoization is safe.
        cache_key = (tree, base_ring)
        cached = tree_image_cache.get(cache_key)
        if cached is not None:
            return cached

        dec = decoration(tree)
        kids = children(tree)
        k = vertex_arity(tree)

        # Map the root generator via the cached cooperad-level comodule map,
        # then embed it into the cobar construction once per distinct
        # generator decoration.
        root_image = _root_image_for_generator(dec, k, base_ring)

        # Compose with child images from left to right (∘_1, ∘_2, ..., ∘_k).
        #
        # Left-to-right order is essential: the γ-composition (simultaneous
        # grafting) in the Hadamard product E⊗Ω(C) carries a Koszul sign
        # from the interchange law.  When implemented via sequential partial
        # compositions ∘_i, left-to-right ordering reproduces this sign
        # exactly, while right-to-left introduces extra (-1)^{|d_i|·|d_j|}
        # terms from the free-operad after-degree A() accumulation.
        #
        # After composing child_j (arity a_j) at its current position, all
        # positions to the right shift by (a_j − 1).  We track this offset.
        result = root_image
        cumulative_shift = 0
        for j in range(1, k + 1):
            child = kids[j - 1]
            if is_leaf(child):
                # Composing with the unit is the identity, skip.
                # Leaf arity is 1, so shift += 0.
                continue
            child_image = _extend_tree(child, base_ring)
            compose_pos = j + cumulative_shift
            result = target_factory.compose(result, compose_pos, child_image)
            cumulative_shift += child._tree_arity - 1

        # Fix leaf ordering.  Operadic composition assigns consecutive
        # blocks of labels to each child: child_1 gets {1,…,a_1},
        # child_2 gets {a_1+1,…,a_1+a_2}, etc.  But the actual cobar
        # tree may interleave those labels differently (e.g. child_1
        # has leaves {1,3} while child_2 is leaf 2).  Permute the
        # result so that standard position i maps to the tree's actual
        # leaf label.  This is only needed (and valid as an S_n
        # permutation) when the tree's leaves are exactly {1,…,n};
        # recursive calls on subtrees with non-consecutive leaves are
        # composed correctly by the parent and do not need relabeling.
        n = tree._tree_arity
        if tree._leaves == frozenset(range(1, n + 1)):
            leaf_order: list[int] = []
            for child in kids:
                if is_leaf(child):
                    leaf_order.append(child)
                else:
                    leaf_order.extend(sorted(child._leaves))
            if leaf_order != list(range(1, n + 1)):
                result = result.permute(leaf_order)

        tree_image_cache[cache_key] = result
        return result

    def _on_element(element: Any) -> Any:
        """Apply the morphism to an arbitrary cobar element by linearity."""
        parent = element.parent()
        n = parent.arity()
        base_ring = parent.base_ring()
        target_n = target_factory(n, base_ring)

        # Accumulate as dict for efficiency
        result_dict: dict = {}
        R = base_ring

        for tree, coeff in element:
            tree_image = _extend_tree(tree, base_ring)
            for key, img_coeff in tree_image:
                combined = R(coeff * img_coeff)
                if key in result_dict:
                    result_dict[key] += combined
                else:
                    result_dict[key] = combined

        return target_n._from_dict(result_dict, remove_zeros=True)

    return OperadMorphism(cobar, target_factory, _on_element)
