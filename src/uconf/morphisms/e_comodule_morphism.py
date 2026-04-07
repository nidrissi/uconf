"""E-comodule morphism Δ: Ω(C) → E ⊗ Ω(C).

For a quasi-planar cooperad C, the Berger–Fresse E-comodule structure gives
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
    r"""Compute the Berger–Fresse E-comodule map on a cooperad element.

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
            acted = target.zero()
            for (be_key, coop_key), t_coeff in planar_result:
                # RIGHT action on BE: right-multiply each simplex element by σ.
                be_right = tuple(p * sigma for p in be_key)
                be_perm = be_component(be_right)
                coop_perm = cooperad_component(coop_key).permute(sigma)
                for be_k, be_c in be_perm:
                    for cp_k, cp_c in coop_perm:
                        acted += (
                            t_coeff
                            * be_c
                            * cp_c
                            # https://github.com/sagemath/sage/issues/41882
                            * tensor([be_component(be_k), cooperad_component(cp_k)])
                        )
            total_result += pl_coeff * acted

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
    result = target.zero()

    # Pre-materialise the non-identity permutations once so that the inner
    # loop never triggers GAP's strong_generating_system on each recursion.
    n = cooperad_component.arity()
    non_id_perms = _non_identity_perms(n)
    # Cache the k=0 BE basis element (identity simplex) to avoid repeated
    # construction on every recursion entry.
    be_id_elem = be_component((identity_n,))

    def make_cooperad_factor(pl_elem, sigma_prod=None):
        """Build the cooperad factor for the formula.

        For k=0 (no permutations applied), returns the element as-is.
        For k≥1, applies the cumulative permutation σ_prod to the
        cooperad element via the S_n action: c ↦ c·σ_prod.
        """
        if sigma_prod is not None and sigma_prod != identity_n:
            return pl_elem.permute(sigma_prod)
        return pl_elem

    def recurse(current_d_elem, sigma_bar):
        """Accumulate contributions, pruning zero branches."""
        nonlocal result

        if not current_d_elem:
            return

        k = len(sigma_bar)

        if k == 0:
            result += be_id_elem.tensor(current_d_elem)
        else:
            be_elem = be_component.rho(list(sigma_bar))

            if be_elem:
                sigma_prod = identity_n
                for s in sigma_bar:
                    sigma_prod = s * sigma_prod

                coop_factor = make_cooperad_factor(current_d_elem, sigma_prod)
                result += be_elem.tensor(coop_factor)

        for sigma in non_id_perms:
            next_d = cooperad_component.d_sigma(current_d_elem, sigma)
            if next_d:
                recurse(next_d, sigma_bar + [sigma])

    recurse(planar_elem, [])
    return result


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

    def _extend_tree(tree: Any, base_ring: Any) -> Any:
        """Extend the morphism to a single tree by the free operad universal property."""
        if is_leaf(tree):
            return target_factory.unit(base_ring)

        dec = decoration(tree)
        kids = children(tree)
        k = vertex_arity(tree)

        # Map the root generator via e_comodule_on_generator.
        # This returns an element of E(k) ⊗ C(k) (cooperad level).
        cooperad_parent = cooperad_cls(k, base_ring)
        gen_elem = cooperad_parent(dec)

        root_tensor = e_comodule_on_generator(gen_elem)

        # Embed C(k) into Ω(C)(k) via the canonical inclusion ι,
        # then convert tensor([BE(k), Ω(C)(k)]) → HadamardProduct element.
        cobar_k = cobar(k, base_ring)
        target_k = target_factory(k, base_ring)
        root_image = target_k.zero()
        be_component = BarrattEccles(k, base_ring)
        for (be_key, coop_key), t_coeff in root_tensor:
            # Embed cooperad element as single-vertex cobar tree: (dec, 1, …, k)
            cobar_tree = RootedTree(coop_key, *range(1, k + 1))
            cobar_elem = cobar_k(cobar_tree)
            # Koszul sign from commuting the BE element (degree |e|) past
            # the desuspension s⁻¹ (degree −1) in the inclusion ι: C ↪ Ω(C):
            #   e ⊗ c  ↦  (−1)^{|e|} e ⊙ s⁻¹c
            koszul = sign_from_exponent(be_component.degree_on_basis(be_key))
            for cobar_key, c_coeff in cobar_elem:
                root_image += koszul * t_coeff * c_coeff * target_k((be_key, cobar_key))

        # Compose with child images from right to left (∘_k, ∘_{k-1}, ..., ∘_1)
        # This preserves input positions 1, ..., j-1 at each step.
        result = root_image
        for j in range(k, 0, -1):
            child = kids[j - 1]
            if is_leaf(child):
                # Composing with the unit is the identity, skip
                continue
            child_image = _extend_tree(child, base_ring)
            result = target_factory.compose(result, j, child_image)

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

        return result

    def _on_element(element: Any) -> Any:
        """Apply the morphism to an arbitrary cobar element by linearity."""
        parent = element.parent()
        n = parent.arity()
        base_ring = parent.base_ring()
        target_n = target_factory(n, base_ring)
        result = target_n.zero()
        for tree, coeff in element:
            result += coeff * _extend_tree(tree, base_ring)
        return result

    return OperadMorphism(cobar, target_factory, _on_element)
