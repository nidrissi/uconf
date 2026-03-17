"""Free P-algebra on a dg-module M.

The free P-algebra on a dg-module M is the composite product

    Free_P(M) = P ∘ M = ⊕_{n≥1} P(n) ⊗_{S_n} M^{⊗n}

with:
- Degree: deg(tree, m_tuple) = Σ_{v internal} deg_P(dec(v)) + Σ_i deg_M(m_i)
  (no suspension factor; compare with the bar construction which uses deg_P + 1).
- Differential d = d_P + d_M from the Koszul sign rule on the interleaved
  DFS pre-order of vertices and leaves.
- P-algebra structure γ: P(k) ⊗ Free_P(M)^⊗k → Free_P(M) given by grafting
  a new root vertex over k subtrees.

The basis keys are pairs ``(tree, m_tuple)`` where:

- ``tree`` is an integer leaf (= 1 in arity 1) or a tuple representing a
  decorated rooted tree with leaves labeled ``1, ..., n``.
- ``m_tuple`` is a tuple of ``n`` basis keys of the inner module M.

The inclusion η: M → Free_P(M) sends a basis key m to (1, (m,)).

Reference: Loday-Vallette "Algebraic Operads", Section 5.2.
"""

from __future__ import annotations

import itertools
from typing import ClassVar

from sage.all import CombinatorialFreeModule

from uconf.algebraic.algebra import OperadAlgebra
from uconf.algebraic.tree_module import TreeModule
from uconf.core.operad import OperadLike
from uconf.core.vertex_decorated import VertexDecoratedLike
from uconf.core.trees import is_leaf, relabel_leaves


class FreeAlgebraModule(TreeModule):
    """Underlying dg-module of the free P-algebra ``P ∘ M``.

    Basis keys are ``(tree, m_tuple)`` pairs.  The differential is
    ``d = d_P + d_M`` using the DFS-interleaved Koszul sign rule.

    This class is normally not instantiated directly; use
    :class:`FreeOperadAlgebra` instead.
    """

    name: ClassVar[str] = "Free"

    def __init__(
        self,
        operad_cls: VertexDecoratedLike,
        inner_module,
        base_ring,
        *,
        vertex_degree_shift: int = 0,
        name: str | None = None,
    ):
        """Initialize the free P-algebra module ``P ∘ M``.

        Args:
            operad_cls: Arity-indexed vertex-decoration provider used on
                internal vertices (typically an operad, but may be any object
                matching the shared structural protocol).
            inner_module: Generating dg-module M (a ``CombinatorialFreeModule``).
            base_ring: Coefficient ring.
            vertex_degree_shift: Per-vertex degree offset (0 = standard free,
                +1 = bar/suspension convention, -1 = cobar/desuspension).
            name: Display name override.  Defaults to ``P ∘ M``.

        """
        if name is None:
            name = f"{operad_cls.name} ∘ {inner_module}"
        super().__init__(
            symmetric_sequence_cls=operad_cls,
            inner_module=inner_module,
            base_ring=base_ring,
            vertex_degree_shift=vertex_degree_shift,
            name=name,
        )
        # Backward-compatible alias expected by subclasses and callers.
        self._operad_cls = operad_cls


class FreeOperadAlgebra(OperadAlgebra):
    """Free P-algebra on a dg-module M.

    Constructs the composite product ``P ∘ M`` as a
    :class:`FreeAlgebraModule` and equips it with the canonical P-algebra
    structure given by grafting (operadic composition on tree decorations).

    Args:
        operad_cls: Operad provider P (class or wrapper instance).
        inner_module: The generating dg-module M.
        base_ring: Coefficient ring.

    The inclusion ``η: M → Free_P(M)`` is::

        m_key  ↦  free_algebra.module.term((1, (m_key,)))

    Examples::

        free_lie = FreeOperadAlgebra(Lie, module_M, R)
        # Apply the Lie bracket to two generators:
        bracket = free_lie.act(Lie(2, R).term((1,)), [x, y])

    """

    def __init__(self, operad_cls: OperadLike, inner_module: CombinatorialFreeModule, base_ring):
        free_module = FreeAlgebraModule(operad_cls, inner_module, base_ring)
        super().__init__(free_module, operad_cls, self._act_impl)
        self._inner_module = inner_module

    def _act_impl(self, p_element, algebra_elements):
        """P-algebra action γ(p; x_1, ..., x_k) by grafting.

        Grafts the k input subtrees ``x_j ∈ Free_P(M)`` under a new root
        vertex decorated by ``p ∈ P(k)``.  Leaf labels of each ``x_j`` are
        shifted by ``Σ_{l<j} |x_l|`` and the M-tuples are concatenated.

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

        result = self.module.zero()
        input_term_lists = [list(x) for x in inputs]

        for p_key, p_coeff in p_element:
            for term_combo in itertools.product(*input_term_lists):
                input_keys = [bk for (bk, _) in term_combo]
                coeff = p_coeff
                for _, c in term_combo:
                    coeff = coeff * c
                new_tree, new_m_tuple = self._graft(p_key, input_keys)
                result += coeff * self.module.term((new_tree, new_m_tuple))

        return result

    @staticmethod
    def _graft(p_key, input_keys):
        """Build ``(new_tree, new_m_tuple)`` by grafting ``input_keys`` under ``p_key``.

        Each ``input_keys[j]`` is a ``(tree_j, m_j)`` basis key of
        ``FreeAlgebraModule``.  Leaves of ``tree_j`` are relabeled by the
        offset ``Σ_{l<j} n_l`` so that the full tree has leaves
        ``{1, ..., Σ n_j}``.

        Returns:
            A ``(new_tree, new_m_tuple)`` pair that is a valid
            ``FreeAlgebraModule`` basis key.

        """
        n_list = [len(k[1]) for k in input_keys]
        offsets = []
        running = 0
        for n in n_list:
            offsets.append(running)
            running += n

        new_children = []
        new_m: list = []
        for j, (tree_j, m_j) in enumerate(input_keys):
            off = offsets[j]
            if is_leaf(tree_j):
                new_children.append(off + 1)
            else:
                relabel = {leaf: leaf + off for leaf in range(1, n_list[j] + 1)}
                new_children.append(relabel_leaves(tree_j, relabel))
            new_m.extend(m_j)

        new_tree = (p_key,) + tuple(new_children)
        return new_tree, tuple(new_m)

    def include(self, m_key):
        """Return the image of basis key ``m_key`` under the inclusion η: M → P ∘ M.

        Args:
            m_key: A basis key of the inner module M.

        Returns:
            The element ``module.term((1, (m_key,)))``.

        """
        return self.module.term((1, (m_key,)))
