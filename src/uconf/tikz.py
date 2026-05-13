"""TikZ/forest rendering for bar/cobar tree elements.

This module turns the SageMath element representation produced by
:mod:`uconf.constructions` and :mod:`uconf.algebraic` into compact
``forest``-style LaTeX snippets.

Two entry points are exposed:

- :func:`tree_to_forest`: render a single decorated
  :class:`~uconf.core.trees.RootedTree` as the body of a ``forest`` block.
  This is the low-level primitive and is independent of any module dispatch.

- :func:`element_to_tikz`: render an entire module element, dispatching on
  its parent type to choose appropriate styling for nested bar / cobar
  layers (red outer bar, dashed cobar, solid inner bar, etc.).

The output assumes ``tex/uconf-trees.sty`` is loaded, which defines the
style names ``rv``, ``bv``, ``bx``, ``lf``, ``re``, ``de`` and the
``uconf tree`` forest preset.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from uconf.core.trees import RootedTree, is_leaf

# Forest style names.  Must match those defined in tex/uconf-trees.sty.
STYLE_RED_VERTEX = "rv"
STYLE_BLACK_VERTEX = "bv"
STYLE_BLUE_BOX = "bx"
STYLE_LEAF = "lf"
STYLE_RED_EDGE = "re"
STYLE_DASHED_EDGE = "de"


# ---------------------------------------------------------------------------
# Layer dispatch
# ---------------------------------------------------------------------------


class Layer:
    """Styling choices for one layer of a nested bar/cobar element.

    Attributes:
        vertex_style: forest style for internal vertices in this layer
            (e.g. ``"rv"`` for red, ``"bv"`` for black).
        edge_style: forest style for the edge from each non-root vertex
            in this layer to its parent (e.g. ``"re"`` for red, ``"de"``
            for dashed, ``None`` for default solid black).
    """

    __slots__ = ("vertex_style", "edge_style")

    def __init__(self, vertex_style: str, edge_style: str | None):
        self.vertex_style = vertex_style
        self.edge_style = edge_style


def default_layer_for_depth(depth: int) -> Layer:
    """Return the default :class:`Layer` for nesting depth ``depth``.

    - ``depth == 0``: outer bar (red vertices, red edges)
    - ``depth == 1``: cobar (black vertices, dashed edges)
    - ``depth >= 2``: inner bar inside cobar (black vertices, solid edges)
    """
    if depth == 0:
        return Layer(STYLE_RED_VERTEX, STYLE_RED_EDGE)
    if depth == 1:
        return Layer(STYLE_BLACK_VERTEX, STYLE_DASHED_EDGE)
    return Layer(STYLE_BLACK_VERTEX, None)


# ---------------------------------------------------------------------------
# Low-level tree rendering
# ---------------------------------------------------------------------------


def _node_open(content: str, *styles: str | None) -> str:
    """Open a forest node: ``[{content}, style1, style2``.

    The caller is responsible for emitting any children and the closing ``]``.
    """
    pieces = [f"{{{content}}}"]
    pieces.extend(s for s in styles if s)
    return "[" + ", ".join(pieces)


def tree_to_forest(
    tree: RootedTree | int,
    *,
    decoration_formatter: Callable[[Any, int], str],
    layer: Layer | None = None,
    leaf_style: str = STYLE_LEAF,
    leaf_renderer: Callable[[int], str] | None = None,
) -> str:
    """Render a single decorated rooted tree as a ``forest`` body.

    The result starts with ``[`` and ends with ``]``; the caller wraps it in
    ``\\begin{forest} ... \\end{forest}``.

    Args:
        tree: A :class:`~uconf.core.trees.RootedTree` or a leaf ``int``.
        decoration_formatter: Callback ``(decoration, arity) -> str`` that
            produces the LaTeX text for each internal vertex's decoration.
        layer: Vertex and edge styling for this whole tree.  Defaults to
            the depth-0 layer (red vertices, red edges).
        leaf_style: Forest style to apply to leaf nodes when no
            ``leaf_renderer`` is given.
        leaf_renderer: Optional callback ``(leaf_label) -> forest_fragment``.
            When supplied, replaces the default leaf rendering and is
            responsible for returning a complete ``[...]`` fragment for the
            leaf (the edge style from the parent is the caller's
            responsibility -- the renderer is invoked at the leaf position).
    """
    if layer is None:
        layer = default_layer_for_depth(0)

    def render(node, *, is_root: bool) -> str:
        edge = None if is_root else layer.edge_style
        if is_leaf(node):
            if leaf_renderer is not None:
                # Layer transitions use default (unstyled) edges: the inner
                # layer's edge style applies within itself, and the outer
                # layer's edge style applies between same-layer vertices.
                return leaf_renderer(node)
            return _node_open(str(node), leaf_style, edge) + "]"
        arity = node._arity
        dec_str = decoration_formatter(node._decoration, arity)
        children_fragments = "".join(render(c, is_root=False) for c in node._children)
        return _node_open(dec_str, layer.vertex_style, edge) + children_fragments + "]"

    return render(tree, is_root=True)


# ---------------------------------------------------------------------------
# Element-level rendering with module dispatch
# ---------------------------------------------------------------------------


def _operad_decoration_formatter(operad_cls, base_ring) -> Callable[[Any, int], str]:
    """Build a decoration formatter that delegates to the operad's
    ``_latex_term`` per arity."""

    def fmt(dec: Any, arity: int) -> str:
        comp = operad_cls(arity, base_ring)
        latex_term = getattr(comp, "_latex_term", None)
        if callable(latex_term):
            return latex_term(dec)
        repr_term = getattr(comp, "_repr_term", None)
        if callable(repr_term):
            return repr_term(dec)
        return str(dec)

    return fmt


def _is_bar_construction_component(parent: Any) -> bool:
    from uconf.constructions.bar_construction import BarConstruction

    return isinstance(parent, BarConstruction.Component)


def _is_cobar_construction_component(parent: Any) -> bool:
    from uconf.constructions.cobar_construction import CobarConstruction

    return isinstance(parent, CobarConstruction.Component)


def _is_cofree_or_free_module(parent: Any) -> bool:
    """Modules whose basis keys are ``(c_or_p_key, m_tuple)`` pairs.

    Covers :class:`CofreeCoalgebraModule` (e.g. ``BarAlgebraModule``) and
    :class:`FreeAlgebraModule`.
    """
    from uconf.algebraic.cofree_coalgebra import CofreeCoalgebraModule
    from uconf.algebraic.free_algebra import FreeAlgebraModule

    return isinstance(parent, (CofreeCoalgebraModule, FreeAlgebraModule))


def _basis_key_to_forest(
    key: Any,
    parent: Any,
    *,
    depth: int,
) -> str:
    """Render a single basis key of ``parent`` as a forest fragment.

    Dispatches on ``type(parent)``:

    - Bar / cobar component: ``key`` is a :class:`RootedTree`; render as a
      tree at the appropriate layer depth.
    - Cofree / free module: ``key`` is ``(tree_key, m_tuple)``; render the
      tree, recursing into each leaf via the inner module.
    - Tensor module (Sage ``CombinatorialFreeModuleTensor``): ``key`` is a
      tuple of factor keys; emit each factor in sequence, with the first
      factor as a blue box and the rest recursed into.
    - Anything else: emit a single leaf-style node using ``_latex_term`` or
      ``_repr_term`` (falling back to ``str``).
    """
    if _is_bar_construction_component(parent):
        # Outermost bar (depth 0) uses red styling; bar nested inside cobar
        # (depth >= 2 in the typical bar-of-cobar-of-bar pattern) uses
        # solid black edges.
        layer = (
            Layer(STYLE_RED_VERTEX, STYLE_RED_EDGE)
            if depth == 0
            else Layer(STYLE_BLACK_VERTEX, None)
        )
        fmt = _operad_decoration_formatter(parent.factory.operad_cls, parent.base_ring())
        return tree_to_forest(key, decoration_formatter=fmt, layer=layer)

    if _is_cobar_construction_component(parent):
        # Cobar always uses dashed black edges, regardless of depth.
        layer = Layer(STYLE_BLACK_VERTEX, STYLE_DASHED_EDGE)
        fmt = _operad_decoration_formatter(parent.factory.cooperad_cls, parent.base_ring())
        return tree_to_forest(key, decoration_formatter=fmt, layer=layer)

    if _is_cofree_or_free_module(parent):
        return _render_corolla(key, parent, depth=depth)

    # Sage tensor module: basis keys are tuples of factor keys, with
    # ``tensor_factors()`` exposing the component modules.
    factors = getattr(parent, "tensor_factors", None)
    if callable(factors) and isinstance(key, tuple):
        return _render_tensor(key, parent, depth=depth)

    # Fallback: emit the key's latex/repr as a single leaf node.
    label = _module_term_latex(parent, key)
    return _node_open(label, STYLE_LEAF) + "]"


def _underlying_operad_cls(operad_cls: Any) -> Any:
    """If ``operad_cls`` is a Bar/Cobar construction wrapper, return the
    underlying (co)operad whose basis decorates the tree vertices.

    Vertex decorations of a :class:`BarConstruction(P)` basis tree are
    ``P``-elements, not bar trees — so to render those decorations via a
    per-vertex formatter we need ``P``'s ``_latex_term``, not the bar
    construction's (which expects a whole tree).
    """
    from uconf.constructions.bar_construction import BarConstruction
    from uconf.constructions.cobar_construction import CobarConstruction

    if isinstance(operad_cls, BarConstruction):
        return operad_cls.operad_cls
    if isinstance(operad_cls, CobarConstruction):
        return operad_cls.cooperad_cls
    return operad_cls


def _render_corolla(key: tuple, parent: Any, *, depth: int) -> str:
    """Render a ``(tree_key, m_tuple)`` corolla of a cofree/free module.

    Emits the tree with its layer styling; each leaf integer ``i`` is
    rendered by recursing into ``parent._inner_module`` with basis key
    ``m_tuple[i-1]``.
    """
    tree_key, m_tuple = key
    operad_cls = getattr(parent, "_cooperad_cls", None) or parent._operad_cls
    # When the corolla's (co)operad is itself a Bar/Cobar construction, the
    # basis tree's vertex decorations come from the *underlying* (co)operad —
    # use that for the per-vertex formatter so we don't try to render a single
    # vertex decoration as if it were a whole tree.
    underlying = _underlying_operad_cls(operad_cls)
    layer = default_layer_for_depth(depth)
    fmt = _operad_decoration_formatter(underlying, parent.base_ring())

    inner = parent._inner_module

    def leaf_renderer(leaf_label: int) -> str:
        idx = leaf_label - 1
        if not (0 <= idx < len(m_tuple)):
            return _node_open(str(leaf_label), STYLE_LEAF) + "]"
        return _basis_key_to_forest(m_tuple[idx], inner, depth=depth + 1)

    return tree_to_forest(
        tree_key,
        decoration_formatter=fmt,
        layer=layer,
        leaf_renderer=leaf_renderer,
    )


def _render_tensor(key: tuple, parent: Any, *, depth: int) -> str:
    """Render a tensor basis key.

    The first factor is rendered as a blue box (typical role: manifold-
    chain / coefficients factor).  Remaining factors are recursed into,
    with the previous fragment as their forest parent.  The visual is a
    chain ``[bx]-[recursed]-...``.
    """
    factors = parent.tensor_factors()
    if not isinstance(key, tuple) or len(key) != len(factors):
        label = _module_term_latex(parent, key)
        return _node_open(label, STYLE_LEAF) + "]"

    # Render each factor, then nest them as parent -> child -> ...
    fragments: list[str] = []
    for factor_module, factor_key in zip(factors, key):
        if _is_blue_box_module(factor_module):
            label = _module_term_latex(factor_module, factor_key)
            fragments.append(_node_open(label, STYLE_BLUE_BOX) + "]")
        else:
            fragments.append(_basis_key_to_forest(factor_key, factor_module, depth=depth))

    # Nest: fragments[0] gets fragments[1] as its child, etc.  We achieve
    # this by stripping the closing ']' of each except the last and
    # appending the next.
    result = fragments[-1]
    for frag in reversed(fragments[:-1]):
        if frag.endswith("]"):
            result = frag[:-1] + result + "]"
        else:
            result = frag + result
    return result


def _is_blue_box_module(module: Any) -> bool:
    """Heuristic: modules whose basis keys are not trees and that represent
    'coefficient' / 'manifold-chain' factors get rendered as a single blue
    box rather than recursed into.  We treat anything that is not a
    cofree/free module or bar/cobar component as a blue box.
    """
    if _is_bar_construction_component(module):
        return False
    if _is_cobar_construction_component(module):
        return False
    if _is_cofree_or_free_module(module):
        return False
    if callable(getattr(module, "tensor_factors", None)):
        return False
    return True


def _module_term_latex(module: Any, key: Any) -> str:
    latex_term = getattr(module, "_latex_term", None)
    if callable(latex_term):
        try:
            return latex_term(key)
        except Exception:
            pass
    repr_term = getattr(module, "_repr_term", None)
    if callable(repr_term):
        try:
            return repr_term(key)
        except Exception:
            pass
    return str(key)


# ---------------------------------------------------------------------------
# Public element-level entry point
# ---------------------------------------------------------------------------


def element_to_tikz(
    element: Any,
    *,
    env: bool = True,
    forest_preset: str = "uconf tree",
) -> str:
    """Render a module element as a ``forest`` LaTeX snippet.

    The element is iterated as a linear combination ``Σ c_i · b_i``.  Each
    basis term is rendered as a separate ``\\begin{forest} ... \\end{forest}``
    block, joined by ``+`` (with leading scalar where the coefficient is not
    ``1``).

    Args:
        element: A Sage free-module element whose parent is a (nested)
            bar / cobar / cofree / tensor module supported by this module.
        env: When ``True`` (default), wrap each tree in
            ``\\begin{forest} <forest_preset> ... \\end{forest}``.  When
            ``False``, emit just the forest bodies (useful for piping into
            an existing forest environment).
        forest_preset: Forest style preset name applied to each block.
            Must be defined in ``tex/uconf-trees.sty`` (default
            ``"uconf tree"``).
    """
    parent = element.parent()

    terms = list(element)
    if not terms:
        return "0"

    rendered: list[str] = []
    for basis_key, coeff in terms:
        body = _basis_key_to_forest(basis_key, parent, depth=0)
        if env:
            block = f"\\begin{{forest}} {forest_preset}\n{body}\n\\end{{forest}}"
        else:
            block = body
        prefix = _format_coefficient(coeff)
        rendered.append(prefix + block if prefix else block)

    return " + ".join(rendered).replace("+ -", "- ")


def _format_coefficient(coeff: Any) -> str:
    """Return a LaTeX prefix for ``coeff`` (empty string when ``coeff == 1``)."""
    if coeff == 1:
        return ""
    if coeff == -1:
        return "-"
    return f"{coeff} \\cdot "


# ---------------------------------------------------------------------------
# Bulk export helper for homology_repr.py
# ---------------------------------------------------------------------------


def reps_to_tex_document(
    representatives: dict[int, list],
    *,
    header_comment: str = "",
) -> str:
    """Format a ``{degree: [element, ...]}`` map as a single ``.tex`` file.

    The file is a self-contained fragment intended to be ``\\input`` into a
    document that has ``\\usepackage{uconf-trees}`` loaded.  Each
    representative is wrapped in a labelled comment and a ``forest`` block.
    """
    lines: list[str] = []
    if header_comment:
        for line in header_comment.splitlines():
            lines.append(f"% {line}")
        lines.append("")
    lines.append("% Requires: \\usepackage{uconf-trees}")
    lines.append("")
    for degree in sorted(representatives):
        reps = representatives[degree]
        lines.append(f"% ===== Degree {degree}: {len(reps)} representative(s) =====")
        for i, rep in enumerate(reps):
            lines.append(f"% --- degree {degree}, representative {i} ---")
            lines.append(element_to_tikz(rep))
            lines.append("")
    return "\n".join(lines) + "\n"


__all__ = [
    "Layer",
    "default_layer_for_depth",
    "tree_to_forest",
    "element_to_tikz",
    "reps_to_tex_document",
    "STYLE_RED_VERTEX",
    "STYLE_BLACK_VERTEX",
    "STYLE_BLUE_BOX",
    "STYLE_LEAF",
    "STYLE_RED_EDGE",
    "STYLE_DASHED_EDGE",
]
