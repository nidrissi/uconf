"""Tests for the TikZ/forest tree renderer in :mod:`uconf.tikz`."""

from sage.all import QQ

from uconf import (
    BarConstruction,
    CobarConstruction,
    CoAssociative,
    Lie,
    Surjection,
    element_to_tikz,
    tree_to_forest,
)
from uconf.core.trees import RootedTree
from uconf.tikz import (
    STYLE_BLACK_VERTEX,
    STYLE_BLUE_BOX,
    STYLE_DASHED_EDGE,
    STYLE_RED_EDGE,
    STYLE_RED_VERTEX,
    Layer,
    default_layer_for_depth,
)


# ---------------------------------------------------------------------------
# tree_to_forest: low-level primitive
# ---------------------------------------------------------------------------


def _stub_formatter(dec, arity):
    if isinstance(dec, tuple) and len(dec) == 1:
        return f"d{dec[0]}/a{arity}"
    return f"d{dec}/a{arity}"


def test_tree_to_forest_leaf_only():
    """A bare leaf integer renders as ``[{1}, lf]``."""
    out = tree_to_forest(1, decoration_formatter=_stub_formatter)
    assert out == "[{1}, lf]"


def test_tree_to_forest_default_layer_is_red():
    """The default layer (depth 0) uses red vertex + red edge styles."""
    tree = RootedTree((42,), 1, 2)
    out = tree_to_forest(tree, decoration_formatter=_stub_formatter)
    # Root vertex rv, no edge style at root
    assert out.startswith("[{d42/a2}, rv[")
    # Children are leaves with rv-edge into the lf node
    assert "[{1}, lf, re]" in out
    assert "[{2}, lf, re]" in out


def test_tree_to_forest_explicit_layer():
    """Caller can swap to the black/dashed layer for cobar-style rendering."""
    tree = RootedTree((7,), 1, 2)
    layer = Layer(STYLE_BLACK_VERTEX, STYLE_DASHED_EDGE)
    out = tree_to_forest(tree, decoration_formatter=_stub_formatter, layer=layer)
    assert "rv" not in out
    assert "[{d7/a2}, bv[" in out
    # The two leaves carry the dashed edge style
    assert "[{1}, lf, de]" in out
    assert "[{2}, lf, de]" in out


def test_tree_to_forest_nested():
    """Nested internal vertices each get the vertex style; inner edge style."""
    inner = RootedTree((1,), 2, 3)
    tree = RootedTree((0,), 1, inner)
    out = tree_to_forest(tree, decoration_formatter=_stub_formatter)
    # Outer root: rv with no edge
    assert out.startswith("[{d0/a2}, rv[")
    # Inner vertex: rv with re edge from outer root
    assert "[{d1/a2}, rv, re[" in out


def test_tree_to_forest_leaf_renderer_overrides_edge_styling():
    """A leaf_renderer fragment is inserted verbatim -- layer transitions
    use default (solid) edges, matching the article's convention."""
    tree = RootedTree((9,), 1, 2)

    def renderer(leaf_label):
        return f"[{{custom-{leaf_label}}}, {STYLE_BLUE_BOX}]"

    out = tree_to_forest(tree, decoration_formatter=_stub_formatter, leaf_renderer=renderer)
    assert "[{custom-1}, bx]" in out
    assert "[{custom-2}, bx]" in out
    # The leaf_renderer output is NOT given a red edge style
    assert "bx, re" not in out


# ---------------------------------------------------------------------------
# element_to_tikz: integration with bar / cobar components
# ---------------------------------------------------------------------------


def test_element_to_tikz_bar_construction_lie():
    """A bar element of Lie produces a forest block with rv vertices."""
    BLie = BarConstruction(Lie)
    B2 = BLie(2, QQ)
    tree = RootedTree((1,), 1, 2)  # Lie arity-2 tree
    elem = B2(tree)

    out = element_to_tikz(elem)
    assert "\\begin{forest} uconf tree" in out
    assert "\\end{forest}" in out
    assert "rv" in out
    assert "lf" in out


def test_element_to_tikz_cobar_construction_uses_dashed():
    """A cobar element with >1 internal vertices uses bv vertices + de edges."""
    OmegaCoAss = CobarConstruction(CoAssociative)
    O3 = OmegaCoAss(3, QQ)
    # CoAssociative arity-2 basis keys are permutations of (1, 2).
    inner = RootedTree((1, 2), 1, 2)
    bigger = RootedTree((1, 2), inner, 3)  # arity-3 tree, 2 internal vertices
    elem = O3(bigger)

    out = element_to_tikz(elem)
    assert STYLE_BLACK_VERTEX in out
    assert STYLE_DASHED_EDGE in out


def test_element_to_tikz_linear_combination():
    """Sum of two distinct basis elements emits two forest blocks plus ``+``."""
    BSurj = BarConstruction(Surjection)
    B2 = BSurj(2, QQ)
    # Two distinct Surjection arity-2 basis keys.
    tree_a = RootedTree((1, 2), 1, 2)
    tree_b = RootedTree((2, 1), 1, 2)
    elem = B2(tree_a) + B2(tree_b)
    assert len(list(elem)) == 2, "Sum must have two distinct basis terms"

    out = element_to_tikz(elem)
    assert out.count("\\begin{forest}") == 2
    assert " + " in out


def test_element_to_tikz_scalar_prefix():
    """Coefficient != 1 prefixes the forest block."""
    BLie = BarConstruction(Lie)
    B2 = BLie(2, QQ)
    tree = RootedTree((1,), 1, 2)
    elem = QQ(3) * B2(tree)

    out = element_to_tikz(elem)
    assert "3" in out and "\\cdot" in out


def test_element_to_tikz_no_env():
    """``env=False`` strips the forest environment wrapper."""
    BLie = BarConstruction(Lie)
    B2 = BLie(2, QQ)
    tree = RootedTree((1,), 1, 2)
    elem = B2(tree)

    out = element_to_tikz(elem, env=False)
    assert "\\begin{forest}" not in out
    assert out.startswith("[")


# ---------------------------------------------------------------------------
# Layer dispatch
# ---------------------------------------------------------------------------


def test_default_layer_depth_0_is_red():
    layer = default_layer_for_depth(0)
    assert layer.vertex_style == STYLE_RED_VERTEX
    assert layer.edge_style == STYLE_RED_EDGE


def test_default_layer_depth_1_is_cobar():
    layer = default_layer_for_depth(1)
    assert layer.vertex_style == STYLE_BLACK_VERTEX
    assert layer.edge_style == STYLE_DASHED_EDGE


def test_default_layer_depth_2_is_inner_bar():
    layer = default_layer_for_depth(2)
    assert layer.vertex_style == STYLE_BLACK_VERTEX
    assert layer.edge_style is None


# ---------------------------------------------------------------------------
# reps_to_tex_document
# ---------------------------------------------------------------------------


def test_reps_to_tex_document_structure():
    """A reps map emits one labelled block per representative, in degree order."""
    from uconf import reps_to_tex_document

    BLie = BarConstruction(Lie)
    B2 = BLie(2, QQ)
    tree = RootedTree((1,), 1, 2)
    elem = B2(tree)

    doc = reps_to_tex_document({0: [elem], 1: [elem, elem]}, header_comment="hello")
    assert "% hello" in doc
    assert "% Requires: \\usepackage{uconf-trees}" in doc
    assert "Degree 0: 1 representative(s)" in doc
    assert "Degree 1: 2 representative(s)" in doc
    # Three forest blocks total
    assert doc.count("\\begin{forest}") == 3
