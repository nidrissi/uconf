"""Regression tests for explicit Element repr methods on model classes."""

from sage.all import QQ

from uconf import Associative, BarrattEccles, Commutative, Lie, Surjection
from uconf.algebraic.cofree_coalgebra import CofreeCoalgebraModule
from uconf.models.coassociative import CoAssociative
from uconf.models.simplicial import SimplicialChains, SimplicialCochains


def _assert_latex_repr(elem):
    ltx = elem._repr_latex_()
    assert ltx


def test_associative_element_repr_latex():
    _assert_latex_repr(Associative(2, QQ)((1, 2)))


def test_commutative_element_repr_latex():
    _assert_latex_repr(Commutative(2, QQ)(()))


def test_surjection_element_repr_latex():
    _assert_latex_repr(Surjection(2, QQ)((1, 2, 1)))


def test_barratt_eccles_element_repr_latex():
    _assert_latex_repr(BarrattEccles(2, QQ)(((1, 2),)))


def test_lie_element_repr_latex():
    _assert_latex_repr(Lie(3, QQ)((1, 2)))


def test_simplicial_chain_element_repr_latex():
    _assert_latex_repr(SimplicialChains(QQ)((0, 1, 2)))


def test_simplicial_cochain_element_repr_latex():
    _assert_latex_repr(SimplicialCochains(3, QQ)((0, 2)))


def test_cofree_module_element_repr_latex():
    inner = SimplicialChains(QQ)
    cofree = CofreeCoalgebraModule(CoAssociative, inner)
    _assert_latex_repr(cofree(((1,), ((0, 1),))))
