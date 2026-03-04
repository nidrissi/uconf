"""Canonical operad/cooperad and simplicial model classes."""

from uconf.models.associative import Associative
from uconf.models.barratt_eccles import BarrattEccles
from uconf.models.coassociative import CoAssociative
from uconf.models.cocommutative import CoCommutative
from uconf.models.commutative import Commutative
from uconf.models.lie import Lie
from uconf.models.simplicial import SimplicialChains, SimplicialCochains
from uconf.models.surjection import Surjection
from uconf.models.surjection_dual import SurjectionDual

__all__ = [
    "Associative",
    "BarrattEccles",
    "CoAssociative",
    "CoCommutative",
    "Commutative",
    "Lie",
    "SimplicialChains",
    "SimplicialCochains",
    "Surjection",
    "SurjectionDual",
]
