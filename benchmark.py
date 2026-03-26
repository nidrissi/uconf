from uconf import euclidean_unordered_configuration_model, compute_chain_complex
from sage.all import QQ

from sage.all import *
# or more specifically:
from sage.rings.finite_rings.finite_field_constructor import GF

#Compute the chain complex over any field:

F = GF(2)

model = euclidean_unordered_configuration_model(F, 2)
cc = compute_chain_complex(model, degrees=range(-2, 4), weight = 3)



