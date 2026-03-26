from uconf import euclidean_unordered_configuration_model, compute_chain_complex
from sage.all import GF

model = euclidean_unordered_configuration_model(GF(2), 2)
cc = compute_chain_complex(model, degrees=range(0, 1), weight=4)
