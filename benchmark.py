from uconf import euclidean_unordered_configuration_model, compute_chain_complex
from sage.all import QQ

model = euclidean_unordered_configuration_model(QQ, 2)
cc = compute_chain_complex(model, degrees=range(-2, 4), weight=3)
