from uconf import euclidean_unordered_configuration_model, chain_complex
from sage.all import QQ

model = euclidean_unordered_configuration_model(QQ, 2)
cc = chain_complex(model, degrees=range(-2, 3), weight=2)
