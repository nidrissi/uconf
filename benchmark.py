import time

from sage.all import GF

from uconf import compute_chain_complex, euclidean_unordered_configuration_model

model = euclidean_unordered_configuration_model(GF(2), 2)

start = time.time()
cc = compute_chain_complex(model.module, degrees=range(-1, 1), weight=4)
print(f"Time taken: {time.time() - start:.2f} seconds")
