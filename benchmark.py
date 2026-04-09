import cProfile
import pstats
from datetime import datetime
from pathlib import Path

from sage.all import GF

from uconf.algebraic.configuration import _build_euclidean_layers
from uconf.homology import compute_chain_complex

layers = _build_euclidean_layers(GF(2), 2)
obj = layers.bar
mod = obj.module

w = 3
degs = range(-1, 5)


profile = cProfile.Profile()
profile.enable()
# ---------------
compute_chain_complex(mod, degrees=degs, weight=3)
# ----------------
profile.disable()

report_path = Path("benchmark_profile.txt")

with report_path.open("w") as report_file:
    report_file.write(f"Date: {datetime.now()}\n")
    report_file.write(f"Test run on {mod} with weight {w}\n")
    for k in degs:
        report_file.write(f"Degree {k}: {len(mod.graded_basis_by_weight(k, w))} elements\n")
    pstats.Stats(profile, stream=report_file).sort_stats("cumulative").print_stats()

print(f"Profile written to {report_path}")
