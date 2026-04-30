import cProfile
import os
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
n_jobs = 8

worker_profile_paths: list[str] = []

profile = cProfile.Profile()
profile.enable()
# ---------------
compute_chain_complex(
    mod,
    degrees=degs,
    weight=w,
    n_jobs=n_jobs,
    worker_profile_paths=worker_profile_paths,
    worker_profile_parent=profile,
    progress=True,
)
# ----------------
profile.disable()

report_path = Path("benchmark_profile.txt")

with report_path.open("w") as report_file:
    report_file.write(f"Date: {datetime.now()}\n")
    report_file.write(f"Test run on {mod} with weight {w} using {n_jobs} jobs\n")
    for k in degs:
        report_file.write(f"Degree {k}: {len(mod.graded_basis_by_weight(k, w))} elements\n")
    if worker_profile_paths:
        report_file.write(f"Merging {len(worker_profile_paths)} worker profile(s)\n")
    stats = pstats.Stats(profile, stream=report_file)
    for worker_profile_path in worker_profile_paths:
        stats.add(worker_profile_path)
    stats.sort_stats("cumulative").strip_dirs().print_stats()

for worker_profile_path in worker_profile_paths:
    os.unlink(worker_profile_path)

print(f"Profile written to {report_path}")
