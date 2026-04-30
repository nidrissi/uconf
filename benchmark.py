import cProfile
import os
import pstats
import time
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


def _profile_chain_complex_pass(pass_name: str) -> tuple[float, cProfile.Profile, list[str]]:
    worker_profile_paths: list[str] = []
    profile = cProfile.Profile()
    start = time.perf_counter()
    profile.enable()
    compute_chain_complex(
        mod,
        degrees=degs,
        weight=w,
        n_jobs=n_jobs,
        worker_profile_paths=worker_profile_paths,
        worker_profile_parent=profile,
        progress=True,
    )
    profile.disable()
    elapsed = time.perf_counter() - start
    print(f"{pass_name} pass: {elapsed:.4f}s")
    return elapsed, profile, worker_profile_paths


cold_elapsed, cold_profile, cold_worker_profiles = _profile_chain_complex_pass("Cold")
warm_elapsed, warm_profile, warm_worker_profiles = _profile_chain_complex_pass("Warm")

report_path = Path("benchmark_profile.txt")

with report_path.open("w") as report_file:
    report_file.write(f"Date: {datetime.now()}\n")
    report_file.write(f"Test run on {mod} with weight {w} using {n_jobs} jobs\n")
    for k in degs:
        report_file.write(f"Degree {k}: {len(mod.graded_basis_by_weight(k, w))} elements\n")
    report_file.write(f"Cold pass wall time: {cold_elapsed:.4f}s\n")
    report_file.write(f"Warm pass wall time: {warm_elapsed:.4f}s\n")
    report_file.write(f"Cold→warm cache-fill overhead: {max(cold_elapsed - warm_elapsed, 0.0):.4f}s\n")

    for pass_name, profile, worker_profile_paths in (
        ("Cold", cold_profile, cold_worker_profiles),
        ("Warm", warm_profile, warm_worker_profiles),
    ):
        report_file.write("\n" + "=" * 80 + "\n")
        report_file.write(f"{pass_name} pass profile\n")
        if worker_profile_paths:
            report_file.write(f"Merging {len(worker_profile_paths)} worker profile(s)\n")
        stats = pstats.Stats(profile, stream=report_file)
        for worker_profile_path in worker_profile_paths:
            stats.add(worker_profile_path)
        stats.sort_stats("cumulative").strip_dirs().print_stats()

for worker_profile_path in cold_worker_profiles + warm_worker_profiles:
    os.unlink(worker_profile_path)

print(f"Profile written to {report_path}")
