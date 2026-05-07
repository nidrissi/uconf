#!python

import cProfile
import os
import pstats
import time
from datetime import datetime
from pathlib import Path
import argparse

from sage.all import GF

from uconf import compute_chain_complex, euclidean_unordered_configuration_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "compute.py", description="Compute the chain complex of the unordered configuration model."
    )
    parser.add_argument("--dim", "-d", type=int, default=2, help="The dimension of the sphere.")
    parser.add_argument(
        "--weight", "-w", type=int, default=2, help="The weight of the configuration."
    )
    parser.add_argument("--deg_max", "-m", type=int, default=3, help="The maximum degree.")
    parser.add_argument(
        "--jobs", "-j", type=int, default=1, help="The number of parallel jobs to use."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print timestamped phase diagnostics to stderr."
    )
    args = parser.parse_args()

    w = args.weight
    degs = range(-1, args.deg_max + 1)
    n_jobs = args.jobs
    dim = args.dim
    verbose = args.verbose
    print(f"dim={dim}, weight={w}, degs={list(degs)}, n_jobs={n_jobs}")

    if verbose:
        import sys as _sys
        import time as _time
        print(f"[{_time.strftime('%H:%M:%S')}] building model...", file=_sys.stderr, flush=True)
    model = euclidean_unordered_configuration_model(GF(2), dim)
    mod = model.module
    if verbose:
        print(f"[{_time.strftime('%H:%M:%S')}] model ready", file=_sys.stderr, flush=True)

    worker_profile_paths: list[str] = []
    profile = cProfile.Profile()
    start = time.perf_counter()
    profile.enable()
    cc = compute_chain_complex(
        mod,
        degrees=degs,
        weight=w,
        n_jobs=n_jobs,
        worker_profile_paths=worker_profile_paths,
        worker_profile_parent=profile,
        progress=True,
        verbose=verbose,
    )
    profile.disable()
    elapsed = time.perf_counter() - start

    path_suffix = f"{dim}_{w}_{args.deg_max}_{n_jobs}"
    report_path = Path(
        f"dump/benchmark_profile_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{path_suffix}.txt"
    )
    csv_path = Path(f"dump/cc_F2_{path_suffix}.csv")
    cc_path = Path(f"dump/cc_F2_{path_suffix}")

    with report_path.open("w") as report_file:
        report_file.write(f"Date: {datetime.now()}\n")
        report_file.write(f"Test run on {mod} with weight {w} using {n_jobs} jobs\n")
        for k in degs:
            report_file.write(f"Degree {k}: {len(mod.graded_basis_by_weight(k, w))} elements\n")
        report_file.write(f"Elapsed time: {elapsed:.4f}s\n")

        report_file.write("\n" + "=" * 80 + "\n")
        if worker_profile_paths:
            report_file.write(f"Merging {len(worker_profile_paths)} worker profile(s)\n")
        stats = pstats.Stats(profile, stream=report_file)
        for worker_profile_path in worker_profile_paths:
            stats.add(worker_profile_path)
        stats.strip_dirs().sort_stats("cumulative").print_stats()

    for worker_profile_path in worker_profile_paths:
        os.unlink(worker_profile_path)

    print(f"Profile written to {report_path}")

    with csv_path.open("w") as f:
        print("d,dim,betti", file=f)
        for d in degs:
            print(f"{d},{cc.free_module_rank(d)},{cc.betti(d)}", file=f)
        print(f"Betti numbers saved to {csv_path}")

    cc.save(cc_path)
    print(f"Chain complex save to {cc_path}.sobj")
