#!python

import cProfile
import os
import pstats
import sys
import time
from datetime import datetime
from pathlib import Path
import argparse

from sage.all import GF, save

from uconf import compute_chain_complex, euclidean_unordered_configuration_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the chain complex of the unordered configuration model."
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
    parser.add_argument(
        "--no-prewarm", action="store_true", help="Disable cache prewarm before forking workers."
    )
    parser.add_argument(
        "--no-profile", action="store_true",
        help="Disable cProfile (faster runs when profiling data is not needed).",
    )
    args = parser.parse_args()

    w = args.weight
    degs = range(-1, args.deg_max + 1)
    n_jobs = args.jobs
    dim = args.dim
    verbose = args.verbose
    do_prewarm = not args.no_prewarm
    do_profile = not args.no_profile
    print(f"dim={dim}, weight={w}, degs={list(degs)}, n_jobs={n_jobs}", flush=True)

    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] building model...", file=sys.stderr, flush=True)
    model = euclidean_unordered_configuration_model(GF(2), dim)
    mod = model.module
    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] model ready", file=sys.stderr, flush=True)

    worker_profile_paths: list[str] | None = [] if do_profile else None
    profile: cProfile.Profile | None = cProfile.Profile() if do_profile else None
    prewarm_profile: cProfile.Profile | None = cProfile.Profile() if do_profile else None

    start = time.perf_counter()
    if profile is not None:
        profile.enable()
    cc = compute_chain_complex(
        mod,
        degrees=degs,
        weight=w,
        n_jobs=n_jobs,
        prewarm=do_prewarm,
        prewarm_profiler=prewarm_profile,
        worker_profile_paths=worker_profile_paths,
        worker_profile_parent=profile,
        progress=True,
        verbose=verbose,
    )
    if profile is not None:
        profile.disable()
    elapsed = time.perf_counter() - start

    path_suffix = f"{dim}_{w}_{args.deg_max}_{n_jobs}"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = Path(f"dump/cc_F2_{path_suffix}.csv")
    cc_path = Path(f"dump/cc_F2_{path_suffix}")

    if do_profile:
        assert profile is not None
        assert prewarm_profile is not None
        assert worker_profile_paths is not None

        report_path = Path(f"dump/benchmark_profile_{timestamp}_{path_suffix}.txt")
        prewarm_report_path = Path(f"dump/benchmark_prewarm_profile_{timestamp}_{path_suffix}.txt")

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
            for wp in worker_profile_paths:
                stats.add(wp)
            stats.strip_dirs().sort_stats("cumulative").print_stats()

        for wp in worker_profile_paths:
            os.unlink(wp)

        print(f"Profile written to {report_path}")

        if prewarm_profile.getstats():
            with prewarm_report_path.open("w") as prewarm_file:
                prewarm_file.write(f"Date: {datetime.now()}\n")
                prewarm_file.write(f"Prewarm profile for {mod} weight={w} n_jobs={n_jobs}\n")
                prewarm_file.write(f"Prewarm enabled: {do_prewarm}\n\n")
                prewarm_file.write("=" * 80 + "\n")
                pstats.Stats(prewarm_profile, stream=prewarm_file).strip_dirs().sort_stats(
                    "cumulative"
                ).print_stats()
            print(f"Prewarm profile written to {prewarm_report_path}")
        else:
            print("Prewarm profiler inactive (n_jobs=1 or prewarm disabled) — no prewarm profile written")
    else:
        print(f"Elapsed time: {elapsed:.4f}s")

    with csv_path.open("w") as f:
        print("d,dim,betti", file=f)
        for d in degs:
            print(f"{d},{cc.free_module_rank(d)},{cc.betti(d)}", file=f)
        print(f"Betti numbers saved to {csv_path}")

    cc.save(cc_path)
    print(f"Chain complex saved to {cc_path}.sobj")

    bases_path = Path(f"dump/bases_F2_{path_suffix}.sobj")
    bases = {d: list(mod.graded_basis_by_weight(d, w)) for d in degs}
    save(bases, bases_path)
    print(f"Graded bases saved to {bases_path}")
