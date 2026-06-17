#!python

import cProfile
import os
import pstats
import sys
import time
from datetime import datetime
from pathlib import Path
import argparse

from sage.all import GF, QQ, ZZ, Integer, save

from uconf import compute_chain_complex, euclidean_unordered_configuration_model
from uconf.algebraic.torus_configuration import surjection_torus_configuration_model


def parse_field(s: str):
    """Parse a --field argument into (Sage ring, filename token).

    Accepts 'Q'/'QQ' for the rationals, 'Z'/'ZZ' for the integers, or a prime-power integer for a finite field.
    """
    if s.strip().lower() in ("q", "qq"):
        return QQ, "Q"
    if s.strip().lower() in ("z", "zz"):
        return ZZ, "Z"
    try:
        n = Integer(int(s))
    except ValueError as e:
        raise ValueError(f"--field must be 'Q', 'Z', or a prime-power integer, got {s!r}") from e
    if n < 2 or not n.is_prime_power():
        raise ValueError(f"--field={n} is not a prime power; GF(n) only exists for prime powers")
    return GF(n), f"F{n}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the chain complex of the unordered configuration model."
    )

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--sphere-dim", "-S", type=int, help="The dimension of the sphere.")
    model_group.add_argument(
        "--torus",
        "-T",
        action="store_true",
        help="Use the torus configuration model instead of the sphere.",
    )

    parser.add_argument(
        "--weight", "-w", type=int, default=2, help="The weight of the configuration."
    )
    parser.add_argument("--deg_max", "-m", type=int, default=3, help="The maximum degree.")
    parser.add_argument(
        "--field",
        "-f",
        default="2",
        help="Base field: prime p for GF(p), or 'Q' for QQ (default: 2).",
    )
    parser.add_argument(
        "--jobs", "-j", type=int, default=1, help="The number of parallel jobs to use."
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print timestamped phase diagnostics to stderr.",
    )
    parser.add_argument(
        "--no-prewarm", action="store_true", help="Disable cache prewarm before forking workers."
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable cProfile (faster runs when profiling data is not needed).",
    )
    args = parser.parse_args()

    w = args.weight
    degs = range(-1, args.deg_max + 1)
    n_jobs = args.jobs
    sphere_dim = args.sphere_dim
    use_torus = args.torus
    verbose = args.verbose
    do_prewarm = not args.no_prewarm
    do_profile = args.profile
    base_ring, field_token = parse_field(args.field)

    model_name = "T2" if use_torus else f"S{sphere_dim}"
    print(
        f"{model_name}, weight={w}, degs={list(degs)}, n_jobs={n_jobs}, field={field_token}",
        flush=True,
    )

    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] building model...", file=sys.stderr, flush=True)
    if use_torus:
        model = surjection_torus_configuration_model(base_ring)
    else:
        model = euclidean_unordered_configuration_model(base_ring, sphere_dim)
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

    path_prefix_short = f"{field_token}_{model_name}_w{w}_m{args.deg_max}"
    path_prefix = f"{path_prefix_short}_j{n_jobs}"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = Path(f"dump/{path_prefix_short}_cc.csv")
    cc_path = Path(f"dump/{path_prefix_short}_cc.sobj")

    if do_profile:
        assert profile is not None
        assert prewarm_profile is not None
        assert worker_profile_paths is not None

        report_path = Path(f"dump/{path_prefix}_benchmark_profile_{timestamp}.txt")
        prewarm_report_path = Path(f"dump/{path_prefix}_benchmark_prewarm_profile_{timestamp}.txt")

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
            print(
                "Prewarm profiler inactive (n_jobs=1 or prewarm disabled) — no prewarm profile written"
            )
    else:
        print(f"Elapsed time: {elapsed:.4f}s")

    cc.save(cc_path)
    print(f"Chain complex saved to {cc_path}")

    bases_path = Path(f"dump/{path_prefix_short}_bases.sobj")
    bases = {d: [x.support()[0] for x in mod.graded_basis_by_weight(d, w)] for d in degs}
    save(bases, bases_path)
    print(f"Graded bases saved to {bases_path}")

    print("Computing Betti numbers...", flush=True)
    with csv_path.open("w") as f:
        print("d,dim,betti", file=f)
        for d in degs:
            print(f"-> Computing Betti numbers for degree {d}...", flush=True)
            rank, betti = cc.free_module_rank(d), cc.betti(d)
            print(f"-> Degree {d}: dim={rank}, betti={betti}", flush=True)
            print(f"{d},{rank},{betti}", file=f)
        print(f"Betti numbers saved to {csv_path}")
