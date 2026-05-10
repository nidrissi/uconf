#!python

import cProfile
import pstats
import sys
import time
from datetime import datetime
from pathlib import Path
import argparse

from sage.all import GF, load, save

from uconf import compute_homology_representatives, euclidean_unordered_configuration_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compute homology representatives from a saved chain complex dump. "
            "The dump must have been produced by benchmark.py (or an equivalent script) "
            "with the same --dim, --weight, and --deg_max parameters."
        )
    )
    parser.add_argument(
        "dump",
        type=str,
        help=(
            "Path to the saved chain complex (the .sobj file produced by benchmark.py, "
            "with or without the .sobj extension)."
        ),
    )
    parser.add_argument("--dim", "-d", type=int, default=2, help="The dimension of the sphere.")
    parser.add_argument(
        "--weight", "-w", type=int, default=2, help="The weight of the configuration."
    )
    parser.add_argument("--deg_max", "-m", type=int, default=3, help="The maximum degree.")
    parser.add_argument(
        "--deg_min",
        type=int,
        default=-1,
        help="The minimum degree (default: -1, matching benchmark.py).",
    )
    parser.add_argument(
        "--algorithm",
        "-a",
        choices=["fast", "sage"],
        default="fast",
        help="Algorithm to use for computing representatives (default: fast).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print timestamped phase diagnostics to stderr.",
    )
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="Disable cProfile (faster runs when profiling data is not needed).",
    )
    args = parser.parse_args()

    w = args.weight
    degs = range(args.deg_min, args.deg_max + 1)
    dim = args.dim
    verbose = args.verbose
    do_profile = not args.no_profile
    algorithm = args.algorithm

    # Normalise the dump path: strip .sobj if present so we can use it as a
    # stem both for loading and for output file naming.
    dump_path = args.dump
    if dump_path.endswith(".sobj"):
        dump_stem = dump_path[: -len(".sobj")]
    else:
        dump_stem = dump_path
    dump_sobj = dump_stem + ".sobj"

    print(
        f"dim={dim}, weight={w}, degs={list(degs)}, algorithm={algorithm}",
        flush=True,
    )

    # --- Load chain complex ---
    if verbose:
        print(
            f"[{time.strftime('%H:%M:%S')}] loading chain complex from {dump_sobj}...",
            file=sys.stderr,
            flush=True,
        )
    start_load = time.perf_counter()
    cc = load(dump_sobj)
    elapsed_load = time.perf_counter() - start_load
    if verbose:
        print(
            f"[{time.strftime('%H:%M:%S')}] chain complex loaded in {elapsed_load:.3f}s",
            file=sys.stderr,
            flush=True,
        )

    # --- Build the module ---
    if verbose:
        print(
            f"[{time.strftime('%H:%M:%S')}] building model (dim={dim})...",
            file=sys.stderr,
            flush=True,
        )
    start_model = time.perf_counter()
    model = euclidean_unordered_configuration_model(GF(2), dim)
    mod = model.module
    elapsed_model = time.perf_counter() - start_model
    if verbose:
        print(
            f"[{time.strftime('%H:%M:%S')}] model ready in {elapsed_model:.3f}s",
            file=sys.stderr,
            flush=True,
        )

    # --- Compute representatives ---
    profile: cProfile.Profile | None = cProfile.Profile() if do_profile else None

    representatives: dict[int, list] = {}

    start = time.perf_counter()
    if profile is not None:
        profile.enable()

    for d in degs:
        if verbose:
            print(
                f"[{time.strftime('%H:%M:%S')}] computing representatives for degree {d}...",
                file=sys.stderr,
                flush=True,
            )
        t0 = time.perf_counter()
        reps = compute_homology_representatives(mod, d, w, cc, algorithm=algorithm)
        t1 = time.perf_counter()
        representatives[d] = reps
        if verbose:
            print(
                f"[{time.strftime('%H:%M:%S')}] degree {d}: "
                f"{len(reps)} representative(s) in {t1 - t0:.3f}s",
                file=sys.stderr,
                flush=True,
            )

    if profile is not None:
        profile.disable()
    elapsed = time.perf_counter() - start

    # --- Print summary to stdout ---
    for d in degs:
        reps = representatives[d]
        betti = len(reps)
        print(f"Degree {d}: {betti} representative(s)")
        for i, r in enumerate(reps):
            print(f"  [{i}] {r}")

    # --- Output paths ---
    path_suffix = f"{dim}_{w}_{args.deg_max}"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dump_dir = Path("dump")
    reps_path = dump_dir / f"homology_reps_F2_{path_suffix}"
    txt_path = dump_dir / f"homology_reps_F2_{path_suffix}.txt"

    # --- Save profiling data ---
    if do_profile:
        assert profile is not None
        report_path = dump_dir / f"homology_reps_profile_{timestamp}_{path_suffix}.txt"
        with report_path.open("w") as report_file:
            report_file.write(f"Date: {datetime.now()}\n")
            report_file.write(
                f"Homology representatives for {mod} with weight {w}, algorithm={algorithm}\n"
            )
            report_file.write(f"Degrees: {list(degs)}\n")
            report_file.write(f"Elapsed time (representatives only): {elapsed:.4f}s\n")
            report_file.write(f"Load time: {elapsed_load:.4f}s\n")
            report_file.write(f"Model build time: {elapsed_model:.4f}s\n")
            report_file.write("\n" + "=" * 80 + "\n")
            pstats.Stats(profile, stream=report_file).strip_dirs().sort_stats(
                "cumulative"
            ).print_stats()
        print(f"Profile written to {report_path}")
    else:
        print(f"Elapsed time: {elapsed:.4f}s")

    # --- Save text report ---
    with txt_path.open("w") as f:
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"dim={dim}, weight={w}, degs={list(degs)}, algorithm={algorithm}\n")
        f.write(f"Source dump: {dump_sobj}\n")
        f.write(f"Elapsed time (representatives only): {elapsed:.4f}s\n\n")
        for d in degs:
            reps = representatives[d]
            f.write(f"Degree {d}: {len(reps)} representative(s)\n")
            for i, r in enumerate(reps):
                f.write(f"  [{i}] {r}\n")
    print(f"Text report saved to {txt_path}")

    # --- Save representatives as .sobj ---
    # The saved object is a dict mapping degree -> list of module elements.
    save(representatives, str(reps_path))
    print(f"Representatives saved to {reps_path}.sobj")
