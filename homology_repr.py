#!python

import cProfile
import pstats
import re
import sys
import time
from datetime import datetime
from pathlib import Path
import argparse

from sage.all import GF, QQ, load, save

from uconf import compute_homology_representatives, euclidean_unordered_configuration_model

# Matches benchmark.py's output name pattern for chain complex dumps, e.g.
# "F2_d2_w3_m4_cc.sobj", "F3_d2_w3_m4_cc", or "Q_d2_w3_m4_cc".
_CC_FILENAME_RE = re.compile(r"^(F\d+|Q)_d(\d+)_w(\d+)_m(\d+)_cc$")


def parse_field(s: str):
    """Parse a --field argument into (Sage ring, filename token)."""
    if s.strip().lower() in ("q", "qq"):
        return QQ, "Q"
    p = int(s)
    return GF(p), f"F{p}"


def _guess_params_from_dump(dump_stem: str) -> tuple[str, int, int, int] | None:
    """Try to extract (field_token, dim, weight, deg_max) from a benchmark.py-style dump filename."""
    name = Path(dump_stem).name
    m = _CC_FILENAME_RE.match(name)
    if m is None:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compute homology representatives from a saved chain complex dump. "
            "The dump must have been produced by benchmark.py (or an equivalent script). "
            "Parameters (--dim, --weight, --deg_max) are guessed from the filename when possible."
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
    parser.add_argument(
        "--dim",
        "-d",
        type=int,
        default=None,
        help="The dimension of the sphere (guessed from filename if omitted).",
    )
    parser.add_argument(
        "--weight",
        "-w",
        type=int,
        default=None,
        help="The weight of the configuration (guessed from filename if omitted).",
    )
    parser.add_argument(
        "--deg_max",
        "-m",
        type=int,
        default=None,
        help="The maximum degree (guessed from filename if omitted).",
    )
    parser.add_argument(
        "--deg_min",
        type=int,
        default=-1,
        help="The minimum degree (default: -1, matching benchmark.py).",
    )
    parser.add_argument(
        "--field",
        "-f",
        default=None,
        help=(
            "Base field: prime p for GF(p), or 'Q' for QQ "
            "(guessed from filename if omitted; defaults to '2')."
        ),
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
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt when parameters are guessed from the filename.",
    )
    args = parser.parse_args()

    # Normalise the dump path: strip .sobj if present so we can use it as a
    # stem both for loading and for output file naming.
    dump_path = args.dump
    if dump_path.endswith(".sobj"):
        dump_stem = dump_path[: -len(".sobj")]
    else:
        dump_stem = dump_path
    dump_sobj = dump_stem + ".sobj"

    # --- Parameter guessing from filename ---
    guessed = _guess_params_from_dump(dump_stem)
    if guessed is not None:
        guessed_field, guessed_dim, guessed_w, guessed_deg_max = guessed
    else:
        guessed_field, guessed_dim, guessed_w, guessed_deg_max = None, None, None, None

    _DEFAULTS = {"dim": 2, "weight": 2, "deg_max": 3, "field": "2"}

    def _resolve_param(name, user_val, guessed_val):
        """Return the effective parameter value, emitting warnings/info as appropriate."""
        if user_val is not None and guessed_val is not None and user_val != guessed_val:
            print(
                f"Warning: --{name}={user_val} was given but the filename suggests {name}={guessed_val}.",
                flush=True,
            )
        if user_val is not None:
            return user_val
        if guessed_val is not None:
            return guessed_val
        return _DEFAULTS[name]

    dim_resolved = _resolve_param("dim", args.dim, guessed_dim)
    w_resolved = _resolve_param("weight", args.weight, guessed_w)
    deg_max_resolved = _resolve_param("deg_max", args.deg_max, guessed_deg_max)
    # For the field, compare on the canonical token ("F2", "F3", "Q") rather than
    # the raw user string (so e.g. --field q matches a guessed "Q").
    user_field_token = parse_field(args.field)[1] if args.field is not None else None
    field_token = _resolve_param("field", user_field_token, guessed_field)
    base_ring, field_token = parse_field(field_token)

    # If any parameter was guessed (i.e. not provided by the user), ask for confirmation.
    guessed_params = {}
    if args.dim is None and guessed_dim is not None:
        guessed_params["dim"] = dim_resolved
    if args.weight is None and guessed_w is not None:
        guessed_params["weight"] = w_resolved
    if args.deg_max is None and guessed_deg_max is not None:
        guessed_params["deg_max"] = deg_max_resolved
    if args.field is None and guessed_field is not None:
        guessed_params["field"] = field_token

    if guessed_params and not args.yes:
        guessed_str = ", ".join(f"{k}={v}" for k, v in guessed_params.items())
        print(f"Guessed from filename: {guessed_str}")
        answer = input("Proceed with these parameters? [Y/n] ").strip().lower()
        if answer not in ("", "y", "yes"):
            print("Aborted.")
            sys.exit(0)

    dim = dim_resolved
    w = w_resolved
    degs = range(args.deg_min, deg_max_resolved + 1)
    verbose = args.verbose
    do_profile = not args.no_profile
    algorithm = args.algorithm

    print(
        f"dim={dim}, weight={w}, degs={list(degs)}, algorithm={algorithm}, field={field_token}",
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
    model = euclidean_unordered_configuration_model(base_ring, dim)
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
    path_prefix = f"{field_token}_d{dim}_w{w}_m{deg_max_resolved}"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dump_dir = Path("dump")
    dump_dir.mkdir(parents=True, exist_ok=True)
    reps_path = dump_dir / f"{path_prefix}_homology_reps.sobj"
    txt_path = dump_dir / f"{path_prefix}_homology_reps.txt"

    # --- Save profiling data ---
    if do_profile:
        assert profile is not None
        report_path = dump_dir / f"{path_prefix}_homology_reps_profile.txt"
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

    # --- Save representatives as a pickle file ---
    # BarAlgebraModule elements contain references to local closures that
    # Python's pickle cannot serialise directly.  We save the monomial
    # coefficients (a plain dict {basis_key: coeff}) for each element instead.
    # To reconstruct element `e` from its monomial coefficients `mc`, use:
    #   e = sum(coeff * mod.monomial(key) for key, coeff in mc.items(), mod.zero())
    serializable = {
        d: [r.monomial_coefficients() for r in reps] for d, reps in representatives.items()
    }
    save(serializable, reps_path)
    print(f"Representatives saved to {reps_path}")
