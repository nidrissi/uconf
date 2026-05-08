"""Chain-complex construction and homology helpers for dg-modules.

Given any module exposing ``graded_basis(d)`` and ``boundary`` (as used by
operad/cooperad components, bar/cobar constructions, free algebras, cofree
coalgebras, etc.), this module builds a SageMath
:class:`~sage.homology.chain_complex.ChainComplex` and provides helpers to
extract homology representatives as native module elements.

EXAMPLES::

    sage: from sage.all import QQ
    sage: from uconf import Surjection
    sage: from uconf.homology import compute_chain_complex, homology_basis
    sage: S2 = Surjection(2, QQ)
    sage: C = compute_chain_complex(S2, degrees=range(4))
    sage: 0 in C.betti()
    True
    sage: homology_basis(S2, degree=0)
    [{2 1}]
"""

from __future__ import annotations

import cProfile
import inspect
import os
import sys
import tempfile
import time
from typing import Any, Iterator, Literal, Sequence, TextIO

import gc
import multiprocessing
import multiprocessing.connection
import signal
from sage.all import ChainComplex, matrix

from uconf.core.signs import get_on_basis


def _vprint(msg: str, verbose: bool, *, stream: TextIO | None = None, end: str = "\n") -> None:
    """Print a timestamped diagnostic message when verbose is True."""
    if not verbose:
        return
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=stream or sys.stderr, flush=True, end=end)


# Keep multiprocessing helpers at module scope so forked workers can reuse the
# same top-level functions and per-call inherited state without re-serializing
# the model.
_BOUNDARY_MATRIX_STATE: dict[str, Any] = {}
_BOUNDARY_MATRIX_CHUNKS_PER_WORKER = 4


class _ChainComplexProgressReporter:
    """Low-overhead progress reporting for chain-complex assembly."""

    def __init__(
        self,
        total_columns: int,
        *,
        enabled: bool,
        stream: TextIO | None = None,
        min_interval: float = 0.2,
    ):
        self._total_columns = total_columns
        self._enabled = enabled and total_columns > 0
        self._stream = stream if stream is not None else sys.stderr
        self._min_interval = min_interval
        self._completed = 0
        self._last_emit = 0.0
        self._interactive = bool(
            hasattr(self._stream, "isatty")
            and callable(self._stream.isatty)
            and self._stream.isatty()
        )
        self._last_width = 0

    def update(self, n_columns: int, *, degree: int | None) -> None:
        """Advance the progress display by ``n_columns`` columns."""
        if not self._enabled or n_columns <= 0:
            return
        self._completed += n_columns
        now = time.monotonic()
        if self._completed < self._total_columns and now - self._last_emit < self._min_interval:
            return
        self._last_emit = now
        self._emit(degree)

    def close(self) -> None:
        """Finish the progress display."""
        if not self._enabled:
            return
        if self._completed < self._total_columns:
            self._completed = self._total_columns
            self._emit(None)
        if self._interactive:
            print(file=self._stream, flush=True)

    def _emit(self, degree: int | None) -> None:
        pct = int((100 * self._completed) / self._total_columns)
        degree_msg = f", degree {degree}" if degree is not None else ""
        message = (
            "compute_chain_complex: "
            f"{self._completed}/{self._total_columns} columns ({pct}%){degree_msg}"
        )
        if self._interactive:
            width = max(self._last_width, len(message))
            self._last_width = width
            _vprint(f"{message.ljust(width)}\r", True, stream=self._stream, end="")
            return
        print(message, file=self._stream, flush=True)


# ---------------------------------------------------------------------------
# Boundary matrix
# ---------------------------------------------------------------------------


def _build_matrix_from_entries(
    base_ring: Any,
    n_target: int,
    n_source: int,
    entries: dict[tuple[int, int], Any],
    *,
    sparse: bool,
) -> Any:
    """Build a Sage matrix from sparse ``(row, col) -> coeff`` entries."""
    if not entries:
        return matrix(base_ring, n_target, n_source, sparse=sparse)
    if sparse:
        return matrix(base_ring, n_target, n_source, entries, sparse=True)
    M = matrix(base_ring, n_target, n_source, sparse=False)
    for (i, j), coeff in entries.items():
        M[i, j] = coeff
    return M


def _boundary_entries_for_key(
    key: Any,
    j: int,
    *,
    boundary_terms: Any,
    key_to_idx_target: dict,
    normalize_key: Any,
    normalization_cache: dict | None = None,
    profile: dict[str, float | int] | None = None,
) -> dict[tuple[int, int], Any]:
    """Return the sparse matrix entries contributed by one source key."""
    entries: dict[tuple[int, int], Any] = {}
    for bdry_key, coeff in boundary_terms:
        i = key_to_idx_target.get(bdry_key)
        if i is not None:
            ij = (i, j)
            if ij in entries:
                entries[ij] += coeff
            else:
                entries[ij] = coeff
            continue
        if normalize_key is not None:
            if profile is not None:
                profile["normalization_lookups"] = profile.get("normalization_lookups", 0) + 1
                start = time.perf_counter()
            cached_matches = (
                normalization_cache.get(bdry_key) if normalization_cache is not None else None
            )
            if cached_matches is None:
                matches = []
                for norm_coeff, norm_key in normalize_key(bdry_key):
                    i = key_to_idx_target.get(norm_key)
                    if i is not None:
                        matches.append((i, norm_coeff))
                cached_matches = tuple(matches)
                if normalization_cache is not None:
                    normalization_cache[bdry_key] = cached_matches
            if profile is not None:
                profile["normalization_seconds"] = profile.get("normalization_seconds", 0.0) + (
                    time.perf_counter() - start
                )
                if cached_matches:
                    profile["normalized_matches"] = profile.get("normalized_matches", 0) + len(
                        cached_matches
                    )
            if cached_matches:
                for i, norm_coeff in cached_matches:
                    ij = (i, j)
                    combined = coeff * norm_coeff
                    if ij in entries:
                        entries[ij] += combined
                    else:
                        entries[ij] = combined
                continue
        raise ValueError(
            f"Boundary of basis key {key!r} contains key {bdry_key!r} "
            "not found in target basis keys"
        )
    return entries


def _boundary_matrix_worker(task: tuple) -> dict[tuple[int, int], Any]:
    """Compute sparse boundary entries for a chunk of source columns.

    ``task`` is ``(chunk_source_keys, j_start, key_to_idx_target)`` where
    per-degree data is passed inline so that a persistent worker pool (forked
    once) can handle tasks across multiple degrees without re-forking.  Static
    data (``boundary_on_basis``, ``normalize_key``, profiling flags) is read
    from the module-global ``_BOUNDARY_MATRIX_STATE``, which is set in the
    parent before pool creation and inherited via copy-on-write.
    """
    chunk_source_keys, j_start, key_to_idx_target = task
    state = _BOUNDARY_MATRIX_STATE
    boundary_on_basis = state["boundary_on_basis"]
    normalize_key = state["normalize_key"]
    worker_profile = state["worker_profile"]
    parent_profiler = state["parent_profiler"]
    collect_boundary_profile = state.get("collect_boundary_profile", False)
    normalization_cache: dict = {}
    profile_path = None
    profiler = None
    boundary_profile = None
    if collect_boundary_profile:
        boundary_profile = {
            "boundary_on_basis_calls": 0,
            "boundary_on_basis_seconds": 0.0,
            "normalization_lookups": 0,
            "normalization_seconds": 0.0,
            "normalized_matches": 0,
            "entry_merge_seconds": 0.0,
        }
    if worker_profile:
        if parent_profiler is not None:
            parent_profiler.disable()
        profiler = cProfile.Profile()
        profiler.enable()

    try:
        entries: dict[tuple[int, int], Any] = {}
        for idx, key in enumerate(chunk_source_keys):
            j = j_start + idx
            boundary_start = time.perf_counter() if boundary_profile is not None else None
            boundary_terms = boundary_on_basis(key)
            if boundary_profile is not None:
                boundary_profile["boundary_on_basis_calls"] += 1
                boundary_profile["boundary_on_basis_seconds"] += (
                    time.perf_counter() - boundary_start
                )
            column_entries = _boundary_entries_for_key(
                key,
                j,
                boundary_terms=boundary_terms,
                key_to_idx_target=key_to_idx_target,
                normalize_key=normalize_key,
                normalization_cache=normalization_cache,
                profile=boundary_profile,
            )
            merge_start = time.perf_counter() if boundary_profile is not None else None
            for ij, coeff in column_entries.items():
                if ij in entries:
                    entries[ij] += coeff
                else:
                    entries[ij] = coeff
            if boundary_profile is not None:
                boundary_profile["entry_merge_seconds"] += time.perf_counter() - merge_start
    finally:
        if profiler is not None:
            profiler.disable()
            fd, profile_path = tempfile.mkstemp(
                prefix="uconf-chain-complex-worker-",
                suffix=".prof",
            )
            os.close(fd)
            profiler.dump_stats(profile_path)

    return {
        "entries": entries,
        "profile_path": profile_path,
        "columns_done": len(chunk_source_keys),
        "boundary_profile": boundary_profile,
    }


def _chunk_ranges(n_items: int, chunk_size: int) -> list[tuple[int, int]]:
    """Return contiguous chunk ranges covering ``range(n_items)``."""
    return [(start, min(start + chunk_size, n_items)) for start in range(0, n_items, chunk_size)]


def _merge_sparse_entries(
    entries: dict[tuple[int, int], Any],
    new_entries: dict[tuple[int, int], Any],
) -> None:
    """Accumulate sparse entries in-place."""
    for ij, coeff in new_entries.items():
        if ij in entries:
            entries[ij] += coeff
        else:
            entries[ij] = coeff


def _merge_boundary_matrix_profile(
    profile: dict[str, float | int],
    update: dict[str, float | int] | None,
) -> None:
    """Accumulate boundary-matrix profiling counters in-place."""
    if update is None:
        return
    for key, value in update.items():
        profile[key] = profile.get(key, 0) + value


def _prewarm_parallel_boundary_caches(
    module: Any,
    basis_source_keys: Sequence,
    *,
    profile: dict[str, float | int] | None = None,
    verbose: bool = False,
    profiler: cProfile.Profile | None = None,
    parent_profiler: cProfile.Profile | None = None,
) -> None:
    """Populate module-level caches in the parent before forked workers start."""
    prewarm = getattr(module, "_prewarm_parallel_boundary_caches", None)
    if not callable(prewarm):
        return
    start = time.perf_counter()
    _vprint(f"prewarm: starting for {len(basis_source_keys)} source keys", verbose)
    prewarm_source_keys = (
        basis_source_keys if isinstance(basis_source_keys, tuple) else tuple(basis_source_keys)
    )
    try:
        sig = inspect.signature(prewarm)
        accepts_verbose = "verbose" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
    except (ValueError, TypeError):
        accepts_verbose = False
    if profiler is not None:
        if parent_profiler is not None:
            parent_profiler.disable()
        profiler.enable()
    try:
        if accepts_verbose:
            prewarm(prewarm_source_keys, verbose=verbose)
        else:
            prewarm(prewarm_source_keys)
    finally:
        if profiler is not None:
            profiler.disable()
            if parent_profiler is not None:
                parent_profiler.enable()
    elapsed = time.perf_counter() - start
    _vprint(f"prewarm: done ({elapsed:.1f}s)", verbose)
    if profile is not None:
        profile["parallel_cache_prewarm_calls"] = profile.get("parallel_cache_prewarm_calls", 0) + 1
        profile["parallel_cache_prewarm_keys"] = profile.get(
            "parallel_cache_prewarm_keys", 0
        ) + len(basis_source_keys)
        profile["parallel_cache_prewarm_seconds"] = (
            profile.get("parallel_cache_prewarm_seconds", 0.0) + elapsed
        )


# ---------------------------------------------------------------------------
# Raw-fork worker orchestration
# ---------------------------------------------------------------------------
#
# WHY NOT multiprocessing.Pool?
# ==============================
# Pool teardown is slow and can hang indefinitely when workers hold large
# caches (multiple GB), because graceful Python shutdown triggers GC,
# finalizers, and destructor chains.  Pool.terminate() can also hang when the
# Pool's internal feeder thread is blocked on a full queue.
#
# The intended model here is "crash-only" disposable workers:
#   compute → emit result → idle forever → SIGKILL by parent
#
# WHY gc.freeze() BEFORE FORKING?
# =================================
# gc.freeze() moves all currently tracked Python objects into the permanent
# (gen 2) GC generation.  The CPython GC does not scan permanent-generation
# objects, so their internal reference counts are never touched by GC traversal.
# This avoids copy-on-write (CoW) faults on pages that contain the GC headers
# of the warmed cache objects, which would otherwise force the OS to copy pages
# into the worker's private address space just because the GC incremented a
# refcount, negating most of the RAM savings from fork-CoW sharing.
#
# WHY SIGKILL (not SIGTERM or graceful close)?
# =============================================
# Workers hold multi-GB dictionaries.  Letting the Python interpreter run its
# shutdown sequence (atexit handlers, __del__ finalizers, GC collection) would
# take minutes and sometimes hang.  SIGKILL is immediate: the OS reclaims all
# memory instantly without running any Python cleanup code.  The join() call
# that follows is purely for zombie reaping, not for graceful exit.
#
# LINUX-ONLY ASSUMPTION
# ======================
# This code assumes Linux fork(2) copy-on-write semantics.  The "fork" start
# method is selected explicitly.  spawn / forkserver are not used because they
# cannot inherit the pre-warmed parent caches.
#
# COW INVARIANT
# =============
# After gc.freeze() and before workers are forked, shared global structures
# MUST remain READ-ONLY.  The following patterns are DANGEROUS and must be
# avoided in worker code:
#   - Inserting into or deleting from module-level dicts (e.g. _BOUNDARY_MATRIX_STATE)
#   - Appending to shared lists
#   - Triggering lazy cache population (e.g. calling @cached_method on a shared object)
#   - Touching any global mutable state
# Any write to a CoW-shared page causes Linux to copy the entire 4 KB page into
# the worker's private address space, defeating the memory savings.


def _worker_loop(
    task_conn: "multiprocessing.connection.Connection",
    result_conn: "multiprocessing.connection.Connection",
) -> None:
    """Worker main loop: receive tasks, compute results, send back, repeat.

    This function runs in a forked child process.  It intentionally does NOT
    call ``sys.exit()``, run cleanup handlers, or perform graceful teardown.
    After all tasks are processed (or on pipe EOF), the worker blocks forever
    and waits for SIGKILL from the parent.

    IMPORTANT — CoW safety:
        This function reads from ``_BOUNDARY_MATRIX_STATE`` via
        ``_boundary_matrix_worker`` but must NEVER write to it or any other
        module-level shared structure.  Any write would cause Linux to copy the
        underlying memory page, negating the memory savings from fork CoW.

    Parameters
    ----------
    task_conn:
        Read-only end of the task pipe (parent writes tasks here).
    result_conn:
        Write-only end of the result pipe (worker writes results here).
    """
    # Workers must not attempt graceful cleanup — intentional design.
    # Do NOT add sys.exit(), atexit handlers, or resource finalizers here.
    #
    # Ctrl-C should be handled by the parent process only.  If workers inherit
    # SIGINT and are blocked in recv()/sleep(), they emit noisy tracebacks
    # instead of letting the parent tear the whole pool down cleanly.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        while True:
            try:
                task = task_conn.recv()
            except EOFError:
                # Parent closed the write end of the pipe (or was SIGKILLed).
                # Block forever and wait for SIGKILL from the parent.  Do NOT
                # call sys.exit() — that would trigger Python interpreter
                # teardown, which is exactly what we want to avoid.
                while True:
                    time.sleep(3600)
            result = _boundary_matrix_worker(task)
            result_conn.send(result)
    except Exception as exc:  # noqa: BLE001
        # Unexpected error: send an error result so the parent can detect it,
        # then block forever until SIGKILL.  Do NOT call sys.exit().
        try:
            result_conn.send(
                {
                    "error": str(exc),
                    "entries": {},
                    "profile_path": None,
                    "columns_done": 0,
                    "boundary_profile": None,
                }
            )
        except Exception:  # noqa: BLE001
            pass
        while True:
            time.sleep(3600)


class _WorkerManager:
    """Manages raw forked worker processes for parallel boundary-matrix assembly.

    Replaces ``multiprocessing.Pool`` with explicit ``Process`` lifecycle
    management to avoid Pool's slow graceful teardown and potential hangs.

    See the module-level comments above for the full rationale.

    **LINUX-ONLY**: Relies on fork(2) copy-on-write semantics.

    Worker lifecycle
    ----------------
    1. Workers are forked once at construction time.
    2. Tasks are submitted via ``map_unordered`` (round-robin across workers).
    3. Results are yielded in completion order as they arrive.
    4. After all results are collected, call ``kill_all()`` to SIGKILL every
       worker immediately.  ``join(timeout=...)`` is used only for zombie
       reaping — the parent never blocks indefinitely.

    CoW invariant
    -------------
    Workers inherit the parent's address space via fork copy-on-write.  Shared
    global structures (e.g. ``_BOUNDARY_MATRIX_STATE``, module caches) must
    remain READ-ONLY after construction.  Any write causes Linux to copy the
    underlying page, negating the RAM savings.
    """

    # Maximum time (seconds) to wait for a worker to be reaped after SIGKILL.
    # SIGKILL is immediate; the join timeout is purely for zombie reaping.
    _REAP_TIMEOUT: float = 5.0

    def __init__(self, n_jobs: int) -> None:
        # Use the "fork" context explicitly.  spawn/forkserver are not used
        # because they cannot inherit the pre-warmed parent caches.
        ctx = multiprocessing.get_context("fork")
        self._workers: list[multiprocessing.Process] = []
        # One unidirectional Pipe pair per worker:
        #   task:   parent → worker  (parent sends tasks)
        #   result: worker → parent  (worker sends results)
        self._task_conns: list[multiprocessing.connection.Connection] = []
        self._result_conns: list[multiprocessing.connection.Connection] = []
        self._n_jobs = n_jobs
        self._alive = True

        for _ in range(n_jobs):
            # Pipe(duplex=False) returns (readable_end, writable_end).
            # task pipe:   parent writes tasks  → worker reads them
            # result pipe: worker writes results → parent reads them
            child_task_conn, parent_task_conn = ctx.Pipe(duplex=False)
            parent_result_conn, child_result_conn = ctx.Pipe(duplex=False)
            p = ctx.Process(
                target=_worker_loop,
                args=(child_task_conn, child_result_conn),
                # daemon=True: if the parent exits unexpectedly (e.g. crash),
                # the OS will send SIGKILL to daemon children automatically.
                daemon=True,
            )
            p.start()
            # Close child-side pipe ends in the parent to ensure proper EOF
            # detection when workers die and to avoid resource leaks.
            child_task_conn.close()
            child_result_conn.close()

            self._workers.append(p)
            self._task_conns.append(parent_task_conn)
            self._result_conns.append(parent_result_conn)

    def map_unordered(self, tasks: list[Any]) -> Iterator[Any]:
        """Distribute *tasks* across workers and yield results as they arrive.

        Tasks are distributed in a bounded round-robin schedule.  At most one
        task per worker is in flight at a time, which prevents the parent from
        deadlocking while trying to push many large task descriptors into a
        worker's pipe before it has finished its current task.  Results are
        yielded in completion order (not submission order) for maximum
        throughput.

        If a worker sends an error result or dies unexpectedly, this method
        kills all workers, reaps zombies, and raises ``RuntimeError``.

        Parameters
        ----------
        tasks:
            List of task descriptors to distribute.

        Yields
        ------
        Result dicts returned by ``_boundary_matrix_worker``.
        """
        if not tasks:
            return

        conn_to_idx = {id(conn): i for i, conn in enumerate(self._result_conns)}
        active_conns: list[multiprocessing.connection.Connection] = []
        next_task_idx = 0
        completed = 0

        def _send_task(worker_idx: int, task: Any) -> None:
            try:
                self._task_conns[worker_idx].send(task)
            except (BrokenPipeError, EOFError, OSError) as exc:
                self.kill_all()
                raise RuntimeError(
                    f"Worker {worker_idx} (pid {self._workers[worker_idx].pid}) "
                    "died while receiving a task"
                ) from exc

        # Seed each worker with at most one task.  Task descriptors can be
        # large because they currently carry both chunk basis keys and the
        # target index map, so unbounded eager submission can fill the pipe and
        # deadlock the parent while workers are still busy with earlier tasks.
        for worker_idx in range(min(self._n_jobs, len(tasks))):
            _send_task(worker_idx, tasks[next_task_idx])
            next_task_idx += 1
            active_conns.append(self._result_conns[worker_idx])

        while completed < len(tasks):
            # Wait for any result connection to become readable (60 s timeout
            # to avoid hanging forever if a worker stalls unexpectedly).
            ready = multiprocessing.connection.wait(active_conns, timeout=60.0)
            if not ready:
                # Timeout: check whether any worker has died.
                dead = [p for p in self._workers if not p.is_alive()]
                if dead:
                    self.kill_all()
                    raise RuntimeError(
                        f"{len(dead)} worker process(es) died unexpectedly "
                        "during boundary matrix assembly"
                    )
                continue

            for conn in ready:
                try:
                    result = conn.recv()
                except EOFError:
                    # Worker pipe closed before sending a result — worker died.
                    w_idx = conn_to_idx[id(conn)]
                    self.kill_all()
                    raise RuntimeError(
                        f"Worker {w_idx} (pid {self._workers[w_idx].pid}) "
                        "died unexpectedly (EOF on result pipe)"
                    )
                if result.get("error") is not None:
                    self.kill_all()
                    raise RuntimeError(f"Worker raised an exception: {result['error']}")
                completed += 1
                w_idx = conn_to_idx[id(conn)]
                if next_task_idx < len(tasks):
                    _send_task(w_idx, tasks[next_task_idx])
                    next_task_idx += 1
                else:
                    active_conns = [c for c in active_conns if c is not conn]
                yield result

    def kill_all(self, *, reap_timeout: float = _REAP_TIMEOUT) -> None:
        """SIGKILL all workers immediately, then reap zombies.

        Why SIGKILL and not SIGTERM or graceful close?
            Workers hold multi-GB caches.  Graceful Python interpreter teardown
            triggers GC, finalizers, and destructor chains that take minutes
            and sometimes hang.  SIGKILL is instant — the OS reclaims all
            memory without running any Python cleanup code.

        The subsequent ``join(timeout=...)`` is purely for zombie reaping — it
        does not wait for graceful exit.  If a worker somehow survives SIGKILL
        (which should not happen on Linux), we move on rather than hanging.

        Parameters
        ----------
        reap_timeout:
            Total time budget (seconds) for zombie reaping across all workers.
            This is a safety upper bound; SIGKILL should make reaping
            essentially instant.
        """
        if not self._alive:
            return
        self._alive = False

        # SIGKILL every live worker immediately.
        for p in self._workers:
            if p.is_alive():
                try:
                    os.kill(p.pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass  # Already dead.

        # Reap zombies.  Divide the timeout budget equally across workers.
        per_worker_timeout = reap_timeout / max(1, len(self._workers))
        for p in self._workers:
            p.join(timeout=per_worker_timeout)
            # If still alive after SIGKILL + timeout, something is very wrong,
            # but we still do NOT block forever — just move on.

        # Close parent-side pipe ends.
        for conn in self._task_conns + self._result_conns:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self) -> "_WorkerManager":
        return self

    def __exit__(self, *args: Any) -> None:
        self.kill_all()


def _boundary_matrix(
    module: Any,
    basis_source_keys: list,
    key_to_idx_target: dict,
    n_target: int,
    *,
    sparse: bool,
    n_jobs: int = 1,
    basis_source: list | None = None,
    progress: _ChainComplexProgressReporter | None = None,
    degree: int | None = None,
    worker_profile_paths: list[str] | None = None,
    worker_profile_parent: cProfile.Profile | None = None,
    profile: dict[str, float | int] | None = None,
    pool: Any | None = None,
) -> Any:
    """Compute the boundary matrix from ``basis_source_keys`` to target basis.

    Constructs the matrix representation of the boundary map for a single
    degree in a chain complex. Each column corresponds to a source basis
    element (indexed by ``basis_source_keys``) and each row corresponds to a
    target basis element (indexed by ``key_to_idx_target``).

    Parameters
    ----------
    module:
        The chain complex module; must expose ``base_ring()`` and
        ``boundary()``.
    basis_source_keys:
        Keys of the source basis elements (domain of the boundary map).
    key_to_idx_target:
        Mapping from target basis key to its row index in the matrix.
    n_target:
        Number of target basis elements (number of rows in the matrix).
    sparse:
        If ``True``, return a sparse matrix; otherwise return a dense matrix.
    n_jobs:
        Number of parallel worker processes to use. Values greater than 1
        require either a pre-forked ``pool`` or a POSIX system with fork
        support.
    basis_source:
        Pre-computed source basis elements corresponding to
        ``basis_source_keys``. If ``None``, elements are fetched from the
        module during computation.
    progress:
        Optional progress reporter for tracking column-by-column progress.
    degree:
        The homological degree of the boundary map being computed. Used for
        progress reporting and profiling.
    worker_profile_paths:
        Optional collection used to record generated ``cProfile`` output
        paths from parallel work. Profiling data is written per processed
        chunk/task, not via a fixed one-path-per-worker slot, so a single
        worker may emit multiple ``.prof`` files and the total number of
        files depends on task partitioning. Callers should therefore expect
        zero or more generated profile files, and may need to merge and
        remove multiple files after analysis.
    worker_profile_parent:
        A ``cProfile.Profile`` instance from the calling process. Paused
        during parallel sections and resumed afterwards.
    profile:
        Optional mutable dict accumulating timing and count statistics.
        Updated in-place with keys such as ``boundary_matrix_seconds``.
    pool:
        A pre-forked :class:`_WorkerManager` instance.
        When provided, avoids the overhead of forking new workers per degree.
        Static worker state must already have been set in
        ``_BOUNDARY_MATRIX_STATE`` before this manager was created.

    Returns
    -------
    Any
        A matrix (dense or sparse, depending on ``sparse``) over
        ``module.base_ring()`` representing the boundary map.
    """

    base_ring = module.base_ring()
    n_source = len(basis_source_keys)
    normalize_key = getattr(module, "_normalize_key", None)
    boundary_on_basis = get_on_basis(module.boundary)

    if boundary_on_basis is not None:
        use_parallel = (
            n_jobs > 1
            and n_source > 0
            and (
                pool is not None
                or (os.name == "posix" and "fork" in multiprocessing.get_all_start_methods())
            )
        )
        if use_parallel:
            # Use several chunks per worker so workers can absorb uneven
            # boundary costs without paying per-column scheduling overhead.
            chunk_size = max(
                1,
                (n_source + _BOUNDARY_MATRIX_CHUNKS_PER_WORKER * n_jobs - 1)
                // (_BOUNDARY_MATRIX_CHUNKS_PER_WORKER * n_jobs),
            )
            # Build tasks with per-degree data inline: each task carries its
            # own source-key slice and the target index map so that a
            # persistent worker manager (forked once) can handle multiple
            # degrees.
            tasks = [
                (basis_source_keys[start:stop], start, key_to_idx_target)
                for start, stop in _chunk_ranges(n_source, chunk_size)
            ]
            entries: dict[tuple[int, int], Any] = {}
            if pool is not None:
                # Persistent worker manager: static state was set (and workers
                # forked) once in compute_chain_complex; no prewarm or state
                # setup needed here.
                active_manager: _WorkerManager = pool
                own_manager = False
            else:
                # Per-call temporary worker manager: prewarm caches, set
                # static state, freeze GC, then fork workers.
                _prewarm_parallel_boundary_caches(
                    module,
                    basis_source_keys,
                    profile=profile,
                )
                _BOUNDARY_MATRIX_STATE.clear()
                _BOUNDARY_MATRIX_STATE.update(
                    {
                        "boundary_on_basis": boundary_on_basis,
                        "normalize_key": normalize_key,
                        "worker_profile": worker_profile_paths is not None,
                        "parent_profiler": worker_profile_parent,
                        "collect_boundary_profile": profile is not None,
                    }
                )
                # Freeze GC before forking: moves all tracked objects to the
                # permanent generation so that the GC will not touch their
                # reference counts in workers, avoiding CoW page faults.
                gc.freeze()
                active_manager = _WorkerManager(n_jobs)
                own_manager = True
            try:
                for chunk_result in active_manager.map_unordered(tasks):
                    _merge_sparse_entries(entries, chunk_result["entries"])
                    if profile is not None:
                        _merge_boundary_matrix_profile(
                            profile,
                            chunk_result.get("boundary_profile"),
                        )
                    if (
                        worker_profile_paths is not None
                        and chunk_result["profile_path"] is not None
                    ):
                        worker_profile_paths.append(chunk_result["profile_path"])
                    if progress is not None:
                        progress.update(chunk_result["columns_done"], degree=degree)
            finally:
                if own_manager:
                    # SIGKILL workers immediately — no graceful teardown.
                    active_manager.kill_all()
                    _BOUNDARY_MATRIX_STATE.clear()
            return _build_matrix_from_entries(
                base_ring,
                n_target,
                n_source,
                entries,
                sparse=sparse,
            )

        serial_entries: dict[tuple[int, int], Any] = {}
        normalization_cache: dict = {}
        for j, key in enumerate(basis_source_keys):
            boundary_start = time.perf_counter() if profile is not None else None
            boundary_terms = boundary_on_basis(key)
            if profile is not None:
                profile["boundary_on_basis_calls"] = profile.get("boundary_on_basis_calls", 0) + 1
                profile["boundary_on_basis_seconds"] = profile.get(
                    "boundary_on_basis_seconds", 0.0
                ) + (time.perf_counter() - boundary_start)
            column_entries = _boundary_entries_for_key(
                key,
                j,
                boundary_terms=boundary_terms,
                key_to_idx_target=key_to_idx_target,
                normalize_key=normalize_key,
                normalization_cache=normalization_cache,
                profile=profile,
            )
            merge_start = time.perf_counter() if profile is not None else None
            _merge_sparse_entries(serial_entries, column_entries)
            if profile is not None:
                profile["entry_merge_seconds"] = profile.get("entry_merge_seconds", 0.0) + (
                    time.perf_counter() - merge_start
                )
            if progress is not None:
                progress.update(1, degree=degree)
        return _build_matrix_from_entries(
            base_ring,
            n_target,
            n_source,
            serial_entries,
            sparse=sparse,
        )

    if basis_source is None:
        raise ValueError(
            "basis_source elements are required when module.boundary lacks an on_basis fast path"
        )

    M = matrix(base_ring, n_target, n_source, sparse=sparse)
    for j, elem in enumerate(basis_source):
        for bdry_key, coeff in module.boundary(elem):
            i = key_to_idx_target.get(bdry_key)
            if i is not None:
                M[i, j] += coeff
                continue
            if normalize_key is not None:
                found_match = False
                for norm_coeff, norm_key in normalize_key(bdry_key):
                    i = key_to_idx_target.get(norm_key)
                    if i is not None:
                        M[i, j] += coeff * norm_coeff
                        found_match = True
                if found_match:
                    continue
            raise ValueError(
                f"Boundary of basis element {elem} contains key {bdry_key} "
                "not found in target basis keys"
            )
        if progress is not None:
            progress.update(1, degree=degree)
    return M


def _get_basis_elements(
    module: Any,
    d: int,
    weight: int | None = None,
    *,
    keep_elements: bool,
) -> tuple[list | None, list]:
    """Return ``(basis_elements_or_none, basis_keys)``.

    When *weight* is ``None``, we apply a ``connectivity`` short-circuit:
    if *d* is below the module's reported connectivity, the result is
    trivially empty.  When a *weight* filter is active we skip this check,
    because ``connectivity`` only captures the weight-1 minimum degree —
    higher-weight terms can reach lower degrees (e.g. when the cooperad in
    a cofree coalgebra has negative connectivity).
    """
    if weight is None:
        connectivity = getattr(module, "connectivity", None)
        if isinstance(connectivity, int) and d < connectivity:
            return [], []

    seen: set = set()
    elems: list | None = [] if keep_elements else None
    keys: list = []
    if weight is not None:
        family = module.graded_basis_by_weight(d, weight)
    else:
        family = module.graded_basis(d)
    for b in family:
        if hasattr(b, "leading_support"):
            key = b.leading_support()
        else:
            support = list(b.support())
            if not support:
                continue
            key = support[0]
        if key is None:
            continue
        if key not in seen:
            seen.add(key)
            if elems is not None:
                elems.append(b)
            keys.append(key)
        else:
            raise ValueError(
                f"Duplicate basis key {key} in degree {d} weight {weight} in {type(module).__name__}"
            )
    return elems, keys


def compute_chain_complex(
    module: Any,
    degrees: range,
    *,
    weight: int | None = None,
    check: bool = False,
    sparse: bool = True,
    n_jobs: int = 1,
    progress: bool = False,
    verbose: bool = False,
    prewarm: bool = True,
    prewarm_profiler: cProfile.Profile | None = None,
    worker_profile_paths: list[str] | None = None,
    worker_profile_parent: cProfile.Profile | None = None,
) -> Any:
    """Build a SageMath :class:`ChainComplex` from a dg-module.

    Parameters
    ----------
    module:
        Any object exposing ``graded_basis(d)`` (returning a ``Family`` of
        basis elements), ``boundary`` (a linear map), and ``base_ring()``.
        All operad/cooperad components, bar/cobar constructions, free
        algebras, cofree coalgebras, and similar objects from *uconf*
        satisfy this interface.
    degrees:
        A :class:`range` of integer degrees.  The returned chain complex
        covers *degrees* plus one additional degree above (``max(degrees)+1``)
        so that the differential into ``max(degrees)`` is fully accounted for.
        Homology is correct for every degree in *degrees*; the Betti number
        at ``max(degrees)+1`` may be inflated by the truncation.
    weight:
        Optional fixed weight.  When provided, only basis elements of the
        given weight are included.  The module must expose
        ``graded_basis_by_weight(d, weight)``; if it does not, a
        :class:`ValueError` is raised.  Weight is the total number of
        "tensor factors" as defined by the module's ``_weight_on_basis``
        (for free algebras: the arity; for tree modules: the sum of leaf
        weights; for plain modules: the number of leaves).
    sparse:
        Whether to build differential matrices in sparse format.  This is
        usually faster and significantly lighter in memory for dg-modules
        with sparse boundaries.
    n_jobs:
        Number of worker processes to use for boundary-matrix assembly on the
        ``on_basis`` fast path.  Values above ``1`` currently use POSIX
        ``fork``-based multiprocessing and otherwise fall back to serial
        assembly.
    progress:
        When ``True``, emit a low-overhead progress indicator showing how many
        boundary columns have been assembled across all requested degrees.
    worker_profile_paths:
        Optional mutable list that receives per-worker ``.prof`` files when
        parallel boundary assembly is active.  These files can be merged into a
        parent :mod:`pstats` report with ``Stats.add(*worker_profile_paths)``.
    worker_profile_parent:
        Optional active top-level :class:`cProfile.Profile`.  When provided for
        a parallel run, forked workers disable their inherited copy before
        starting a per-worker profiler.

    Returns
    -------
    A SageMath :class:`~sage.homology.chain_complex.ChainComplex` with
    ``degree_of_differential=-1``.

    EXAMPLES::

        sage: from sage.all import QQ
        sage: from uconf import Surjection
        sage: from uconf.homology import compute_chain_complex
        sage: S2 = Surjection(2, QQ)
        sage: C = compute_chain_complex(S2, degrees=range(4))
        sage: C.betti()[0]
        1
    """
    base_ring = module.base_ring()
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be >= 1, got {n_jobs}")

    if not degrees:
        return ChainComplex({}, base_ring=base_ring, degree_of_differential=-1)

    if weight is not None and not hasattr(module, "graded_basis_by_weight"):
        raise ValueError(
            f"weight={weight} was specified but module {type(module).__name__} "
            "does not support the weight API.  "
            "Implement _weight_on_basis(key), basis_weight_iter(d, w), and "
            "graded_basis_by_weight(d, w) on the module first."
        )

    # Extend degrees by one in each direction to ensure we have the necessary basis elements to build the differentials for all degrees in the input range.  The chain complex will be correct in the input degrees; the extra degrees are just to ensure the differentials are correct and the homology computation can be performed without missing data.
    extended_degrees = range(min(degrees) - 1, max(degrees) + 2)

    # Collect basis elements and keys for each degree
    basis_by_degree: dict[int, list] = {}
    keys_by_degree: dict[int, list] = {}
    key_to_idx: dict[int, dict] = {}
    use_boundary_fast_path = get_on_basis(module.boundary) is not None

    _vprint("collecting basis elements...", verbose)
    for d in extended_degrees:
        _vprint(f"  basis degree {d}...", verbose)
        basis_elems, basis_keys = _get_basis_elements(
            module,
            d,
            weight,
            keep_elements=not use_boundary_fast_path,
        )
        basis_by_degree[d] = basis_elems
        keys_by_degree[d] = basis_keys
        key_to_idx[d] = {k: i for i, k in enumerate(basis_keys)}
        _vprint(f"  basis degree {d}: {len(basis_keys)} keys", verbose)

    # Build differential matrices: d_n : C_n -> C_{n-1}
    differentials: dict[int, Any] = {}
    total_columns = sum(len(keys_by_degree[d]) for d in extended_degrees if d - 1 in key_to_idx)
    _vprint(f"total boundary columns: {total_columns}", verbose)
    progress_reporter = _ChainComplexProgressReporter(total_columns, enabled=progress)

    # O1: Create a single worker manager for all boundary matrices so that
    # workers survive across degrees and keep their caches warm between
    # boundaries.
    # O2: Prewarm all cooperad caches for EVERY source key before forking so
    # that workers inherit fully populated caches via copy-on-write.
    use_persistent_pool = (
        n_jobs > 1
        and use_boundary_fast_path
        and os.name == "posix"
        and "fork" in multiprocessing.get_all_start_methods()
        and any(keys_by_degree[d] for d in extended_degrees if d - 1 in key_to_idx)
    )
    shared_pool: _WorkerManager | None = None
    try:
        if use_persistent_pool:
            boundary_on_basis_fn = get_on_basis(module.boundary)
            normalize_key_fn = getattr(module, "_normalize_key", None)
            # Collect all source keys across all boundary matrices to prewarm once.
            all_source_keys: tuple = tuple(
                k for d in extended_degrees if d - 1 in key_to_idx for k in keys_by_degree[d]
            )
            if prewarm:
                _vprint(
                    f"prewarming caches ({len(all_source_keys)} source keys across all degrees)...",
                    verbose,
                )
                _prewarm_parallel_boundary_caches(
                    module,
                    all_source_keys,
                    verbose=verbose,
                    profiler=prewarm_profiler,
                    parent_profiler=worker_profile_parent,
                )
            else:
                _vprint("prewarm disabled — skipping", verbose)
            # Set static state before forking; workers inherit it copy-on-write.
            # IMPORTANT: After this point, _BOUNDARY_MATRIX_STATE and all warmed
            # module caches MUST remain read-only.  Any write would trigger Linux
            # CoW page copies in workers, negating the memory savings.
            _BOUNDARY_MATRIX_STATE.clear()
            _BOUNDARY_MATRIX_STATE.update(
                {
                    "boundary_on_basis": boundary_on_basis_fn,
                    "normalize_key": normalize_key_fn,
                    "worker_profile": worker_profile_paths is not None,
                    "parent_profiler": worker_profile_parent,
                    "collect_boundary_profile": False,
                }
            )
            # Freeze GC after warmup and before forking.
            # This moves all currently tracked objects into the permanent GC
            # generation so the GC never updates their reference counts in
            # workers, preventing CoW page faults on the warm cache pages.
            gc.freeze()
            _vprint(f"forking workers ({n_jobs} processes)...", verbose)
            shared_pool = _WorkerManager(n_jobs)
            _vprint("workers ready", verbose)

        for d in extended_degrees:
            if d - 1 not in key_to_idx:
                # Target degree not in range; if there are source basis elements,
                # we still need a zero matrix so the complex knows the rank of C_d.
                if keys_by_degree[d]:
                    differentials[d] = matrix(base_ring, 0, len(keys_by_degree[d]), sparse=sparse)
                continue
            n_target = len(keys_by_degree[d - 1])
            source = basis_by_degree[d]
            if source is not None and not source and n_target == 0:
                continue
            _vprint(
                f"boundary matrix degree {d}: {len(keys_by_degree[d])} source × {n_target} target",
                verbose,
            )
            _t0 = time.perf_counter()
            differentials[d] = _boundary_matrix(
                module,
                keys_by_degree[d],
                key_to_idx[d - 1],
                n_target,
                sparse=sparse,
                n_jobs=n_jobs,
                basis_source=source,
                progress=progress_reporter,
                degree=d,
                worker_profile_paths=worker_profile_paths,
                worker_profile_parent=worker_profile_parent,
                pool=shared_pool,
            )
            _vprint(f"boundary matrix degree {d}: done ({time.perf_counter() - _t0:.1f}s)", verbose)
    finally:
        progress_reporter.close()
        if shared_pool is not None:
            # SIGKILL all workers immediately — no graceful shutdown.
            # This is intentional: workers hold large caches whose teardown
            # would be slow or could hang.
            shared_pool.kill_all()
            _BOUNDARY_MATRIX_STATE.clear()

    return ChainComplex(differentials, base_ring=base_ring, degree_of_differential=-1, check=check)


def compute_homology_representatives(
    module: Any,
    degree: int,
    weight: int | None,
    cc,
    *,
    algorithm: Literal["fast", "sage"] = "fast",
) -> list:
    """Compute cycle representatives for a basis of ``H_{degree}(cc)``.

    Two algorithms are available, controlled by the *algorithm* keyword:

    ``"fast"`` (default)
        Uses explicit linear algebra without invoking SageMath's Smith
        normal form machinery:

        1. Compute ``ker = ker(d_degree: C_degree → C_{degree-1})`` via
           :meth:`right_kernel` on the outgoing differential matrix.
        2. Compute ``im = im(d_{degree+1}: C_{degree+1} → C_degree)`` via
           the column space of the incoming differential.
        3. Row-reduce ``[im_basis; ker_basis]`` to echelon form.  The rows
           beyond the first ``dim(im)`` give homology representatives —
           they are in ``ker`` and their pivot columns are outside those of
           ``im``, so they are linearly independent modulo the image.

        This avoids the expensive ``generators=True`` path in SageMath and
        is significantly faster for large chain complexes.

    ``"sage"``
        Delegates to ``cc.homology(degree, generators=True)``.  Slower but
        produces representatives whose coefficients are expressed in
        SageMath's canonical reduced form, which can be easier to read.

    Args:
        module: The dg-module for which to compute representative cycles.
        degree: The homological degree at which to compute representatives.
        weight: Weight filter for basis selection, or ``None``.
        cc: A SageMath :class:`~sage.homology.chain_complex.ChainComplex`
            built from *module* via :func:`compute_chain_complex`.  Must
            contain differentials for *degree* and *degree*+1.
        algorithm: Which algorithm to use: ``"fast"`` (default) or
            ``"sage"``.

    Returns:
        A list of module elements that are cycles (``boundary(x) == 0``)
        whose homology classes form a basis of ``H_{degree}(cc)``.

    Raises:
        ValueError: If *algorithm* is not ``"fast"`` or ``"sage"``.
    """
    if algorithm == "sage":
        return _compute_homology_representatives_sage(module, degree, weight, cc)
    if algorithm == "fast":
        return _compute_homology_representatives_fast(module, degree, weight, cc)
    raise ValueError(f"Unknown algorithm {algorithm!r}: expected 'fast' or 'sage'.")


def _compute_homology_representatives_fast(
    module: Any, degree: int, weight: int | None, cc
) -> list:
    """Fast linear-algebra implementation; see :func:`compute_homology_representatives`."""
    basis_elems, _ = _get_basis_elements(module, degree, weight, keep_elements=True)
    if not basis_elems:
        return []

    # Step 1: ker(d_degree: C_degree -> C_{degree-1}).
    d_out = cc.differential(degree)
    ker = d_out.right_kernel()
    if ker.dimension() == 0:
        return []

    # Step 2: im(d_{degree+1}: C_{degree+1} -> C_degree).
    # cc.differential(degree+1) has shape dim(C_degree) × dim(C_{degree+1}).
    # Its column space is im(d_{degree+1}).
    d_in = cc.differential(degree + 1)
    K = ker.basis_matrix()  # k × n_degree, rows are cycles (row vectors)
    if d_in.ncols() == 0 or d_in.T.image().dimension() == 0:
        # im is trivial; homology = kernel
        rep_vecs = list(K.rows())
    else:
        I = d_in.column_space().basis_matrix()  # i × n_degree, rows span im(d_{d+1})
        # Step 3: row-reduce [I; K] so that the first dim(im) rows span im
        # and the remaining rows give independent homology representatives.
        E = I.stack(K).echelon_form()
        rank_I = I.rank()
        rep_vecs = [E.row(j) for j in range(rank_I, E.rank())]

    result: list = []
    for vec in rep_vecs:
        elem = module.zero()
        for i, coeff in enumerate(vec):
            if coeff:
                elem += coeff * basis_elems[i]
        result.append(elem)
    return result


def _compute_homology_representatives_sage(
    module: Any, degree: int, weight: int | None, cc
) -> list:
    """SageMath ``generators=True`` implementation; see :func:`compute_homology_representatives`."""
    basis_elems, _ = _get_basis_elements(module, degree, weight, keep_elements=True)

    ho = cc.homology(degree, generators=True)
    # cc.homology returns a bare tuple (not a list) when there are no
    # generators, so guard against that case.
    if not isinstance(ho, list):
        return []
    result: list = []
    for _vspace, gen in ho:
        vec = gen.vector(degree)
        elem = module.zero()
        for i, coeff in enumerate(vec):
            if coeff:
                elem += coeff * basis_elems[i]
        result.append(elem)
    return result


def homology_basis(
    module: Any,
    degree: int,
    *,
    degrees: range | None = None,
    weight: int | None = None,
) -> list:
    """Return cycle representatives for a basis of the homology in *degree*.

    Parameters
    ----------
    module:
        A dg-module (same requirements as :func:`compute_chain_complex`).
    degree:
        The homological degree in which to compute homology.
    degrees:
        Optional range of degrees to use when constructing the underlying
        chain complex.  Must include at least ``degree - 1``, ``degree``,
        and ``degree + 1`` so that both the incoming and outgoing
        differentials are available.  If ``None``, a minimal range
        ``range(degree - 1, degree + 2)`` is used (negative
        degrees are clamped to 0).  Pass a wider range if the module
        has non-trivial basis below ``degree - 1``.
    weight:
        Passed through to :func:`compute_chain_complex`. See its documentation.

    Returns
    -------
    A list of elements of *module* that are cycles (``boundary(x) == 0``)
    and whose homology classes form a basis of ``H_degree(module)``.

    EXAMPLES::

        sage: from sage.all import QQ
        sage: from uconf import Surjection
        sage: from uconf.homology import homology_basis
        sage: S2 = Surjection(2, QQ)
        sage: homology_basis(S2, 0)
        [{2 1}]
    """
    if degrees is None:
        degrees = range(degree - 1, degree + 2)
    else:
        if degree not in degrees:
            raise ValueError(f"degree {degree} must be contained in the supplied range {degrees}")

    C = compute_chain_complex(module, degrees, weight=weight)

    return compute_homology_representatives(module, degree, weight, C)
