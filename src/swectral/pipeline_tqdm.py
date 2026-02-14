# -*- coding: utf-8 -*-
"""
Swectral - Pipeline file-based flexible progress bar

The following source code was created with AI assistance and has been human reviewed and edited.

Copyright (c) 2025 Siwei Luo. MIT License.
"""

import os
import sys
import time

# Regular expression
import re

from typing import Sequence, Optional, Any

import traceback
import threading
from pathos.helpers import mp as pmp

from .specio import simple_type_validator


# %% Scan patterns


def scan_patterns(root: str, patterns: list[str]) -> list[int]:
    """
    Scan a root directory and count files matching each step pattern.

    Each step pattern is a regex string that matches filenames belonging
    to that pipeline step (e.g., step_1, step_2, ...).

    Parameters
    ----------
    root : str
        Root directory containing all generated files.
    patterns : list of str
        Ordered list of regex patterns, one per pipeline step.

    Returns
    -------
    list of int
        Number of files matching each step pattern.
    """
    compiled: list[re.Pattern[str]] = [re.compile(p) for p in patterns]
    counts: list[int] = [0] * len(compiled)

    try:
        names = os.listdir(root)
    except FileNotFoundError:
        return counts

    for name in names:
        full = os.path.join(root, name)
        if not os.path.isfile(full):
            continue

        for i, pat in enumerate(compiled):
            if pat.search(name):
                counts[i] += 1
                break

    return counts


# %% Remaining potential of branching


def remaining_factors(nbranches: list[int]) -> list[int]:
    """
    Compute remaining branching potential for each pipeline stage.

    For a pipeline with branching factors b1, b2, ..., bN, the remaining factor at stage i is::

        remaining[i] = ∏_{j=i+1..N} b_j

    This represents how many final-stage files a single file at stage i can still potentially generate.

    Parameters
    ----------
    nbranches : list of int
        Branching factors for each pipeline stage.

    Returns
    -------
    list of int
        A list of integers of the same length as nbranches, where each element is the remaining multiplicative potential for that stage.

    Examples
    --------
    >>> nbranches = [2, 3, 5]
    >>> returns  [15, 5, 1]
    """  # noqa: E501
    n: int = len(nbranches)
    rem: list[int] = [1] * n

    acc: int = 1
    for i in range(n - 1, -1, -1):
        rem[i] = acc
        acc *= nbranches[i]

    return rem


# %% Pipeline progress computation


def pipeline_progress(
    counts: Sequence[int],
    nbranches: Sequence[int],
    nsample: int,
) -> float:
    """
    Compute global pipeline progress from a filesystem snapshot.

    This function implements a *branching-potential conservation model*.
    Each file contributes a fraction of a final output based on how many
    downstream nbranches remain.

    Properties::

        - Snapshot-based (filesystem only).
        - Independent of scheduling and multiprocessing.
        - Monotonic under forward-only file movement.
        - Reaches 1.0 if and only if all final-stage files exist.

    Progress is defined as::

        progress = (realized final-output potential) / (total potential)

    where::

        total potential = samples × ∏ branches

    Parameters
    ----------
    counts : Sequence of int
        Number of files currently present at each pipeline stage.
    nbranches : Sequence of int
        Branching factors for each pipeline stage.
    nsample : int
        Number of independent samples entering the pipeline.

    Returns
    -------
    float
        A float in [0.0, 1.0] representing the fraction of total final output that has been effectively realized so far.
    """
    n: int = len(counts)
    if n == 0:
        return 0.0

    rem: list[int] = remaining_factors(list(nbranches))

    total_final: int = nsample
    for b in nbranches:
        total_final *= b

    if total_final == 0:
        return 0.0

    done: float = 0.0
    for k, r in zip(counts, rem):
        done += k / r

    return min(done / total_final, 1.0)


# %% TODO: File-based pipeline tqdm


# Status Constants
STATUS_RUNNING: int = 0
STATUS_DONE: int = 1
STATUS_ERROR: int = 2


def _tqdm_scanner_worker_pathos(
    root: str,
    patterns: list[str],
    nbranches: list[int],
    nsample: int,
    poll_interval: float,
    shared_data: dict[str, Any],
    stop_event: object,
) -> None:
    """
    Worker function using Pathos Manager for shared state.

    Parameters
    ----------
    root : str
        Root directory containing pattern subdirectories.
    patterns : list of str
        Ordered list of pattern directory names.
    nbranches : list of int
        Branching factors for each pattern.
    nsample : int
        Number of independent samples entering the pipeline.
    poll_interval : float
        Time in seconds between filesystem scans.
    shared_data : dict map str to Any
        Multiprocessing manager for shared data of ``pathos.helper.mp.Manager.dict``.
    stop_event : object
        Multiprocessing ``pathos.helper.mp.Event``.
    """
    try:
        # Re-import checks
        # from your_module import scan_patterns, pipeline_progress
        assert hasattr(stop_event, "is_set")
        while not stop_event.is_set():
            # 1. Check for completion file
            if os.path.exists(os.path.join(root, ".__finished.s")):
                counts: list[int] = scan_patterns(root, patterns)

                # Update Manager dict
                shared_data['counts'] = counts
                shared_data['progress'] = 1.0
                shared_data['status'] = STATUS_DONE
                break

            # 2. Perform Scan
            counts = scan_patterns(root, patterns)
            progress: float = pipeline_progress(counts, nbranches, nsample)

            # 3. Update Manager dict
            # Note: Manager dicts are process-safe.
            shared_data['counts'] = counts
            shared_data['progress'] = progress

            # 4. Completion Check
            if progress >= 1.0:
                shared_data['status'] = STATUS_DONE
                break

            if stop_event.is_set():
                break

            # 5. Smart Sleep
            for _ in range(int(max(0.1, poll_interval) * 10)):
                if stop_event.is_set():
                    return
                time.sleep(0.1)

    except Exception:
        # Log to file and update status
        try:
            with open(os.path.join(root, ".__pipeline_error.log"), "w") as f:
                f.write(traceback.format_exc())
        except OSError:
            pass

        shared_data['status'] = STATUS_ERROR


class PipelineFileTqdm:
    """
    A tqdm-like progress bar driven entirely by filesystem state.

    This class periodically scans pattern directories and computes progress using a branching-potential model.
    It does not depend on loop iterations, task counts, or explicit synchronization with worker processes.

    Intended use::

        - Multiprocessing pipelines
        - Asynchronous or staggered sample execution
        - Branching workflows with unreachable intermediate maxima
    """

    @simple_type_validator
    def __init__(
        self,
        root: str,
        patterns: list[str],
        nbranches: list[int],
        nsample: int,
        width: int = 40,
        poll_interval: float = 0.5,
    ) -> None:
        """
        Initialize the filesystem-driven progress bar.

        Parameters
        ----------
        root : str
            Root directory containing pattern subdirectories.
        patterns : list of str
            Ordered list of pattern directory names.
        nbranches : list of int
            Branching factors for each pattern.
        nsample : int
            Number of independent samples entering the pipeline.
        width : int
            Character width of the progress bar.
        poll_interval : float
            Time in seconds between filesystem scans.
        """
        self.root: str = root
        self.patterns: list[str] = patterns
        self.nbranches: list[int] = nbranches
        self.nsample: int = nsample
        self.width: int = width
        self.poll_interval: float = poll_interval

        # --- PATHOS MANAGER SETUP ---
        self.manager = pmp.Manager()
        self.shared_data = self.manager.dict()

        # Initialize shared state
        self.shared_data['counts'] = [0] * len(patterns)
        self.shared_data['progress'] = 0.0
        self.shared_data['status'] = STATUS_RUNNING

        # We still use a native Event for the kill switch as it's lighter
        self.child_stop_event = pmp.Event()

        self.process: Optional[pmp.Process] = None
        self.print_thread: Optional[threading.Thread] = None
        self.thread_stop_event = threading.Event()

    def _render(self) -> None:
        # Read from Manager dict
        try:
            progress = self.shared_data.get('progress', 0.0)
            counts = self.shared_data.get('counts', [0] * len(self.patterns))
        except (BrokenPipeError, EOFError):
            return

        progress = max(0.0, min(1.0, progress))
        filled = int(self.width * progress)
        bar = "█" * filled + " " * (self.width - filled)

        status_parts = []
        for p, c in zip(self.patterns, counts):
            name = p[1:] if p.startswith("_") else p
            status_parts.append(f"{name}: {c}")
        status = " ".join(status_parts)

        sys.stderr.write(f"\r[{bar}] {progress * 100:6.2f}% | {status}")
        sys.stderr.flush()

    def _printer_loop(self) -> None:
        while not self.thread_stop_event.is_set():
            try:
                status = self.shared_data.get('status', STATUS_RUNNING)
            except (BrokenPipeError, EOFError):
                break

            if status == STATUS_ERROR:
                sys.stderr.write("\n[!!!] Monitor Error. Check '.__pipeline_error.log'\n")
                break

            self._render()

            if status == STATUS_DONE:
                sys.stderr.write("\n")
                break

            time.sleep(0.1)

    def start(self) -> None:
        self.child_stop_event.clear()
        self.thread_stop_event.clear()

        # Reset state
        self.shared_data['counts'] = [0] * len(self.patterns)
        self.shared_data['progress'] = 0.0
        self.shared_data['status'] = STATUS_RUNNING

        # Use pathos.multiprocessing.Process
        self.process = pmp.Process(
            target=_tqdm_scanner_worker_pathos,
            args=(
                self.root,
                self.patterns,
                self.nbranches,
                self.nsample,
                self.poll_interval,
                self.shared_data,
                self.child_stop_event,
            ),
        )
        self.process.start()

        self.print_thread = threading.Thread(target=self._printer_loop, daemon=True)
        self.print_thread.start()

    def join(self) -> None:
        self.child_stop_event.set()

        if self.process:
            self.process.join(timeout=2.0)
            if self.process.is_alive():
                self.process.terminate()

        self.thread_stop_event.set()
        if self.print_thread:
            self.print_thread.join()

        # Optional: Shutdown manager to clean up the server process
        # self.manager.shutdown()

    def __enter__(self) -> "PipelineFileTqdm":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.join()


# %% TODO: DirProgressObserver


def _dir_scanner_worker_pathos(  # noqa: C901
    root: str,
    total: int,
    pattern_str: str,
    poll_interval: float,
    shared_data: dict[str, Any],
    stop_event: object,
) -> None:
    """
    Worker function executed in a separate child process.
    Scans the directory for matching subdirectories and reports count via Queue.

    Parameters
    ----------
    root : str
        Root directory to monitor.
    total : int
        Total number of directories expected.
    pattern : str
        Regular expression pattern used to match directory names.
    poll_interval : float, optional
        Seconds to wait between scans.
    shared_data : dict map str to Any
        Multiprocessing manager for shared data of ``pathos.helper.mp.Manager.dict``.
    stop_event : object
        Multiprocessing ``pathos.helper.mp.Event``.
    """

    try:
        pattern = re.compile(pattern_str)

        assert hasattr(stop_event, "is_set")
        while not stop_event.is_set():
            if os.path.exists(os.path.join(root, ".__finished.s")):
                shared_data['status'] = STATUS_DONE
                break

            try:
                c = 0
                if os.path.exists(root):
                    with os.scandir(root) as it:
                        for entry in it:
                            if entry.is_dir() and pattern.match(entry.name):
                                c += 1
                shared_data['count'] = c
            except OSError:
                shared_data['count'] = 0

            if shared_data.get('count', 0) >= total:
                shared_data['status'] = STATUS_DONE
                break

            for _ in range(int(poll_interval * 10)):
                if stop_event.is_set():
                    return
                time.sleep(0.1)

    except Exception:
        shared_data['status'] = STATUS_ERROR


class DirProgressObserver:

    @simple_type_validator
    def __init__(
        self,
        root: str,
        total: int,
        pattern: str,
        progress_prefix: str,
        poll_interval: float = 0.5,
    ) -> None:
        """
        Observe a root directory and report progress based on how many subdirectories matching a regex pattern have been created.

        Parameters
        ----------
        root : str
            Root directory to monitor.
        total : int
            Total number of directories expected.
        pattern : str
            Regular expression pattern used to match directory names.
        progress_prefix : str
            Prefix printed before the progress counter.
        poll_interval : float, optional
            Seconds to wait between scans.
        """  # noqa: E501

        self.root = root
        self.total = total
        self.pattern_str = pattern
        self.progress_prefix = progress_prefix
        self.poll_interval = poll_interval

        # Pathos Manager
        self.manager = pmp.Manager()
        self.shared_data = self.manager.dict()
        self.shared_data['count'] = 0
        self.shared_data['status'] = STATUS_RUNNING

        self.child_stop_event = pmp.Event()
        self.process: Optional[pmp.Process] = None
        self.print_thread: Optional[threading.Thread] = None
        self.thread_stop_event = threading.Event()

    def _printer_loop(self) -> None:
        while not self.thread_stop_event.is_set():
            try:
                status = self.shared_data.get('status', STATUS_RUNNING)
                count = self.shared_data.get('count', 0)
            except (BrokenPipeError, EOFError):
                break

            if status == STATUS_ERROR:
                sys.stderr.write(f"\n[!!!] {self.progress_prefix} Error.\n")
                break

            display_count = min(count, self.total)
            sys.stderr.write(f"\r{self.progress_prefix}: {display_count}/{self.total}")
            sys.stderr.flush()

            if status == STATUS_DONE:
                sys.stderr.write("\n")
                break

            time.sleep(0.1)

    # ... (start, join, enter, exit methods Identical to PipelineFileTqdm above) ...
    # Just copy the start/join/enter/exit logic from the class above
    def start(self) -> None:
        self.child_stop_event.clear()
        self.thread_stop_event.clear()
        self.shared_data['count'] = 0
        self.shared_data['status'] = STATUS_RUNNING

        self.process = pmp.Process(
            target=_dir_scanner_worker_pathos,
            args=(
                self.root,
                self.total,
                self.pattern_str,
                self.poll_interval,
                self.shared_data,
                self.child_stop_event,
            ),
        )
        self.process.start()
        self.print_thread = threading.Thread(target=self._printer_loop, daemon=True)
        self.print_thread.start()

    def join(self) -> None:
        self.child_stop_event.set()
        if self.process:
            self.process.join(timeout=2.0)
            if self.process.is_alive():
                self.process.terminate()
        self.thread_stop_event.set()
        if self.print_thread:
            self.print_thread.join()

    def __enter__(self) -> "DirProgressObserver":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.join()
