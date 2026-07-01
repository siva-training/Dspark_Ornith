"""Small helpers for reporting/measuring process memory footprint."""

import resource
import sys


def peak_rss_mb():
    """Peak resident set size of the current process, in MiB."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # Linux reports ru_maxrss in KiB; macOS reports it in bytes.
    if sys.platform == "darwin":
        return ru.ru_maxrss / (1024 * 1024)
    return ru.ru_maxrss / 1024


def children_peak_rss_mb():
    """Peak resident set size across reaped child processes, in MiB."""
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    if sys.platform == "darwin":
        return ru.ru_maxrss / (1024 * 1024)
    return ru.ru_maxrss / 1024
