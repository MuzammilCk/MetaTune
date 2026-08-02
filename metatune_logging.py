# metatune_logging.py
"""Shared logging setup for MetaTune's core library: agent.py and its
dependency chain (data_analyzer.py, brain.py, engine.py, bilevel.py).

These modules previously used bare print() everywhere: no log levels, no
way to filter verbosity, no way to route output anywhere but stdout, and
no way for a caller embedding MetaTune in a larger application to tell
its output apart from anything else on stdout.

get_logger() gives each module a standard logging.Logger. By default the
formatter prints just the message — MetaTune's existing narrator-style
output already carries its own cues (❌/⚠️/✓/🤖), so nothing changes
visually at the default level — but it's a real, filterable, redirectable
logger underneath, with level control via --log-level on the agent CLI
(see agent.py's main()).

Usage, in any module:
    from metatune_logging import get_logger
    logger = get_logger(__name__)
    logger.info("Something happened")
    logger.warning("Something recoverable went wrong")
    logger.error("Something failed")
"""
import logging
import sys
from typing import Optional

_CONFIGURED = False


class _PlainFormatter(logging.Formatter):
    """Just the message, matching the previous print()-based output. Use
    a different formatter on the handler yourself (e.g. via
    logging.getLogger("metatune").handlers[0].setFormatter(...)) if you
    want timestamps/levels for production log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()


def configure_logging(level: int = logging.INFO, stream=None) -> None:
    """Configure the "metatune" logger tree once. Safe to call more than
    once (e.g. once per CLI invocation) — later calls just adjust the
    level rather than stacking duplicate handlers.

    Not required before using get_logger(): if this is never called
    (e.g. because data_analyzer.py or brain.py is imported directly by
    something other than agent.py's CLI), Python's logging defaults apply
    (WARNING and above, to stderr) rather than raising or silently
    dropping messages.
    """
    global _CONFIGURED
    root = logging.getLogger("metatune")
    if not _CONFIGURED:
        handler = logging.StreamHandler(stream or sys.stdout)
        handler.setFormatter(_PlainFormatter())
        root.addHandler(handler)
        root.propagate = False  # don't also hand records to the root logger
        _CONFIGURED = True
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Get a module-scoped logger under the shared "metatune" namespace.

    Loggers are namespaced as "metatune.<name>" (pass __name__) so
    configure_logging()'s level applies uniformly across every module,
    and so an embedding application can filter or redirect just
    MetaTune's output via the "metatune" logger name.
    """
    return logging.getLogger(f"metatune.{name}")


def level_from_name(name: str, default: int = logging.INFO) -> int:
    """Resolve a --log-level string (e.g. from argparse) to a logging
    level constant, falling back to `default` for an unrecognized name
    rather than raising — a CLI flag typo shouldn't crash the run."""
    return getattr(logging, name.upper(), default) if isinstance(name, str) else default
