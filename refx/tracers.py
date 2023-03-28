# Taken from flax/core/tracer.py 🏴‍☠️

from functools import partial
import jax
import jax.core
from jax._src.core import _why_alive


def current_trace(tree=()):
    """Returns the innermost Jax tracer."""
    return jax.core.find_top_trace(tree).main


def trace_level(main):
    """Returns the level of the trace of -infinity if it is None."""
    if main:
        return main.level
    return float("-inf")


def current_trace_level(tree=()):
    """Returns the level of the current trace."""
    return trace_level(current_trace(tree))
