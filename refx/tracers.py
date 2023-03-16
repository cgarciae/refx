# Taken from flax/core/tracer.py 🏴‍☠️

import jax


def current_trace():
    """Returns the innermost Jax tracer."""
    return jax.core.find_top_trace(())


def trace_level(main):
    """Returns the level of the trace of -infinity if it is None."""
    if main:
        return main.level
    return float("-inf")


def current_trace_level():
    """Returns the level of the current trace."""
    return trace_level(current_trace())
