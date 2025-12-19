""" Utility helpers for conditional Numba JIT compilation.

This module centralizes all interaction with Numba so that JIT compilation
can be enabled or completely disabled via the ``NUMBA_DISABLE_JIT``
environment variable. When ``NUMBA_DISABLE_JIT=1``, the public decorators
exposed here (``jit``, ``njit``, ``vectorize``) become no-ops or fall back
to pure-Python / NumPy implementations, and ``prange`` degrades to the
built-in :func:`range`. In normal operation (when the variable is unset or
not equal to ``"1"``), these names are direct aliases to their Numba
counterparts and ``NUMBA_TYPES`` exposes :mod:`numba.types`.
Code outside this module should import JIT-related helpers from
``squeze.numba_utils`` instead of importing Numba directly. This allows
tests, debugging sessions, and coverage runs to disable JIT compilation
uniformly, without changing call sites or risking inconsistent behavior
across the codebase.
"""
import os

import numba
import numpy as np

# Handle JIT compilation conditionally for testing/coverage
# Check if JIT is disabled
if os.environ.get('NUMBA_DISABLE_JIT', '0') == '1':
    # Create dummy decorators and use regular range
    # pylint: disable=unused-argument
    def jit(*args, **kwargs):
        """Replacement for numba.jit when JIT is disabled."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            # Used as @jit without parentheses
            return args[0]

        def decorator(func):
            """Decorator that returns the function unchanged."""
            return func

        return decorator

    def njit(*args, **kwargs):
        """A no-op decorator to disable JIT compilation."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            # Used as @njit without parentheses
            return args[0]

        def decorator(func):
            """Return the original function without JIT compilation."""
            return func

        return decorator

    NUMBA_TYPES = None

    prange = range  # pylint: disable=invalid-name

    # For vectorize, we need to directly use np.vectorize
    vectorize = np.vectorize  # pylint: disable=invalid-name

else:
    jit = numba.jit
    njit = numba.njit
    NUMBA_TYPES = numba.types
    vectorize = numba.vectorize  # pylint: disable=invalid-name
    prange = numba.prange  # pylint: disable=invalid-name
