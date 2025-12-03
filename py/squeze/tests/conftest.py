"""
Configuration for pytest tests.
This file is automatically loaded by pytest and runs before any test imports.
"""
import os

# Disable Numba JIT compilation for tests to ensure coverage tools can track
# all code paths
# Set NUMBA_DISABLE_JIT before any modules are imported
os.environ['NUMBA_DISABLE_JIT'] = '1'
