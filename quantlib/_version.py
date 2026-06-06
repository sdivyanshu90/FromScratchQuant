"""
Module: quantlib._version

Single source of truth for the library version string.

Mathematical Background:
    None — version metadata only.

References:
    PEP 440 — Version Identification and Dependency Specification.

Example:
    >>> from quantlib._version import __version__
    >>> __version__
    '0.1.0'
"""

from typing import Final

__version__: Final[str] = "0.1.0"
