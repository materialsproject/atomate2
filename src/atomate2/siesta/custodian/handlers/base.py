"""Base error handler class for SIESTA calculations.

This module re-exports ErrorHandler from the custodian library
for use in SIESTA-specific error handlers.
"""

from __future__ import annotations

# Import ErrorHandler from custodian library
from custodian.custodian import ErrorHandler

# Re-export for backward compatibility
__all__ = ["ErrorHandler"]
