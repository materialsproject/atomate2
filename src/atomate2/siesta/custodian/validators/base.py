"""Base validator class for SIESTA calculations.

This module re-exports Validator from the custodian library
for use in SIESTA-specific validators.
"""

from __future__ import annotations

# Import Validator from custodian library
from custodian.custodian import Validator

# Re-export for backward compatibility
__all__ = ["Validator"]

# Legacy alias for backward compatibility
OutputValidator = Validator
