from __future__ import annotations

import pytest
from custodian.vasp.handlers import ErrorHandler
from pymatgen.core import Lattice, Structure

from atomate2.vasp.run import DEFAULT_HANDLERS

def test_default_handlers():
    assert len(DEFAULT_HANDLERS) >= 8
    assert all(isinstance(handler, ErrorHandler) for handler in DEFAULT_HANDLERS)
