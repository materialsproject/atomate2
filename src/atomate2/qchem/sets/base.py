"""Base Q-Chem input set and input set generator.

The Q-Chem input sets and generators are implemented in
:mod:`pymatgen.io.qchem.sets`. They are re-exported here so that existing imports
from ``atomate2.qchem.sets.base`` continue to work.
"""

from __future__ import annotations

from pymatgen.io.qchem.sets import QCInputGenerator, QCInputSet

__all__ = ["QCInputGenerator", "QCInputSet"]
