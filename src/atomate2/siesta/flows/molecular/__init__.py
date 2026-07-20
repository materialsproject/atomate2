"""Molecular calculation workflows for atomate2siesta.

This module provides general-purpose workflows for gas-phase molecular calculations,
useful for:
- Reference energy calculations (O₂, H₂O, CO₂, etc.)
- Molecular thermochemistry
- Reaction energies
- Molecular properties (dipole moment, polarizability)
"""

from atomate2.siesta.flows.molecular.gas_phase import GasPhaseMoleculeMaker

__all__ = ["GasPhaseMoleculeMaker"]
