"""Analysis tools for electrocatalysis workflows.

This module provides thermodynamic analysis tools for electrochemical reactions:
- Computational Hydrogen Electrode (CHE) model for free energy calculations
- Overpotential analysis for ORR, OER, HER, CO2RR
- Volcano plot generation
- Rate-limiting step identification
"""

from atomate2.siesta.flows.electrocatalysis.analysis.overpotential import (
    calculate_bifunctional_gap,
    calculate_her_overpotential,
    calculate_oer_overpotential,
    calculate_orr_overpotential,
)
from atomate2.siesta.flows.electrocatalysis.analysis.thermodynamics import (
    calculate_free_energy_corrections,
    calculate_reaction_free_energies,
    identify_rate_limiting_step,
)

__all__ = [
    "calculate_bifunctional_gap",
    "calculate_free_energy_corrections",
    "calculate_her_overpotential",
    "calculate_oer_overpotential",
    "calculate_orr_overpotential",
    "calculate_reaction_free_energies",
    "identify_rate_limiting_step",
]
