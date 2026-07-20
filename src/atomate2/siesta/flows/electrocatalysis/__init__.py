"""Electrocatalysis workflow module for atomate2siesta.

This module provides workflows for electrochemical reactions including:
- Oxygen Reduction Reaction (ORR)
- Oxygen Evolution Reaction (OER)
- Hydrogen Evolution Reaction (HER)
- Bifunctional ORR/OER (combined workflow)
- CO₂ Reduction Reaction (CO₂RR)
- General reaction pathway analysis

The module follows a modular design with composable FlowMakers that can be
combined for complex electrocatalysis studies.

Phase 1 (COMPLETE): Gas-phase calculations, spin configuration utilities
Phase 2 (COMPLETE): ORR/OER workflows with thermodynamic analysis, tests, tutorials
Phase 3 (COMPLETE): HER workflow, Bifunctional ORR/OER
Phase 4 (PENDING): CO₂RR workflow
"""

from atomate2.siesta.flows.electrocatalysis.bifunctional import BifunctionalFlowMaker
from atomate2.siesta.flows.electrocatalysis.her import HERFlowMaker
from atomate2.siesta.flows.electrocatalysis.oer import OERFlowMaker
from atomate2.siesta.flows.electrocatalysis.orr import ORRFlowMaker

__all__ = ["BifunctionalFlowMaker", "HERFlowMaker", "OERFlowMaker", "ORRFlowMaker"]
