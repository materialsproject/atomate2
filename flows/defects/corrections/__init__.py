"""
Finite-size correction schemes for defect calculations.

This module provides multiple correction schemes for charged defect calculations:
- Lany-Zunger: Simple isotropic model-charge correction
- Makov-Payne: First-principles electrostatic correction
- Freysoldt: Anisotropic correction with potential alignment
- Kumagai-Oba: Extended Freysoldt with advanced alignment
- Slab2D: Specialized corrections for 2D materials and slabs
"""

from __future__ import annotations

from atomate2.siesta.flows.defects.corrections.base import (
    CorrectionResult,
    CorrectionScheme,
)
from atomate2.siesta.flows.defects.corrections.freysoldt import (
    FreysoldtCorrection,
)
from atomate2.siesta.flows.defects.corrections.lany_zunger import (
    LanyZungerCorrection,
)
from atomate2.siesta.flows.defects.corrections.makov_payne import (
    MakovPayneCorrection,
)
from atomate2.siesta.flows.defects.corrections.kumagai import KumagaiCorrection
from atomate2.siesta.flows.defects.corrections.slab_2d import (
    Slab2DCorrection,
    DielectricProfile,
    detect_slab_geometry,
)

__all__ = [
    "CorrectionScheme",
    "CorrectionResult",
    "LanyZungerCorrection",
    "MakovPayneCorrection",
    "FreysoldtCorrection",
    "KumagaiCorrection",
    "Slab2DCorrection",
    "DielectricProfile",
    "detect_slab_geometry",
]
