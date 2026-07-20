"""Overpotential calculations for electrochemical reactions.

This module provides functions to calculate overpotentials for common electrocatalytic
reactions (ORR, OER, HER, CO2RR) from DFT-calculated free energy diagrams.

The overpotential (η) quantifies the additional voltage needed beyond the thermodynamic
equilibrium potential to drive a reaction at a given rate. Lower |η| indicates better
catalyst performance.

References
----------
- Nørskov et al., J. Phys. Chem. B 108, 17886 (2004)
- Man et al., ChemCatChem 3, 1159 (2011)
- Koper, Chem. Sci. 4, 2710 (2013)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# Standard equilibrium potentials (vs. SHE at pH = 0, 298.15 K)
U_EQ_ORR = 1.23  # V (O₂ + 4H⁺ + 4e⁻ → 2H₂O)
U_EQ_OER = 1.23  # V (2H₂O → O₂ + 4H⁺ + 4e⁻, reverse of ORR)
U_EQ_HER = 0.00  # V (2H⁺ + 2e⁻ → H₂, reference)
U_EQ_CO2RR_CO = -0.10  # V (CO₂ + 2H⁺ + 2e⁻ → CO + H₂O)
U_EQ_CO2RR_CH4 = 0.17  # V (CO₂ + 8H⁺ + 8e⁻ → CH₄ + 2H₂O)
U_EQ_CO2RR_C2H4 = 0.08  # V (2CO₂ + 12H⁺ + 12e⁻ → C₂H₄ + 4H₂O)


def calculate_orr_overpotential(
    delta_G: Sequence[float],  # noqa: N803
    method: str = "max_uphill",
) -> dict[str, float]:
    """
    Calculate ORR overpotential from free energy profile.

    The ORR overpotential quantifies how much additional voltage is needed
    beyond the equilibrium potential (1.23 V) to make all reaction steps
    thermodynamically favorable.

    Two common methods:
    1. **Max uphill** (default): η = U_eq - (U_eq - ΔG_max)
        where ΔG_max is the largest uphill step.
    2. **Onset potential**: U_onset where all steps become downhill.

    Parameters
    ----------
    delta_G : Sequence[float]
        Free energy changes for each reaction step (eV).
        Calculated from CHE model at U = 0 V.
    method : str
        Method for overpotential calculation:
        - 'max_uphill' (default): Based on rate-limiting step
        - 'onset': Onset potential method

    Returns
    -------
    dict
        {
            'eta_ORR': float,           # ORR overpotential (V, positive value)
            'U_onset': float,           # Onset potential (V vs. SHE)
            'U_equilibrium': float,     # Equilibrium potential (V, always 1.23)
            'max_delta_G': float,       # Maximum uphill step (eV)
            'rls_index': int,           # Index of rate-limiting step
        }

    Notes
    -----
    **ORR overpotential definition:**

    For ORR (reduction, cathode):
        η_ORR = U_eq - U_onset

    where:
        - U_eq = 1.23 V (thermodynamic limit)
        - U_onset = potential where all ΔG ≤ 0

    **Physical meaning:**
        - η_ORR > 0: Voltage loss (requires more negative potential)
        - Lower η_ORR → better ORR catalyst
        - Ideal catalyst: η_ORR = 0 V (all steps thermoneutral)

    **Relation to experiment:**
        - Theoretical η from DFT: thermodynamic overpotential
        - Experimental η: includes kinetic barriers (activation energy)
        - DFT η provides lower bound for experimental η

    References
    ----------
    - Nørskov et al., J. Phys. Chem. B 108, 17886 (2004): CHE model
    - Viswanathan et al., ACS Catal. 2, 1654 (2012): Scaling relations

    Examples
    --------
    >>> # ORR pathway with 4 steps
    >>> delta_G = [0.45, 1.20, 0.80, 0.60]  # eV at U = 0 V
    >>> result = calculate_orr_overpotential(delta_G)
    >>> result["eta_ORR"]
    1.20  # V (dominated by largest uphill step: 1.20 eV)
    >>> result["U_onset"]
    0.03  # V (1.23 - 1.20)

    >>> # Ideal catalyst (all steps thermoneutral)
    >>> delta_G_ideal = [0.3075, 0.3075, 0.3075, 0.3075]  # 1.23/4 each
    >>> result = calculate_orr_overpotential(delta_G_ideal)
    >>> result["eta_ORR"]
    0.3075  # V (each step is 0.3075 eV)
    """
    if not delta_G:
        logger.warning("Empty delta_G list, returning zero overpotential")
        return {
            "eta_ORR": 0.0,
            "U_onset": U_EQ_ORR,
            "U_equilibrium": U_EQ_ORR,
            "max_delta_G": 0.0,
            "rls_index": -1,
        }

    # Find maximum uphill step (rate-limiting step)
    max_delta_G = float(np.max(delta_G))  # noqa: N806
    rls_index = int(np.argmax(delta_G))

    # Calculate onset potential
    # At U_onset, the largest step becomes ΔG = 0
    # ΔG(U) = ΔG(U=0) - eU
    # Setting ΔG(U_onset) = 0:
    # U_onset = ΔG(U=0) / e = ΔG_max (in eV, e = 1)
    # But we want U_onset = U_eq - η
    # So: η = U_eq - (U_eq - ΔG_max) = ΔG_max
    U_onset = U_EQ_ORR - max_delta_G  # noqa: N806

    # Overpotential (always positive for ORR)
    eta_ORR = U_EQ_ORR - U_onset  # noqa: N806

    logger.info(
        f"ORR overpotential: η = {eta_ORR:.3f} V "
        f"(U_onset = {U_onset:.3f} V, RLS at step {rls_index})"
    )

    return {
        "eta_ORR": eta_ORR,
        "U_onset": U_onset,
        "U_equilibrium": U_EQ_ORR,
        "max_delta_G": max_delta_G,
        "rls_index": rls_index,
    }


def calculate_oer_overpotential(
    delta_G: Sequence[float],  # noqa: N803
) -> dict[str, float]:
    """
    Calculate OER overpotential from free energy profile.

    OER is the reverse of ORR:
        2H₂O → O₂ + 4H⁺ + 4e⁻  (oxidation, anode)

    The overpotential quantifies additional voltage needed beyond 1.23 V
    to make all oxidation steps thermodynamically favorable.

    Parameters
    ----------
    delta_G : Sequence[float]
        Free energy changes for OER steps (eV).
        Typically: H₂O → OH* → O* → OOH* → O₂

    Returns
    -------
    dict
        {
            'eta_OER': float,           # OER overpotential (V, positive value)
            'U_onset': float,           # Onset potential (V vs. SHE)
            'U_equilibrium': float,     # Equilibrium potential (V, always 1.23)
            'max_delta_G': float,       # Maximum uphill step (eV)
            'rls_index': int,           # Index of rate-limiting step
        }

    Notes
    -----
    **OER overpotential definition:**

    For OER (oxidation, anode):
        η_OER = U_onset - U_eq

    where:
        - U_eq = 1.23 V (thermodynamic limit)
        - U_onset = potential where all ΔG ≤ 0 (all steps downhill)

    **Physical meaning:**
        - η_OER > 0: Voltage loss (requires more positive potential)
        - Lower η_OER → better OER catalyst
        - Ideal catalyst: η_OER = 0 V

    **Relation to ORR:**
        - OER and ORR are reverse reactions
        - η_ORR + η_OER = "overpotential gap"
        - Smaller gap → better bifunctional catalyst

    Examples
    --------
    >>> # OER pathway (reverse of ORR)
    >>> delta_G = [0.60, 0.80, 1.20, 0.45]  # eV
    >>> result = calculate_oer_overpotential(delta_G)
    >>> result["eta_OER"]
    1.20  # V

    >>> result["U_onset"]
    2.43  # V (1.23 + 1.20)
    """
    if not delta_G:
        logger.warning("Empty delta_G list, returning zero overpotential")
        return {
            "eta_OER": 0.0,
            "U_onset": U_EQ_OER,
            "U_equilibrium": U_EQ_OER,
            "max_delta_G": 0.0,
            "rls_index": -1,
        }

    # Find maximum uphill step
    max_delta_G = float(np.max(delta_G))  # noqa: N806
    rls_index = int(np.argmax(delta_G))

    # For OER (oxidation), onset potential is ABOVE equilibrium
    # U_onset = U_eq + ΔG_max
    U_onset = U_EQ_OER + max_delta_G  # noqa: N806

    # Overpotential (always positive for OER)
    eta_OER = U_onset - U_EQ_OER  # noqa: N806

    logger.info(
        f"OER overpotential: η = {eta_OER:.3f} V "
        f"(U_onset = {U_onset:.3f} V, RLS at step {rls_index})"
    )

    return {
        "eta_OER": eta_OER,
        "U_onset": U_onset,
        "U_equilibrium": U_EQ_OER,
        "max_delta_G": max_delta_G,
        "rls_index": rls_index,
    }


def calculate_bifunctional_gap(
    delta_G_ORR: Sequence[float],  # noqa: N803
    delta_G_OER: Sequence[float],  # noqa: N803
) -> dict[str, float]:
    """
    Calculate bifunctional overpotential gap for ORR and OER.

    A good bifunctional catalyst (e.g., for metal-air batteries or
    water splitting) should have low overpotentials for BOTH ORR and OER.

    The gap quantifies the total voltage loss:
        ΔU = η_ORR + η_OER

    Parameters
    ----------
    delta_G_ORR : Sequence[float]
        Free energy changes for ORR pathway (eV).
    delta_G_OER : Sequence[float]
        Free energy changes for OER pathway (eV).

    Returns
    -------
    dict
        {
            'eta_ORR': float,                # ORR overpotential (V)
            'eta_OER': float,                # OER overpotential (V)
            'overpotential_gap': float,      # Total gap η_ORR + η_OER (V)
            'U_ORR_onset': float,            # ORR onset potential (V)
            'U_OER_onset': float,            # OER onset potential (V)
            'voltage_window': float,         # U_OER_onset - U_ORR_onset (V)
        }

    Notes
    -----
    **Ideal bifunctional catalyst:**
        - η_ORR = 0 V
        - η_OER = 0 V
        - Gap = 0 V
        - Voltage window = 0 V (same onset for ORR and OER)

    **Practical targets** (metal-air batteries):
        - Gap < 0.4 V: Excellent
        - Gap < 0.6 V: Good
        - Gap < 0.8 V: Acceptable
        - Gap > 1.0 V: Poor

    Examples
    --------
    >>> delta_G_ORR = [0.45, 1.20, 0.80, 0.60]
    >>> delta_G_OER = [0.60, 0.80, 1.20, 0.45]  # Reverse pathway
    >>> result = calculate_bifunctional_gap(delta_G_ORR, delta_G_OER)
    >>> result["overpotential_gap"]
    2.40  # V (1.20 + 1.20)
    >>> result["voltage_window"]
    2.40  # V (2.43 - 0.03)
    """
    orr_result = calculate_orr_overpotential(delta_G_ORR)
    oer_result = calculate_oer_overpotential(delta_G_OER)

    eta_ORR = orr_result["eta_ORR"]  # noqa: N806
    eta_OER = oer_result["eta_OER"]  # noqa: N806
    gap = eta_ORR + eta_OER

    U_ORR_onset = orr_result["U_onset"]  # noqa: N806
    U_OER_onset = oer_result["U_onset"]  # noqa: N806
    voltage_window = U_OER_onset - U_ORR_onset

    logger.info(
        f"Bifunctional performance: "
        f"η_ORR = {eta_ORR:.3f} V, η_OER = {eta_OER:.3f} V, "
        f"Gap = {gap:.3f} V"
    )

    return {
        "eta_ORR": eta_ORR,
        "eta_OER": eta_OER,
        "overpotential_gap": gap,
        "U_ORR_onset": U_ORR_onset,
        "U_OER_onset": U_OER_onset,
        "voltage_window": voltage_window,
    }


def calculate_her_overpotential(
    delta_G_H: float,  # noqa: N803
) -> dict[str, float]:
    """
    Calculate HER overpotential from hydrogen binding energy.

    HER (Hydrogen Evolution Reaction):
        2H⁺ + 2e⁻ → H₂  (in acidic media)
        2H₂O + 2e⁻ → H₂ + 2OH⁻  (in alkaline media)

    The Sabatier principle states that optimal HER catalysts have
    thermoneutral H binding: ΔG_H* ≈ 0 eV.

    Parameters
    ----------
    delta_G_H : float
        Free energy of H adsorption (eV).
        ΔG_H = G(H*) - G(*) - ½G(H₂)

    Returns
    -------
    dict
        {
            'eta_HER': float,           # HER overpotential (V)
            'delta_G_H': float,         # H binding energy (eV)
            'U_equilibrium': float,     # Equilibrium potential (V, always 0)
        }

    Notes
    -----
    **HER overpotential:**

    The overpotential is determined by the deviation from thermoneutral binding:
        η_HER ≈ |ΔG_H|

    - ΔG_H > 0: H binds too weakly (rate-limited by H adsorption)
    - ΔG_H < 0: H binds too strongly (rate-limited by H₂ desorption)
    - ΔG_H ≈ 0: Optimal (Pt-like behavior)

    **Volcano plot:**
        - Peak activity at ΔG_H = 0 (Pt, Pd)
        - Weak binding: Au, Ag, Cu (ΔG_H > 0.2 eV)
        - Strong binding: W, Mo (ΔG_H < -0.2 eV)

    References
    ----------
    - Nørskov et al., J. Electrochem. Soc. 152, J23 (2005): HER volcano plot
    - Greeley et al., Nat. Mater. 5, 909 (2006): Computational screening

    Examples
    --------
    >>> # Pt-like catalyst (nearly ideal)
    >>> result = calculate_her_overpotential(delta_G_H=0.05)
    >>> result["eta_HER"]
    0.05  # V (very small overpotential)

    >>> # Weak binding (Au-like)
    >>> result = calculate_her_overpotential(delta_G_H=0.50)
    >>> result["eta_HER"]
    0.50  # V

    >>> # Strong binding (W-like)
    >>> result = calculate_her_overpotential(delta_G_H=-0.60)
    >>> result["eta_HER"]
    0.60  # V
    """
    # HER overpotential is |ΔG_H|
    eta_HER = abs(delta_G_H)  # noqa: N806

    logger.info(f"HER overpotential: η = {eta_HER:.3f} V (ΔG_H = {delta_G_H:+.3f} eV)")

    return {
        "eta_HER": eta_HER,
        "delta_G_H": delta_G_H,
        "U_equilibrium": U_EQ_HER,
    }
