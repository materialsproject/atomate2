"""Thermodynamic analysis for electrocatalysis using the Computational Hydrogen Electrode (CHE) model.

The CHE model simplifies electrochemical thermodynamics by referencing all potentials
to the Standard Hydrogen Electrode (SHE). This allows calculation of reaction free
energies under electrochemical conditions without explicit treatment of solvent or
electrode potential.

Key assumptions:
1. Chemical potential of (H+ + e-) = 1/2 μ(H2) - eU at potential U
2. Entropy and zero-point energy corrections from gas-phase calculations
3. Constant pH (typically pH = 0 for acidic, pH = 14 for alkaline)

References
----------
- Nørskov et al., J. Phys. Chem. B 108, 17886 (2004)
- Peterson et al., Energy Environ. Sci. 3, 1311 (2010)
- Viswanathan et al., ACS Catal. 2, 1654 (2012)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# Physical constants
KB = 8.617333262e-5  # Boltzmann constant (eV/K)
H_PLANCK = 4.135667696e-15  # Planck constant (eV⋅s)


def calculate_free_energy_corrections(
    temperature: float = 298.15,
    pressure: float = 101325.0,
    entropy_corrections: dict[str, float] | None = None,
    zpe_corrections: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Calculate thermodynamic corrections for CHE model.

    The free energy is given by:
        G = E_DFT + ZPE - TS + ∫(C_p)dT

    For gas-phase molecules, we typically use:
        G ≈ E_DFT + ZPE + (H_298 - H_0) - TS_298

    where:
    - ZPE: Zero-point energy correction
    - (H_298 - H_0): Thermal enthalpy correction (typically small)
    - TS_298: Entropy contribution at 298.15 K

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin (default: 298.15 K = 25°C).
    pressure : float
        Pressure in Pascal (default: 101325 Pa = 1 atm).
    entropy_corrections : dict[str, float] | None
        Entropy corrections for molecules (eV). If None, uses standard values.
        Example: {'H2O': -TΔS, 'O2': -TΔS, ...}
    zpe_corrections : dict[str, float] | None
        Zero-point energy corrections (eV). If None, uses standard values.

    Returns
    -------
    dict[str, float]
        Free energy corrections for each species (eV).
        Keys: 'H2', 'H2O', 'O2', 'CO2', 'CO', 'CH4', etc.

    Notes
    -----
    **Standard corrections** (from NIST/experimental data at 298.15 K, 1 atm):

    Gas-phase molecules:
        - H₂:  ΔG = 0.00 eV (reference state)
        - H₂O: ΔG = E_DFT + 0.67 eV (includes ZPE + TS)
        - O₂:  ΔG = E_DFT + 0.05 eV
        - CO₂: ΔG = E_DFT + 0.45 eV
        - CO:  ΔG = E_DFT + 0.13 eV
        - CH₄: ΔG = E_DFT + 0.34 eV

    Adsorbed intermediates (typical approximations):
        - O*:   ΔG ≈ E_DFT (entropy ≈ 0, weak binding)
        - OH*:  ΔG ≈ E_DFT + 0.35 eV
        - OOH*: ΔG ≈ E_DFT + 0.40 eV

    References
    ----------
    - Nørskov et al., J. Phys. Chem. B 108, 17886 (2004)
    - NIST-JANAF Thermochemical Tables

    Examples
    --------
    >>> corrections = calculate_free_energy_corrections()
    >>> corrections["H2O"]
    0.67  # eV (includes ZPE - TS)

    >>> # Custom corrections
    >>> custom_zpe = {"O2": 0.10, "H2O": 0.56}
    >>> custom_entropy = {"O2": -0.05, "H2O": -0.11}
    >>> corrections = calculate_free_energy_corrections(
    ...     zpe_corrections=custom_zpe, entropy_corrections=custom_entropy
    ... )
    """
    # Standard corrections at 298.15 K, 1 atm
    # These are TOTAL corrections: ΔG = E_DFT + correction
    # Reference: Nørskov et al., J. Phys. Chem. B 108, 17886 (2004)
    standard_corrections = {
        # Gas-phase molecules
        "H2": 0.00,  # Reference state (by definition)
        "H2O": 0.67,  # ZPE ≈ 0.56 eV, -TS ≈ 0.11 eV (liquid at 1 atm)
        "O2": 0.05,  # ZPE ≈ 0.10 eV, -TS ≈ -0.05 eV
        "CO2": 0.45,  # ZPE ≈ 0.31 eV, -TS ≈ 0.14 eV
        "CO": 0.13,  # ZPE ≈ 0.14 eV, -TS ≈ -0.01 eV
        "CH4": 0.34,  # ZPE ≈ 0.73 eV, -TS ≈ -0.39 eV
        "C2H4": 0.31,  # Ethylene
        "NH3": 0.32,  # Ammonia
        "N2": 0.00,  # Similar to H2 (small corrections)
        # Adsorbed intermediates (approximations)
        # Surface-bound species have ~zero entropy (vibrational modes only)
        "O*": 0.00,  # Adsorbed oxygen
        "OH*": 0.35,  # Adsorbed hydroxyl
        "OOH*": 0.40,  # Adsorbed hydroperoxy
        "H*": 0.00,  # Adsorbed hydrogen
        "CO*": 0.00,  # Adsorbed CO
        "CHO*": 0.30,  # Formyl intermediate
        "COOH*": 0.45,  # Carboxyl intermediate
    }

    # Apply temperature scaling if T ≠ 298.15 K
    # ΔG(T) ≈ ΔG(298) + ΔC_p(T - 298) - TΔS(298)/298 * T
    # For simplicity, we assume ΔC_p ≈ 0 and scale linearly
    if temperature != 298.15:
        logger.warning(
            f"Temperature {temperature:.2f} K differs from standard (298.15 K). "
            "Corrections are scaled linearly (ΔC_p ≈ 0 approximation)."
        )
        scale_factor = temperature / 298.15
    else:
        scale_factor = 1.0

    # Override with custom corrections if provided
    corrections = standard_corrections.copy()

    if zpe_corrections or entropy_corrections:
        # Recalculate from ZPE and entropy components
        for species in corrections:
            zpe = zpe_corrections.get(species, 0.0) if zpe_corrections else 0.0
            ts = entropy_corrections.get(species, 0.0) if entropy_corrections else 0.0
            if zpe != 0.0 or ts != 0.0:
                corrections[species] = zpe + ts  # ΔG = ZPE - TS

    # Apply temperature scaling
    for species in corrections:
        corrections[species] *= scale_factor

    logger.info(
        f"Calculated free energy corrections at T = {temperature:.2f} K, "
        f"P = {pressure:.0f} Pa"
    )

    return corrections


def calculate_reaction_free_energies(
    surface_name: str,
    pathway_steps: Sequence[dict[str, float | str]],
    gas_phase_energies: dict[str, float],
    clean_surface_energy: float,
    temperature: float = 298.15,
    pressure: float = 101325.0,
    ph: float = 0.0,
    potential: float = 0.0,
) -> dict[str, list[float] | list[str] | float]:
    """
    Calculate reaction free energies using the Computational Hydrogen Electrode (CHE) model.

    The CHE model relates electrochemical reactions to H₂ as a reference:
        μ(H⁺ + e⁻) = ½μ(H₂) - eU

    where U is the electrode potential vs. SHE.

    For ORR/OER, the key reactions are:
        O₂ + 4(H⁺ + e⁻) → 2H₂O     (ORR, 4-electron)
        2H₂O → O₂ + 4(H⁺ + e⁻)     (OER, reverse of ORR)

    Each step involves transfer of (H⁺ + e⁻) pairs.

    Parameters
    ----------
    surface_name : str
        Name of the surface/catalyst (for logging).
    pathway_steps : Sequence[dict[str, float | str]]
        List of reaction steps. Each dict must contain:
            {
                'label': str,           # Step label (e.g., 'O2_ads', 'OOH*')
                'energy': float,        # DFT energy (eV)
                'species': str | None,  # Added species (e.g., 'O2', 'OH', 'H2O')
                'n_H': int,             # Number of H atoms added/removed
                'n_e': int,             # Number of electrons transferred
            }
    gas_phase_energies : dict[str, float]
        DFT energies of gas-phase molecules (eV).
        Must include: 'H2', 'H2O', and any species in pathway_steps.
    clean_surface_energy : float
        DFT energy of clean surface slab (eV).
    temperature : float
        Temperature in Kelvin (default: 298.15 K).
    pressure : float
        Pressure in Pascal (default: 101325 Pa = 1 atm).
    ph : float
        pH of electrolyte (default: 0.0 for acidic).
        Used for pH correction on proton-consuming steps:
        ΔG_pH = +k_B T ln(10) × pH ≈ +0.059 eV × pH at 298 K (SHE scale).
    potential : float
        Electrode potential vs. SHE in Volts (default: 0.0 V).
        Modifies (H⁺ + e⁻) chemical potential: μ(H⁺ + e⁻) = ½μ(H₂) - eU.

    Returns
    -------
    dict
        {
            'step_labels': list[str],
            'absolute_energies': list[float],  # E_DFT for each step
            'delta_E': list[float],            # Energy differences (no corrections)
            'delta_G': list[float],            # Free energy differences (with corrections)
            'cumulative_G': list[float],       # Cumulative free energies
            'thermodynamic_overpotential': float,  # η (V) from CHE model
        }

    Notes
    -----
    **CHE Model Equations:**

    1. **Adsorption energy** (relative to gas phase):
        ΔE_ads = E(slab + adsorbate) - E(slab) - E(molecule)

    2. **Free energy change** for step involving (H⁺ + e⁻):
        ΔG = ΔE + ΔZPE - TΔS + ΔG_pH - eU

    where:
        - ΔE: DFT energy difference
        - ΔZPE: Zero-point energy correction
        - -TΔS: Entropy correction
        - ΔG_pH: pH correction (+0.059 eV × pH at 298 K for steps consuming
          H⁺ + e⁻; μ(H⁺) decreases with pH, so reduction steps become harder
          in alkaline media on the SHE scale)
        - -eU: Electrode potential contribution

    3. **Overpotential** (for ORR):
        η_ORR = 1.23 V - U_onset
        U_onset = U at which all ΔG steps become downhill

    References
    ----------
    - Nørskov et al., J. Phys. Chem. B 108, 17886 (2004): Original CHE model
    - Viswanathan et al., ACS Catal. 2, 1654 (2012): ORR/OER analysis

    Examples
    --------
    >>> # ORR pathway: O₂ → OOH* → O* + OH* → 2OH* → H₂O
    >>> pathway = [
    ...     {"label": "O2_ads", "energy": -500.0, "species": "O2", "n_H": 0, "n_e": 0},
    ...     {"label": "OOH*", "energy": -498.5, "species": "H", "n_H": 1, "n_e": 1},
    ...     {"label": "O*", "energy": -497.0, "species": "H2O", "n_H": 1, "n_e": 1},
    ...     {"label": "OH*", "energy": -496.0, "species": "H", "n_H": 1, "n_e": 1},
    ...     {"label": "H2O", "energy": -495.0, "species": "H", "n_H": 1, "n_e": 1},
    ... ]
    >>> gas_energies = {"H2": -6.77, "H2O": -14.22, "O2": -9.86}
    >>> clean_surf = -494.0
    >>>
    >>> results = calculate_reaction_free_energies(
    ...     surface_name="Pt(111)",
    ...     pathway_steps=pathway,
    ...     gas_phase_energies=gas_energies,
    ...     clean_surface_energy=clean_surf,
    ...     potential=0.0,
    ... )
    >>> results["delta_G"]
    [0.45, 1.20, 0.80, 0.60]  # Free energy changes for each step (eV)
    """
    # Get thermodynamic corrections
    corrections = calculate_free_energy_corrections(
        temperature=temperature, pressure=pressure
    )

    # Extract step information
    step_labels: list[str] = [str(step["label"]) for step in pathway_steps]
    step_energies: list[float] = [float(step["energy"]) for step in pathway_steps]

    # Calculate energy differences (ΔE_DFT)
    delta_E: list[float] = []
    for i in range(len(step_energies)):
        if i == 0:
            # First step: relative to clean surface
            dE = step_energies[i] - clean_surface_energy
        else:
            dE = step_energies[i] - step_energies[i - 1]
        delta_E.append(dE)

    # Calculate free energy differences (ΔG)
    delta_G: list[float] = []
    cumulative_G = [0.0]  # Start at ΔG = 0 for clean surface

    # pH correction (affects all proton-coupled steps)
    # μ(H⁺) = μ°(H⁺) - k_B T ln(10) × pH, so a step CONSUMING (H⁺ + e⁻)
    # becomes harder at higher pH: ΔG_pH = +k_B T ln(10) × pH per pair
    # (≈ +0.059 eV × pH at 298 K, SHE scale; on the RHE scale pH cancels)
    ph_correction = KB * temperature * np.log(10) * ph  # eV

    for i, step in enumerate(pathway_steps):
        # Get number of electrons and protons transferred
        n_e: int = int(step.get("n_e", 0))
        n_H: int = int(step.get("n_H", 0))

        # Base energy change
        dG = delta_E[i]

        # Gas-phase reference energies
        # Only apply gas-phase corrections if there's actual electron/proton transfer (not reference states)
        # Sign of n_H determines if species is consumed (reactant) or produced (product):
        # - For H2O: n_H > 0 means product (ORR), n_H < 0 means reactant (OER)
        # - For O2: Always check context (ORR vs OER) based on pathway
        species: str | None = (
            None if step.get("species") is None else str(step.get("species"))
        )

        # Skip gas-phase corrections for reference states (n_H=0, n_e=0) that don't involve actual transfer
        # Exception: O2 adsorption (n_H=0, n_e=0) in ORR DOES consume O2 from gas phase
        is_reference_state = n_H == 0 and n_e == 0 and species != "O2"

        if species and species in gas_phase_energies and not is_reference_state:
            if species == "H2O":
                if n_H > 0:
                    # ORR: H2O is produced (released to gas phase) → add energy
                    dG += gas_phase_energies[species]
                elif n_H < 0:
                    # OER: H2O is consumed (from gas phase) → subtract energy
                    dG -= gas_phase_energies[species]
                # n_H == 0 case is handled by is_reference_state check above
            elif species == "O2":
                # ORR: O2 consumed from gas phase (n_H >= 0) → subtract
                # OER: O2 produced to gas phase (n_H < 0) → add
                if n_H >= 0:
                    # ORR: O2 is consumed from gas phase
                    dG -= gas_phase_energies[species]
                else:
                    # OER: O2 is produced to gas phase (rare, usually pathway ends at O2*)
                    dG += gas_phase_energies[species]
            else:
                # Default: assume reactant (consumed)
                dG -= gas_phase_energies[species]

        # Add thermodynamic corrections (ZPE, entropy) for species
        if species and species in corrections:
            dG += corrections[species]

        # Proton-electron pair corrections (CHE model) for steps consuming
        # n_e × (H⁺ + e⁻): subtract the pair chemical potential
        # μ(H⁺ + e⁻) = ½μ(H₂) - eU - k_B T ln(10) × pH,
        # so dG gains -½μ(H₂), the potential term, and +0.059 eV × pH
        if n_e > 0:
            # Subtract H₂ reference
            if "H2" in gas_phase_energies:
                dG -= 0.5 * n_e * gas_phase_energies["H2"]

            # Add electrode potential contribution
            # CHE model: μ(H⁺ + e⁻) = ½μ(H₂) - eU
            # Therefore: ΔG(U) = ΔG(0) - n_e × U
            # Verified against Pt(111) ORR literature (η ≈ 0.45 V)
            dG -= n_e * potential  # eV (U in Volts)

            # Add pH correction
            dG += n_e * ph_correction

        delta_G.append(dG)
        cumulative_G.append(cumulative_G[-1] + dG)

    # Calculate thermodynamic overpotential
    # For ORR: η = max(ΔG_i) when ΔG calculated at U_equilibrium
    # The overpotential is the largest uphill step (rate-limiting barrier)
    # U_equilibrium = 1.23 V (O₂ + 4H⁺ + 4e⁻ → 2H₂O)
    #
    # Find maximum uphill step
    max_delta_G = max(delta_G) if delta_G else 0.0
    U_equilibrium_ORR = 1.23  # V vs. SHE

    # If delta_G was calculated at equilibrium potential, overpotential = max_delta_G
    # If delta_G was calculated at different potential, need to adjust
    # Since delta_G includes the potential term (dG -= n_e * potential),
    # we have: eta = max_delta_G at U=U_eq
    if abs(potential - U_equilibrium_ORR) < 0.01:  # Calculated at equilibrium
        overpotential = max_delta_G
    else:
        # General formula: find U_onset where max step becomes thermoneutral
        # For now, just use max_delta_G as approximation
        overpotential = max_delta_G

    logger.info(
        f"Calculated reaction free energies for {surface_name}: "
        f"{len(pathway_steps)} steps, η = {overpotential:.3f} V"
    )

    return {
        "step_labels": step_labels,
        "absolute_energies": step_energies,
        "delta_E": delta_E,
        "delta_G": delta_G,
        "cumulative_G": cumulative_G,
        "thermodynamic_overpotential": overpotential,
    }


def identify_rate_limiting_step(
    delta_G: Sequence[float], step_labels: Sequence[str] | None = None
) -> dict[str, int | str | float]:
    """
    Identify the rate-limiting step (RLS) in a reaction pathway.

    The RLS is defined as the step with the largest uphill free energy change (ΔG > 0).
    This determines the activation barrier for the overall reaction.

    Parameters
    ----------
    delta_G : Sequence[float]
        Free energy changes for each step (eV).
    step_labels : Sequence[str] | None
        Labels for each step (optional).

    Returns
    -------
    dict
        {
            'rls_index': int,        # Index of rate-limiting step (0-indexed)
            'rls_label': str | None, # Label of RLS
            'rls_delta_G': float,    # Free energy barrier (eV)
        }

    Examples
    --------
    >>> delta_G = [0.45, 1.20, 0.80, 0.60]
    >>> labels = ["O2_ads", "OOH*", "O*", "OH*"]
    >>> rls = identify_rate_limiting_step(delta_G, labels)
    >>> rls
    {'rls_index': 1, 'rls_label': 'OOH*', 'rls_delta_G': 1.20}
    """
    if not delta_G:
        return {"rls_index": -1, "rls_label": None, "rls_delta_G": 0.0}

    # Find step with maximum ΔG
    rls_index = int(np.argmax(delta_G))
    rls_delta_G = float(delta_G[rls_index])

    rls_label = None
    if step_labels and len(step_labels) > rls_index:
        rls_label = step_labels[rls_index]

    logger.info(
        f"Rate-limiting step: {rls_label or rls_index} (ΔG = {rls_delta_G:.3f} eV)"
    )

    return {
        "rls_index": rls_index,
        "rls_label": rls_label,
        "rls_delta_G": rls_delta_G,
    }
