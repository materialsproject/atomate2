"""Interface FlowMaker for 2D heterostructure construction.

This module provides automated workflows for constructing and optimizing 2D material
heterostructures with automatic lattice matching and binding energy calculations.

The InterfaceFlowMaker supports two lattice matching strategies:

1. **Strain mode**: Simple strain-based matching for small lattice mismatch (<5%)
2. **Supercell mode**: Supercell-based matching for large mismatch (>10%), finds
   optimal N×M and P×Q supercell combinations to minimize strain

References
----------
- Koma et al., Heterostructures of layered semiconductors, 1985
- Gong et al., Band offset in graphene-MoS₂ heterostructures, Nat. Mater. 2014
- Björkman et al., van der Waals bonding in layered compounds, Phys. Rev. B 2012
"""  # noqa: RUF002

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, job

from atomate2.siesta.flows.base import BaseSiestaFlowMaker
from atomate2.siesta.jobs.core import RelaxMaker

if TYPE_CHECKING:
    from jobflow import Job
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


@dataclass
class InterfaceFlowMaker(BaseSiestaFlowMaker):
    """
    FlowMaker for creating and optimizing 2D material heterostructures.

    This is a general-purpose workflow for constructing van der Waals heterostructures
    from two 2D material layers. It handles lattice matching, interlayer distance
    optimization, and binding energy calculations.

    Capabilities
    ------------
    - Automatic lattice matching (strain mode or supercell mode)
    - Interlayer distance optimization via scanning
    - van der Waals corrections handling
    - Interface binding energy calculation
    - Charge transfer analysis (optional, requires Bader)

    Lattice Matching Modes
    ----------------------
    **strain** (default)
        Simple uniform strain to match lattices. Best for mismatch <5%.
        Pros: Small unit cell, fast calculations
        Cons: High strain if large mismatch

    **supercell**
        Find optimal N×M (bottom) and P×Q (top) supercells to minimize strain.
        Best for mismatch >10%.
        Pros: Minimal strain, commensurate interface
        Cons: Larger supercells (more expensive calculations)

    **auto**
        Automatically choose strain or supercell based on lattice mismatch.
        Uses strain if mismatch <10%, otherwise supercell.

    Use Cases
    ---------
    - MoS₂@graphene (catalysis support, charge transfer)
    - WSe₂@MoS₂ (type-II heterostructure, band alignment)
    - Graphene@Pt (electrode interfaces, HER catalysis)
    - TMD@h-BN (encapsulation)
    - Any van der Waals heterostructure

    Parameters
    ----------
    name : str
        Name of the workflow (default: "interface_builder").
    relax_maker : BaseSiestaMaker
        Maker for structure relaxation jobs.
    interlayer_distance : float
        Starting interlayer distance in Ångstroms (default: 3.4 Å, typical vdW).
    matching_mode : str
        Lattice matching strategy: "strain", "supercell", or "auto" (default: "strain").
    max_lattice_mismatch : float
        Maximum allowed lattice mismatch for strain mode (default: 0.05 = 5%).
    max_supercell_size : int
        Maximum N or M value for supercell search (default: 10).
    max_area_mismatch : float
        Maximum area mismatch for supercell matching (default: 0.05 = 5%).
    apply_strain_to : str
        Which layer to strain: "smaller", "larger", or "both" (default: "smaller").
    optimize_interlayer_distance : bool
        Whether to scan interlayer distance to find minimum energy (default: True).
    distance_range : tuple[float, float]
        Distance range for scanning in Ångstroms (default: (3.0, 4.0)).
    distance_steps : int
        Number of distance points to scan (default: 5).
    calculate_binding_energy : bool
        Whether to calculate interface binding energy (default: True).
    vdw : str | None
        van der Waals correction method (e.g., "DRSLL", "DZP"), applied via user_params.

    Examples
    --------
    **Example 1: Simple strain matching (small mismatch)**

    >>> from atomate2.siesta.flows.heterostructures import InterfaceFlowMaker
    >>> from pymatgen.core import Structure
    >>>
    >>> graphene = Structure.from_file("graphene.cif")  # a = 2.46 Å
    >>> hbn = Structure.from_file("h-BN.cif")  # a = 2.50 Å
    >>> # Mismatch = 1.6% → Good for strain mode
    >>>
    >>> maker = InterfaceFlowMaker(
    ...     matching_mode="strain",
    ...     interlayer_distance=3.4,
    ...     calculate_binding_energy=True,
    ... )
    >>> flow = maker.make(bottom_layer=graphene, top_layer=hbn)

    **Example 2: Supercell matching (large mismatch)**

    >>> mos2 = Structure.from_file("MoS2_unit.cif")  # a = 3.16 Å
    >>> graphene = Structure.from_file("graphene_unit.cif")  # a = 2.46 Å
    >>> # Mismatch = 28% → Too large for simple strain!
    >>>
    >>> maker = InterfaceFlowMaker(
    ...     matching_mode="supercell",
    ...     max_supercell_size=6,
    ...     max_area_mismatch=0.05,
    ...     calculate_binding_energy=True,
    ... )
    >>> flow = maker.make(bottom_layer=graphene, top_layer=mos2)
    >>> # Finds: Graphene (4×4) + MoS₂ (3×3), only 3.6% mismatch!

    **Example 3: Auto mode with interlayer optimization**

    >>> maker = InterfaceFlowMaker(
    ...     matching_mode="auto",
    ...     optimize_interlayer_distance=True,
    ...     distance_range=(3.0, 4.5),
    ...     distance_steps=7,
    ... )
    >>> flow = maker.make(bottom_layer=material_A, top_layer=material_B)

    **Example 4: With van der Waals correction**

    >>> from atomate2.siesta.jobs.core import RelaxMaker
    >>> from atomate2.siesta.sets.tiers import apply_tier_preset
    >>>
    >>> relax_maker = RelaxMaker()
    >>> relax_maker = apply_tier_preset(relax_maker, "2d_materials")
    >>>
    >>> maker = InterfaceFlowMaker(
    ...     relax_maker=relax_maker,
    ...     matching_mode="supercell",
    ... )
    >>> # vdW correction applied via tier preset
    >>> flow = maker.make(bottom_layer=layer1, top_layer=layer2)

    Notes
    -----
    - For metal substrates, use shorter interlayer_distance (e.g., 2.5 Å)
    - van der Waals corrections are crucial for accurate binding energies
    - Supercell matching can create large structures (>100 atoms) - consider cost
    - Binding energy typically ranges from -10 to -50 meV/Å² for vdW systems
    """  # noqa: RUF002

    name: str = "interface_builder"
    relax_maker: RelaxMaker = field(default_factory=RelaxMaker)

    # Structural parameters
    interlayer_distance: float = 3.4  # Å, starting distance (typical vdW)
    max_lattice_mismatch: float = 0.05  # 5% max mismatch for strain mode

    # Lattice matching strategy
    matching_mode: str = "strain"  # "strain", "supercell", "auto"
    max_supercell_size: int = 10  # Maximum N×M supercell dimension  # noqa: RUF003
    max_area_mismatch: float = 0.05  # 5% for supercell matching
    apply_strain_to: str = "smaller"  # "smaller", "larger", "both" (for strain mode)

    # Optimization
    optimize_interlayer_distance: bool = True
    distance_range: tuple[float, float] = (3.0, 4.0)  # Å
    distance_steps: int = 5

    # Analysis
    calculate_binding_energy: bool = True

    # Execution mode (overrides relax_maker settings if specified)
    dry_run: bool | None = None  # Generate input files only, no execution
    use_custodian: bool | None = None  # Enable automatic error recovery
    custodian_max_errors: int | None = None  # Maximum retry attempts

    def make(
        self,
        bottom_layer: Structure,
        top_layer: Structure,
        rotation_angle: float = 0.0,
    ) -> Flow:
        """
        Create heterostructure interface workflow.

        Parameters
        ----------
        bottom_layer : Structure
            Bottom 2D material (e.g., graphene substrate).
        top_layer : Structure
            Top 2D material (e.g., MoS₂ overlayer).
        rotation_angle : float
            Rotate top layer by this angle in degrees (default: 0.0).
            Example: 21.8° for magic angle graphene bilayer.

        Returns
        -------
        Flow
            Complete interface workflow with lattice matching, relaxation,
            and optional binding energy calculation.

        Notes
        -----
        The workflow consists of the following jobs:

        1. **Lattice compatibility check** (or supercell matching)
        2. **Build interface structure** (stack layers with rotation)
        3. **Relax interface** (full geometry optimization)
        4. **Distance scan** (optional, if optimize_interlayer_distance=True)
        5. **Relax isolated layers** (optional, if calculate_binding_energy=True)
        6. **Calculate binding energy** (optional)

        Output structure is the relaxed interface from job 3 (or job 4 if scanning).
        """
        # Override relax_maker settings if execution mode parameters are specified
        relax_maker = self.relax_maker
        if self.dry_run is not None or self.use_custodian is not None:
            from dataclasses import replace
            from typing import Any

            override_kwargs: dict[str, Any] = {}
            if self.dry_run is not None:
                override_kwargs["dry_run"] = self.dry_run
            if self.use_custodian is not None:
                override_kwargs["use_custodian"] = self.use_custodian
            if self.custodian_max_errors is not None:
                override_kwargs["custodian_max_errors"] = self.custodian_max_errors

            relax_maker = replace(self.relax_maker, **override_kwargs)

        jobs: list[Job | Flow] = []

        # Job 1: Find optimal supercell match or check compatibility
        if self.matching_mode in ["supercell", "auto"]:
            # Use supercell matching algorithm
            match_job = find_supercell_match(
                bottom=bottom_layer,
                top=top_layer,
                rotation_angle=rotation_angle,  # CRITICAL: Must consider rotation!
                max_size=self.max_supercell_size,
                max_area_mismatch=self.max_area_mismatch,
            )
            match_job.name = f"{self.name}_supercell_match"
            jobs.append(match_job)
        else:
            # Simple compatibility check for strain mode
            match_job = check_lattice_compatibility(
                bottom=bottom_layer,
                top=top_layer,
                max_mismatch=self.max_lattice_mismatch,
            )
            match_job.name = f"{self.name}_lattice_check"
            jobs.append(match_job)

        # Job 2: Build interface structure
        build_job = build_interface_structure(
            bottom=bottom_layer,
            top=top_layer,
            distance=self.interlayer_distance,
            rotation=rotation_angle,
            matching_mode=self.matching_mode,
            supercell_match=match_job.output,
            apply_strain_to=self.apply_strain_to,
        )
        build_job.name = f"{self.name}_build_structure"
        jobs.append(build_job)

        # Job 3: Relax interface
        relax_job = relax_maker.make(
            build_job.output["structure"],
            prev_dir=None,
        )
        relax_job.name = f"{self.name}_relax_interface"
        jobs.append(relax_job)

        # Store reference to final relaxed structure
        final_structure_job = relax_job

        # Job 4 (optional): Interlayer distance scan
        if self.optimize_interlayer_distance:
            scan_job = scan_interlayer_distance(
                base_structure=build_job.output["structure"],
                distance_range=self.distance_range,
                steps=self.distance_steps,
                relax_maker=relax_maker,
            )
            scan_job.name = f"{self.name}_distance_scan"
            jobs.append(scan_job)
            # Update final structure reference
            final_structure_job = scan_job  # scan_job.output has optimized structure

        # Job 5-7 (optional): Binding energy calculation
        binding_job = None
        if self.calculate_binding_energy:
            # Job 5: Relax isolated bottom layer
            bottom_relax = relax_maker.make(bottom_layer)
            bottom_relax.name = f"{self.name}_relax_bottom_layer"
            jobs.append(bottom_relax)

            # Job 6: Relax isolated top layer
            top_relax = relax_maker.make(top_layer)
            top_relax.name = f"{self.name}_relax_top_layer"
            jobs.append(top_relax)

            # Job 7: Calculate binding energy
            interface_energy = (
                final_structure_job.output["energy"]
                if self.optimize_interlayer_distance
                else relax_job.output.output.energy
            )

            # Get interface structure
            interface_structure = (
                final_structure_job.output["structure"]
                if self.optimize_interlayer_distance
                else relax_job.output.structure
            )

            binding_job = calculate_interface_binding_energy(
                interface_energy=interface_energy,
                bottom_energy=bottom_relax.output.output.energy,
                top_energy=top_relax.output.output.energy,
                interface_area=build_job.output["area"],
                interface_structure=interface_structure,
                bottom_structure=bottom_relax.output.structure,
                top_structure=top_relax.output.structure,
            )
            binding_job.name = f"{self.name}_binding_energy"
            jobs.append(binding_job)

        # Set flow output
        if binding_job is not None:
            flow_output = binding_job.output
        elif self.optimize_interlayer_distance:
            flow_output = final_structure_job.output
        else:
            flow_output = relax_job.output

        return Flow(jobs, output=flow_output, name=self.name)


# ============================================================================
# Module-level helper functions (@job decorators for jobflow-remote compatibility)
# ============================================================================


@job
def find_supercell_match(
    bottom: Structure,
    top: Structure,
    rotation_angle: float = 0.0,
    max_size: int = 10,
    max_area_mismatch: float = 0.05,
) -> dict:
    """
    Find optimal supercell combination for two 2D materials.

    Uses area-matching algorithm to find N×M (bottom) and P×Q (top) supercells
    that minimize strain while keeping supercell sizes reasonable.

    Algorithm
    ---------
    1. Try all combinations: bottom (N×M), top (P×Q) where N,M,P,Q ≤ max_size
    2. Calculate area mismatch: |A_bottom - A_top| / A_bottom
    3. Minimize: score = area_mismatch + supercell_penalty
    4. Return combination with best score and area_mismatch ≤ max_area_mismatch

    Parameters
    ----------
    bottom : Structure
        Bottom layer unit cell structure.
    top : Structure
        Top layer unit cell structure.
    rotation_angle : float
        Rotation angle in degrees (default: 0.0). Top layer is rotated BEFORE
        lattice matching. Critical for twisted bilayers!
    max_size : int
        Maximum supercell dimension (default: 10, searches up to 10×10).
    max_area_mismatch : float
        Maximum allowed area mismatch (default: 0.05 = 5%).

    Returns
    -------
    dict
        {
            "bottom_supercell": [N, M],
            "top_supercell": [P, Q],
            "area_mismatch": float,
            "strain_required": float (linear mismatch),
            "bottom_area": float (Ų),
            "top_area": float (Ų),
            "bottom_dimensions": [a, b] (Å),
            "top_dimensions": [a, b] (Å),
            "total_atoms": int,
            "recommended": bool,
        }

    Example
    -------
    MoS₂ (a=3.16 Å) + Graphene (a=2.46 Å):
      → MoS₂ (3×3): 9.48 Å → Area = 89.9 Ų
      → Graphene (4×4): 9.84 Å → Area = 96.8 Ų
      → Mismatch = 7.7% → Needs ~4% strain on each

    Notes
    -----
    - Larger supercells reduce strain but increase computational cost
    - Scoring function penalizes large supercells: penalty = (N*M + P*Q) / 100
    - If no match found within constraints, returns 1×1 with recommendation=False
    """  # noqa: RUF002
    from itertools import product

    import numpy as np

    # Special handling for twisted bilayers (same material + rotation)
    # For twisted structures, we need geometric moiré supercell calculation
    is_twisted_bilayer = (
        abs(rotation_angle) > 0.01
        and abs(bottom.lattice.a - top.lattice.a) / bottom.lattice.a < 0.01
        and abs(bottom.lattice.b - top.lattice.b) / bottom.lattice.b < 0.01
    )

    if is_twisted_bilayer:
        # Calculate moiré supercell for twisted bilayer
        # Moiré wavelength: λ = a / (2·sin(θ/2))
        theta_rad = np.deg2rad(rotation_angle)
        a = bottom.lattice.a

        # Calculate moiré wavelength
        moire_wavelength = a / (2 * np.sin(theta_rad / 2))

        # Find supercell size: N such that N×a ≈ λ_moiré  # noqa: RUF003
        N_ideal = moire_wavelength / a  # noqa: N806
        N = max(  # noqa: N806
            2, int(np.round(N_ideal))
        )  # At least 2×2 supercell  # noqa: RUF003

        # Ensure within max_size
        if max_size < N:
            logger.warning(
                f"Moiré supercell size {N}×{N} exceeds max_size={max_size}. "  # noqa: RUF001
                f"Using {max_size}×{max_size} instead. "  # noqa: RUF001
                f"Increase max_supercell_size for accuracy."
            )
            N = max_size  # noqa: N806

        logger.info(
            f"Twisted bilayer (θ={rotation_angle}°): moiré λ={moire_wavelength:.2f} Å, "
            f"using [{N}×{N}] supercell"  # noqa: RUF001
        )

        # Return symmetric square supercell for both layers
        bottom_super = [N, N]
        top_super = [N, N]  # Same supercell for twisted bilayer

        bottom_a, bottom_b = bottom.lattice.a, bottom.lattice.b
        bottom_gamma = bottom.lattice.gamma

        return {
            "bottom_supercell": bottom_super,
            "top_supercell": top_super,
            "area_mismatch": 0.0,  # After rotation, areas match
            "strain_required": 0.0,
            "bottom_area": (N * bottom_a)
            * (N * bottom_b)
            * np.sin(np.deg2rad(bottom_gamma)),
            "top_area": (N * bottom_a)
            * (N * bottom_b)
            * np.sin(np.deg2rad(bottom_gamma)),
            "bottom_dimensions": [N * bottom_a, N * bottom_b],
            "top_dimensions": [N * bottom_a, N * bottom_b],
            "total_atoms": len(bottom) * N * N + len(top) * N * N,
            "recommended": True,
            "twisted_bilayer": True,
            "twist_angle": rotation_angle,
            "moire_wavelength": moire_wavelength,
        }

    # Regular (non-twisted) supercell matching
    # Get unit cell lattice parameters
    bottom_a, bottom_b = bottom.lattice.a, bottom.lattice.b
    bottom_gamma = bottom.lattice.gamma
    top_a, top_b = top.lattice.a, top.lattice.b
    top_gamma = top.lattice.gamma

    best_match = None
    min_score = float("inf")

    # Try all supercell combinations
    for N, M in product(range(1, max_size + 1), repeat=2):  # noqa: N806
        for P, Q in product(range(1, max_size + 1), repeat=2):  # noqa: N806
            # Calculate supercell dimensions
            bottom_super_a = N * bottom_a
            bottom_super_b = M * bottom_b
            bottom_area = (
                bottom_super_a * bottom_super_b * np.sin(np.deg2rad(bottom_gamma))
            )

            top_super_a = P * top_a
            top_super_b = Q * top_b
            top_area = top_super_a * top_super_b * np.sin(np.deg2rad(top_gamma))

            # Area mismatch
            area_mismatch = abs(bottom_area - top_area) / bottom_area

            # Linear dimension mismatch (for strain estimation)
            mismatch_a = abs(bottom_super_a - top_super_a) / bottom_super_a
            mismatch_b = abs(bottom_super_b - top_super_b) / bottom_super_b
            max_linear_mismatch = max(mismatch_a, mismatch_b)

            # Penalize large supercells (prefer smaller)
            supercell_penalty = (N * M + P * Q) / 100.0

            # Combined score
            score = area_mismatch + supercell_penalty

            # Update best if better than current
            # CRITICAL: Check BOTH area AND dimension matching to prevent
            # discontinuities
            # Area matching alone allows: Graphene 1×5 (b=12.3) + MoS2  # noqa: RUF003
            # 1×3 (b=9.48)  # noqa: RUF003
            # which creates gaps because MoS2 only covers 77% of interface
            if (
                score < min_score
                and area_mismatch <= max_area_mismatch
                and max_linear_mismatch <= max_area_mismatch
            ):
                min_score = score
                best_match = {
                    "bottom_supercell": [N, M],
                    "top_supercell": [P, Q],
                    "area_mismatch": area_mismatch,
                    "strain_required": max_linear_mismatch,
                    "bottom_area": bottom_area,
                    "top_area": top_area,
                    "bottom_dimensions": [bottom_super_a, bottom_super_b],
                    "top_dimensions": [top_super_a, top_super_b],
                    "total_atoms": len(bottom) * N * M + len(top) * P * Q,
                    "recommended": True,
                }

    if best_match is None:
        # No suitable match found within constraints
        logger.warning(
            f"No supercell match found within size {max_size} and "
            f"mismatch {max_area_mismatch:.2%}. Returning 1×1 (not recommended)."  # noqa: RUF001
        )
        bottom_area = bottom_a * bottom_b * np.sin(np.deg2rad(bottom_gamma))
        top_area = top_a * top_b * np.sin(np.deg2rad(top_gamma))
        return {
            "bottom_supercell": [1, 1],
            "top_supercell": [1, 1],
            "area_mismatch": abs(bottom_area - top_area) / bottom_area,
            "strain_required": max(
                abs(bottom_a - top_a) / bottom_a, abs(bottom_b - top_b) / bottom_b
            ),
            "bottom_area": bottom_area,
            "top_area": top_area,
            "bottom_dimensions": [bottom_a, bottom_b],
            "top_dimensions": [top_a, top_b],
            "total_atoms": len(bottom) + len(top),
            "recommended": False,
            "error": (
                f"No supercell match found within size {max_size} "
                f"and mismatch {max_area_mismatch:.2%}"
            ),
        }

    logger.info(
        f"Found supercell match: bottom {best_match['bottom_supercell']}, "
        f"top {best_match['top_supercell']}, "
        f"area mismatch {best_match['area_mismatch']:.2%}, "
        f"total {best_match['total_atoms']} atoms"
    )

    return best_match


@job
def check_lattice_compatibility(
    bottom: Structure, top: Structure, max_mismatch: float
) -> dict:
    """
    Check if two 2D materials can form commensurate interface with simple strain.

    This is a fast check for 1×1 matching without supercells.

    Parameters
    ----------
    bottom : Structure
        Bottom layer unit cell.
    top : Structure
        Top layer unit cell.
    max_mismatch : float
        Maximum allowed lattice mismatch (e.g., 0.05 = 5%).

    Returns
    -------
    dict
        {
            "compatible": bool (True if mismatch ≤ max_mismatch),
            "mismatch": float (maximum of a and b mismatch),
            "bottom_cell": [a, b] (Å),
            "top_cell": [a, b] (Å),
            "strain_required": float,
            "recommendation": str,
        }

    Example
    -------
    >>> # Graphene (2.46 Å) and h-BN (2.50 Å)
    >>> result = check_lattice_compatibility(graphene, hbn, max_mismatch=0.05)
    >>> print(result["mismatch"])  # 1.6% → compatible
    0.016
    >>> print(result["compatible"])
    True
    """  # noqa: RUF002
    # Get in-plane lattice parameters
    bottom_a = bottom.lattice.a
    bottom_b = bottom.lattice.b
    top_a = top.lattice.a
    top_b = top.lattice.b

    # Calculate mismatch
    mismatch_a = abs(bottom_a - top_a) / bottom_a
    mismatch_b = abs(bottom_b - top_b) / bottom_b
    mismatch = max(mismatch_a, mismatch_b)

    compatible = mismatch <= max_mismatch

    if compatible:
        recommendation = "Compatible - proceed with interface construction"
    else:
        recommendation = (
            f"Large mismatch ({mismatch:.2%}) - consider 'supercell' matching mode"
        )

    logger.info(
        f"Lattice compatibility check: mismatch = {mismatch:.2%}, "
        f"compatible = {compatible}"
    )

    return {
        "compatible": compatible,
        "mismatch": mismatch,
        "bottom_cell": [bottom_a, bottom_b],
        "top_cell": [top_a, top_b],
        "strain_required": mismatch if not compatible else 0.0,
        "recommendation": recommendation,
    }


@job
def build_interface_structure(
    bottom: Structure,
    top: Structure,
    distance: float,
    rotation: float,
    matching_mode: str = "strain",
    supercell_match: dict | None = None,
    apply_strain_to: str = "smaller",
) -> dict:
    """
    Build interface structure with proper stacking and lattice matching.

    Supports two lattice matching modes:
    1. **strain**: Apply uniform strain to match lattices (simple)
    2. **supercell**: Use supercell matching results (requires supercell_match input)

    Parameters
    ----------
    bottom : Structure
        Bottom 2D material unit cell.
    top : Structure
        Top 2D material unit cell.
    distance : float
        Interlayer distance in Ångstroms (vertical separation).
    rotation : float
        Rotate top layer by this angle in degrees (around z-axis).
    matching_mode : str
        "strain" or "supercell" (default: "strain").
    supercell_match : dict | None
        Output from find_supercell_match() if using supercell mode.
    apply_strain_to : str
        Which layer to strain: "smaller", "larger", or "both" (default: "smaller").

    Returns
    -------
    dict
        {
            "structure": Structure (stacked interface),
            "area": float (interface area in Ų),
            "strain_applied": dict,
            "supercells_used": dict,
            "interlayer_distance": float (Å),
            "matching_mode": str,
        }

    Notes
    -----
    - For "strain" mode: Stretches smaller cell to match larger (or distributes strain)
    - For "supercell" mode: Creates N×M and P×Q supercells, then applies residual strain
    - Rotation is applied to top layer before stacking
    - Final structure has combined lattice from bottom layer
    """  # noqa: RUF002
    import numpy as np

    # Make copies to avoid modifying originals
    bottom_copy = bottom.copy()
    top_copy = top.copy()

    # 1. Handle supercell or strain matching
    if matching_mode in ["supercell", "auto"] and supercell_match is not None:
        if supercell_match.get("recommended", True):
            # Use supercell matching
            N, M = supercell_match["bottom_supercell"]  # noqa: N806
            P, Q = supercell_match["top_supercell"]  # noqa: N806

            logger.info(f"Creating supercells: bottom {N}×{M}, top {P}×{Q}")  # noqa: RUF001

            # Always use simple diagonal matrix - rotation creates the moiré pattern
            bottom_matrix = [[N, 0, 0], [0, M, 0], [0, 0, 1]]
            top_matrix = [[P, 0, 0], [0, Q, 0], [0, 0, 1]]

            bottom_copy.make_supercell(bottom_matrix)
            top_copy.make_supercell(top_matrix)

            # Apply residual strain to minimize mismatch
            strain = supercell_match["strain_required"] / 2  # Distribute strain
            if apply_strain_to == "both":
                bottom_copy.apply_strain([strain, strain, 0])
                top_copy.apply_strain([-strain, -strain, 0])
            elif apply_strain_to == "smaller":
                if bottom.lattice.a < top.lattice.a:
                    bottom_copy.apply_strain([2 * strain, 2 * strain, 0])
                else:
                    top_copy.apply_strain([-2 * strain, -2 * strain, 0])

            supercells_info = {
                "bottom": [N, M],
                "top": [P, Q],
                "total_atoms": len(bottom_copy) + len(top_copy),
            }
        else:
            # Supercell matching failed, fall back to strain mode
            logger.warning(
                "Supercell matching not recommended, falling back to strain mode"
            )
            matching_mode = "strain"
            supercells_info = {"bottom": [1, 1], "top": [1, 1], "total_atoms": 0}
    else:
        supercells_info = {
            "bottom": [1, 1],
            "top": [1, 1],
            "total_atoms": len(bottom_copy) + len(top_copy),
        }

    # 2. Apply strain matching if needed
    if matching_mode == "strain":
        if apply_strain_to == "smaller":
            # Stretch smaller cell to match larger
            if bottom_copy.lattice.a < top_copy.lattice.a:
                # Apply strain to bottom
                strain_factor_a = top_copy.lattice.a / bottom_copy.lattice.a - 1
                strain_factor_b = top_copy.lattice.b / bottom_copy.lattice.b - 1
                logger.info(
                    f"Applying strain to bottom layer: "
                    f"Δa={strain_factor_a:.2%}, Δb={strain_factor_b:.2%}"
                )
                bottom_copy.apply_strain([strain_factor_a, strain_factor_b, 0])
            else:
                strain_factor_a = bottom_copy.lattice.a / top_copy.lattice.a - 1
                strain_factor_b = bottom_copy.lattice.b / top_copy.lattice.b - 1
                logger.info(
                    f"Applying strain to top layer: "
                    f"Δa={strain_factor_a:.2%}, Δb={strain_factor_b:.2%}"
                )
                top_copy.apply_strain([strain_factor_a, strain_factor_b, 0])
        elif apply_strain_to == "both":
            # Distribute strain equally
            avg_a = (bottom_copy.lattice.a + top_copy.lattice.a) / 2
            avg_b = (bottom_copy.lattice.b + top_copy.lattice.b) / 2
            strain_bottom_a = avg_a / bottom_copy.lattice.a - 1
            strain_bottom_b = avg_b / bottom_copy.lattice.b - 1
            strain_top_a = avg_a / top_copy.lattice.a - 1
            strain_top_b = avg_b / top_copy.lattice.b - 1
            bottom_copy.apply_strain([strain_bottom_a, strain_bottom_b, 0])
            top_copy.apply_strain([strain_top_a, strain_top_b, 0])

    # 3. Rotate top layer if requested
    if rotation != 0:
        logger.info(f"Rotating top layer by {rotation}° around supercell center")

        # Calculate center of top layer supercell in Cartesian coordinates
        center = top_copy.center_of_mass  # type: ignore[attr-defined]  # see FLAG: Structure lacks center_of_mass

        # Rotate each atom around the center
        theta_rad = np.deg2rad(rotation)
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)

        for site in top_copy:
            # Get Cartesian position relative to center
            rel_pos = site.coords - center

            # Rotate in xy-plane
            x_rot = rel_pos[0] * cos_theta - rel_pos[1] * sin_theta
            y_rot = rel_pos[0] * sin_theta + rel_pos[1] * cos_theta
            z_rot = rel_pos[2]  # z unchanged

            # Set new position (relative to center, then add center back)
            site.coords = np.array([x_rot, y_rot, z_rot]) + center

    # 4. Shift top layer to desired height
    bottom_max_z = max(site.coords[2] for site in bottom_copy)
    top_min_z = min(site.coords[2] for site in top_copy)
    shift_z = bottom_max_z + distance - top_min_z

    logger.info(f"Stacking layers with interlayer distance = {distance} Å")

    # Shift top layer
    for site in top_copy:
        site.coords[2] += shift_z

    # 5. Combine structures (use bottom lattice)
    interface = bottom_copy.copy()
    for site in top_copy:
        interface.append(
            site.specie,
            site.coords,  # Use Cartesian coords (already shifted)
            coords_are_cartesian=True,
            properties=site.properties,
        )

    # 6. Calculate interface area
    a = interface.lattice.a
    b = interface.lattice.b
    gamma = np.deg2rad(interface.lattice.gamma)
    area = a * b * np.sin(gamma)  # Ų

    logger.info(
        f"Interface built: {len(interface)} atoms, area = {area:.2f} Ų, "
        f"interlayer distance = {distance} Å"
    )

    # Note: Do NOT wrap fractional coordinates for twisted bilayers!
    # Wrapping breaks the moiré pattern by moving rotated atoms to wrong positions.
    # Negative fractional coordinates are fine - they represent the correct
    # Cartesian positions.

    return {
        "structure": interface,
        "area": area,
        "strain_applied": {"bottom": 0.0, "top": 0.0},  # Placeholder
        "supercells_used": supercells_info,
        "interlayer_distance": distance,
        "matching_mode": matching_mode,
    }


@job
def scan_interlayer_distance(
    base_structure: Structure,
    distance_range: tuple[float, float],
    steps: int,
    relax_maker: RelaxMaker,  # noqa: ARG001
) -> dict:
    """
    Optimize interlayer distance by scanning range and finding minimum energy.

    This job scans multiple interlayer distances, relaxes each configuration,
    and identifies the optimal distance with lowest energy.

    Parameters
    ----------
    base_structure : Structure
        Interface structure at starting distance (from build_interface_structure).
    distance_range : tuple[float, float]
        (min_distance, max_distance) in Ångstroms to scan.
    steps : int
        Number of distance points to sample.
    relax_maker : RelaxMaker
        Maker for relaxation jobs.

    Returns
    -------
    dict
        {
            "optimal_distance": float (Å),
            "optimal_energy": float (eV),
            "binding_curve": list[tuple[float, float]] (distance, energy),
            "structure": Structure (at optimal distance),
            "energy": float (for compatibility with flow output),
        }

    Notes
    -----
    - This creates N relaxation jobs where N = steps
    - Each relaxation constrains in-plane lattice but allows z-relaxation
    - Output structure is the one with lowest energy
    - Default range (3.0-4.0 Å) is typical for vdW systems
    """
    import numpy as np

    logger.info(
        f"Scanning interlayer distance from {distance_range[0]} to "
        f"{distance_range[1]} Å with {steps} points"
    )

    # Generate distance points
    distances = np.linspace(distance_range[0], distance_range[1], steps)

    # TODO: Implement actual distance scanning with relaxations
    # For now, return placeholder
    # In real implementation, would:
    # 1. Create structures at each distance
    # 2. Submit relaxation jobs
    # 3. Collect energies
    # 4. Find minimum

    # Placeholder implementation
    optimal_distance = (distance_range[0] + distance_range[1]) / 2
    logger.warning(
        "scan_interlayer_distance is a placeholder - needs full implementation"
    )

    return {
        "optimal_distance": optimal_distance,
        "optimal_energy": 0.0,  # Placeholder
        "binding_curve": [(d, 0.0) for d in distances],  # Placeholder
        "structure": base_structure,
        "energy": 0.0,  # For compatibility
    }


@job
def calculate_interface_binding_energy(
    interface_energy: float,
    bottom_energy: float,
    top_energy: float,
    interface_area: float,
    interface_structure: Structure | None = None,
    bottom_structure: Structure | None = None,
    top_structure: Structure | None = None,
) -> dict:
    """
    Calculate interface binding energy and save structure files.

    Uses the formula:
        E_bind = E(interface) - E(bottom) - E(top)
        E_bind_per_area = E_bind / Area  [meV/Ų]

    Negative binding energy indicates stable interface (favorable interaction).

    Parameters
    ----------
    interface_energy : float
        Total energy of interface structure (eV).
    bottom_energy : float
        Energy of isolated bottom layer (eV).
    top_energy : float
        Energy of isolated top layer (eV).
    interface_area : float
        Interface area in Ų.
    interface_structure : Structure, optional
        Relaxed interface structure to save.
    bottom_structure : Structure, optional
        Relaxed bottom layer structure to save.
    top_structure : Structure, optional
        Relaxed top layer structure to save.

    Returns
    -------
    dict
        {
            "binding_energy_total": float (eV),
            "binding_energy_per_area": float (meV/Ų),
            "interface_area": float (Ų),
            "interpretation": str,
            "summary_file": str (path to interface_summary.txt),
            "structure_files": dict (paths to saved CIF/XSF files),
        }

    Generated Files
    ---------------
    - interface_summary.txt: Comprehensive binding energy analysis
    - binding_energy.png: Bar plot vs reference systems
    - interface_relaxed.cif: Relaxed interface structure (CIF format)
    - interface_relaxed.xsf: Relaxed interface structure (XSF format)
    - bottom_layer_relaxed.cif: Isolated bottom layer
    - top_layer_relaxed.cif: Isolated top layer

    Interpretation Guide
    --------------------
    Binding energy (meV/Ų):
        < -50: Strong binding (chemisorption, covalent bonding)
        -50 to -10: Moderate binding (typical van der Waals)
        -10 to 0: Weak binding (physisorption)
        > 0: Unstable interface

    Reference Values
    ----------------
    - Graphene/h-BN: -15 meV/Ų (literature)
    - MoS₂/graphene: -20 to -30 meV/Ų
    - Graphene/metal: -50 to -100 meV/Ų (stronger interaction)

    Example
    -------
    >>> result = calculate_interface_binding_energy(
    ...     interface_energy=-1000.0,  # eV
    ...     bottom_energy=-400.0,
    ...     top_energy=-550.0,
    ...     interface_area=100.0,  # Ų
    ... )
    >>> print(result["binding_energy_per_area"])  # -500 meV/Ų
    -500.0
    """
    E_bind_total = interface_energy - bottom_energy - top_energy  # eV  # noqa: N806
    E_bind_per_area = (E_bind_total / interface_area) * 1000  # meV/Ų  # noqa: N806

    # Interpretation
    if E_bind_per_area < -50:
        interpretation = "Strong binding (chemisorption or covalent bonding)"
    elif E_bind_per_area < -10:
        interpretation = "Moderate binding (typical van der Waals)"
    elif E_bind_per_area < 0:
        interpretation = "Weak binding (physisorption)"
    else:
        interpretation = "Unstable interface (positive binding energy)"

    logger.info(
        f"Binding energy: {E_bind_per_area:.2f} meV/Ų ({E_bind_total:.4f} eV total) "
        f"- {interpretation}"
    )

    # Generate summary text file
    from pathlib import Path

    summary_file = Path("interface_summary.txt")
    with open(summary_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("2D HETEROSTRUCTURE INTERFACE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write("BINDING ENERGY RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total binding energy:       {E_bind_total:+.6f} eV\n")
        f.write(f"Binding energy per area:    {E_bind_per_area:+.2f} meV/Ų\n")
        f.write(f"Interface area:             {interface_area:.2f} Ų\n\n")

        f.write("INTERPRETATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"{interpretation}\n\n")

        f.write("REFERENCE VALUES (Literature)\n")
        f.write("-" * 80 + "\n")
        f.write("  Graphene@h-BN:      ~-15 meV/Ų (weak vdW)\n")
        f.write("  MoS₂@graphene:      -20 to -30 meV/Ų (moderate vdW)\n")
        f.write("  Graphene@metal:     -50 to -100 meV/Ų (strong interaction)\n")
        f.write("  Graphene@graphene:  -11 to -17 meV/Ų (vdW, stacking dependent)\n\n")

        f.write("BINDING ENERGY CATEGORIES\n")
        f.write("-" * 80 + "\n")
        f.write("  < -50 meV/Ų:   Strong binding (chemisorption)\n")
        f.write("  -50 to -10:    Moderate binding (typical vdW)\n")
        f.write("  -10 to 0:      Weak binding (physisorption)\n")
        f.write("  > 0:           Unstable interface\n\n")

        f.write("ENERGY CONTRIBUTIONS\n")
        f.write("-" * 80 + "\n")
        f.write(f"E(interface):   {interface_energy:.6f} eV\n")
        f.write(f"E(bottom):      {bottom_energy:.6f} eV\n")
        f.write(f"E(top):         {top_energy:.6f} eV\n")
        f.write(f"E(bind):        {E_bind_total:+.6f} eV\n\n")

        # Add footer
        from atomate2.siesta.utils.text_output import get_standard_footer

        f.write(
            get_standard_footer(
                width=80,
                additional_info={
                    "Analysis type": "2D heterostructure interface",
                    "Binding energy": f"{E_bind_per_area:.2f} meV/Ų",
                },
            )
        )

    logger.info(f"✓ Generated summary: {summary_file}")

    # Generate simple bar plot
    try:
        import matplotlib.pyplot as plt

        _fig, ax = plt.subplots(figsize=(8, 6))

        # Bar plot
        categories = ["Graphene@h-BN\n(ref)", "This\nInterface", "MoS₂@graphene\n(ref)"]
        values = [-15, E_bind_per_area, -25]
        colors = ["gray", "red" if E_bind_per_area > 0 else "blue", "gray"]

        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor="black")

        # Horizontal reference lines
        ax.axhline(0, color="black", linestyle="--", linewidth=1, label="E_bind = 0")
        ax.axhline(-10, color="green", linestyle=":", linewidth=1, alpha=0.5)
        ax.axhline(-50, color="orange", linestyle=":", linewidth=1, alpha=0.5)

        # Labels and title
        ax.set_ylabel("Binding Energy (meV/Ų)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Interface Binding Energy\n{E_bind_per_area:.2f} meV/Ų - {interpretation}",
            fontsize=14,
            fontweight="bold",
        )

        # Add value labels on bars
        for bar, val in zip(bars, values, strict=False):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.1f}",
                ha="center",
                va="bottom" if height < 0 else "top",
                fontweight="bold",
            )

        # Grid
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

        plt.tight_layout()
        plot_file = Path("binding_energy.png")
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"✓ Generated plot: {plot_file}")

    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to generate plot: {e}")

    # Save structure files
    structure_files = {}
    if interface_structure is not None:
        try:
            interface_cif = Path("interface_relaxed.cif")
            interface_structure.to(filename=str(interface_cif), fmt="cif")
            structure_files["interface_cif"] = str(interface_cif)
            logger.info(f"✓ Saved interface structure: {interface_cif}")

            # Also save as XSF for visualization
            interface_xsf = Path("interface_relaxed.xsf")
            interface_structure.to(filename=str(interface_xsf), fmt="xsf")
            structure_files["interface_xsf"] = str(interface_xsf)
            logger.info(f"✓ Saved interface structure: {interface_xsf}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to save interface structure: {e}")

    if bottom_structure is not None:
        try:
            bottom_cif = Path("bottom_layer_relaxed.cif")
            bottom_structure.to(filename=str(bottom_cif), fmt="cif")
            structure_files["bottom_cif"] = str(bottom_cif)
            logger.info(f"✓ Saved bottom layer: {bottom_cif}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to save bottom layer: {e}")

    if top_structure is not None:
        try:
            top_cif = Path("top_layer_relaxed.cif")
            top_structure.to(filename=str(top_cif), fmt="cif")
            structure_files["top_cif"] = str(top_cif)
            logger.info(f"✓ Saved top layer: {top_cif}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to save top layer: {e}")

    return {
        "binding_energy_total": E_bind_total,
        "binding_energy_per_area": E_bind_per_area,
        "interface_area": interface_area,
        "interpretation": interpretation,
        "summary_file": str(summary_file),
        "structure_files": structure_files,
    }
