"""Flows for calculating elastic constants with SIESTA."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jobflow import Flow, job

from atomate2.common.flows.elastic import BaseElasticMaker
from atomate2.siesta.flows.base import BaseSiestaFlowMaker
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker

if TYPE_CHECKING:
    from pathlib import Path

    from atomate2.siesta.jobs.base import BaseSiestaMaker


@dataclass
class ElasticFlowMaker(BaseSiestaFlowMaker, BaseElasticMaker):
    """
    Maker to calculate elastic constants using SIESTA.

    Calculate the elastic tensor of a material using SIESTA. The workflow:

    1. **Structural relaxation**: Performs a tight relaxation to achieve ~zero stress state
    2. **Apply deformations**: Generates strain perturbations on the lattice
    3. **Calculate stresses**: Runs SIESTA calculations for each deformed structure
    4. **Fit elastic tensor**: Uses linear elasticity to fit the 6×6 elastic tensor
    5. **Derive properties**: Calculates bulk modulus, shear modulus, Young's modulus, etc.

    The elastic tensor relates stress (σ) and strain (ε) via Hooke's law:
        σᵢⱼ = Cᵢⱼₖₗ εₖₗ

    where C is the 4th-rank elastic tensor (stored as 6×6 matrix in Voigt notation).

    From the elastic tensor, we can derive:
    - Bulk modulus (K): Resistance to compression
    - Shear modulus (G): Resistance to shear
    - Young's modulus (E): Stiffness in tension
    - Poisson's ratio (ν): Lateral/axial strain ratio
    - Sound velocities
    - Elastic anisotropy

    .. Note::
        It is strongly recommended to symmetrize the input structure first.
        This allows the symmetry reduction to minimize the number of required
        deformation calculations.

    Parameters
    ----------
    name : str
        Name of the workflow.
    order : int
        Order of the tensor expansion (2 or 3). Order 2 is standard elasticity.
        Order 3 includes third-order elastic constants (for high strain).
    sym_reduce : bool
        Whether to use symmetry to reduce the number of deformations.
    symprec : float
        Symmetry precision for spacegroup operations.
    bulk_relax_maker : RelaxMaker or None
        Maker for initial relaxation to zero-stress state.
        Set to None to skip (if structure is already relaxed).
    elastic_relax_maker : StaticMaker or RelaxMaker
        Maker for calculating stresses on deformed structures.
        Typically a static calculation (ions fixed) or fixed-volume relax.
    max_failed_deformations : int or float or None
        Maximum number of failed deformations to tolerate:
        - int: absolute number
        - float (0-1): fraction of total deformations
        - None: allow any number to fail
    generate_elastic_deformations_kwargs : dict
        Additional kwargs for deformation generation.
    fit_elastic_tensor_kwargs : dict
        Additional kwargs for tensor fitting.
    task_document_kwargs : dict
        Additional kwargs for ElasticDocument creation.

    Example
    -------
    >>> from atomate2.siesta.flows.elastic import ElasticFlowMaker
    >>> from pymatgen.core import Structure
    >>>
    >>> structure = Structure.from_file("POSCAR")
    >>>
    >>> # Create elastic maker with custom settings
    >>> elastic_maker = ElasticFlowMaker(
    ...     bulk_relax_maker=RelaxMaker.variable_cell_relaxation(
    ...         {"PAO.BasisSize": "DZP", "Mesh.Cutoff": 300, "a2s_kpts": [6, 6, 6]}
    ...     ),
    ...     elastic_relax_maker=StaticMaker(
    ...         {"PAO.BasisSize": "DZP", "Mesh.Cutoff": 300, "a2s_kpts": [8, 8, 8]}
    ...     ),
    ... )
    >>>
    >>> flow = elastic_maker.make(structure)
    >>>
    >>> # Run the workflow
    >>> from jobflow import run_locally
    >>> results = run_locally(flow)
    """  # noqa: RUF002

    name: str = "elastic"
    order: int = 2
    sym_reduce: bool = True
    symprec: float = 1e-5
    bulk_relax_maker: BaseSiestaMaker | None = field(
        default_factory=RelaxMaker.variable_cell_relaxation
    )
    elastic_relax_maker: BaseSiestaMaker = field(default_factory=StaticMaker)
    max_failed_deformations: int | float | None = None
    generate_elastic_deformations_kwargs: dict = field(default_factory=dict)
    fit_elastic_tensor_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)

    # Note: dry_run, use_custodian, and tier support inherited from BaseSiestaFlowMaker

    @property
    def prev_calc_dir_argname(self) -> str:
        """Name of argument for previous calculation directory in SIESTA."""
        return "prev_dir"

    def make(self, structure, prev_dir: str | Path | None = None, **kwargs) -> Flow:
        """
        Make flow to calculate elastic constants with automatic result saving.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure.
        prev_dir : str or Path or None
            A previous calculation directory to use for copying outputs.
        **kwargs
            Additional keyword arguments passed to BaseElasticFlowMaker.make()

        Returns
        -------
        Flow
            An elastic constants flow with automatic result export.
        """
        from atomate2.siesta.utils.common import print_docstring_in_box

        print_docstring_in_box(self.__doc__, title=self.__class__.__name__)

        # Create the base elastic flow
        elastic_flow = super().make(structure, prev_dir=prev_dir, **kwargs)

        # Add a job to save results automatically
        save_job = save_elastic_results_job(
            elastic_doc=elastic_flow.output, output_folder="."
        )
        save_job.name = f"{self.name}-save_results"

        # Return flow with save job appended
        return Flow(
            jobs=[elastic_flow, save_job], output=elastic_flow.output, name=self.name
        )

    @property
    def stress_sign_correction(self) -> float:
        """
        Correct the sign AND units of the stress tensor.

        SIESTA parser converts stress using EV_PER_A3_TO_KBAR but appears to actually
        output in a unit that's 100x too small (possibly bar instead of kbar, or
        some other unit conversion issue in the parser or TaskDocument).

        The ElasticDocument.from_stresses applies -0.1 assuming input is in kbar.
        Empirically, we need a factor of -1000 to get correct values:
        - Silicon C11 should be ~160 GPa
        - With factor -10, we get 0.89 GPa (100x too small)
        - With factor -1000, we should get ~89 GPa (closer to correct)

        Returns
        -------
        float
            Sign+unit correction factor (-1000.0).
        """
        return -1000.0


@job
def save_elastic_results_job(elastic_doc, output_folder: str = "."):
    """
    Job to save elastic constants results with JSON and TXT formats.

    Parameters
    ----------
    elastic_doc : ElasticDocument
        The elastic document from fit_elastic_tensor job
    output_folder : str
        Directory to save results (default: current directory ".")

    Returns
    -------
    dict
        Dictionary with paths to saved files
    """
    import json
    from datetime import datetime
    from pathlib import Path

    import numpy as np

    # Use output folder (current directory by default)
    output_path = Path(output_folder)
    if output_folder != ".":
        output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Convert to dict if it's a pydantic model
    if hasattr(elastic_doc, "model_dump"):
        elastic_dict = elastic_doc.model_dump()
    elif hasattr(elastic_doc, "dict"):
        elastic_dict = elastic_doc.dict()
    else:
        elastic_dict = elastic_doc

    # Extract data
    formula = elastic_dict.get("formula_pretty", "Unknown")
    elastic_tensor_data = elastic_dict.get("elastic_tensor", {})
    ieee_tensor = elastic_tensor_data.get("ieee_format", [])

    props = elastic_dict.get("derived_properties", {})
    k_vrh = props.get("k_vrh")
    k_voigt = props.get("k_voigt")
    k_reuss = props.get("k_reuss")
    g_vrh = props.get("g_vrh")
    g_voigt = props.get("g_voigt")
    g_reuss = props.get("g_reuss")
    y_mod = props.get("y_mod")
    poisson = props.get("homogeneous_poisson")
    universal_aniso = props.get("universal_anisotropy")

    symmetry = elastic_dict.get("symmetry", {})
    eq_stress = elastic_dict.get("eq_stress")
    fitting_method = elastic_dict.get("fitting_method", "unknown")
    order = elastic_dict.get("order", 2)
    nsites = elastic_dict.get("nsites")
    volume = elastic_dict.get("volume")
    density = elastic_dict.get("density")

    # Convert crystal_system enum to string if needed
    crystal_system = symmetry.get("crystal_system")
    if hasattr(crystal_system, "value"):
        crystal_system = crystal_system.value
    elif crystal_system is not None:
        crystal_system = str(crystal_system)

    # Save JSON file
    json_file = output_path / f"elastic_results_{timestamp}.json"
    results_dict = {
        "metadata": {
            "formula": formula,
            "timestamp": timestamp,
            "crystal_system": crystal_system,
            "space_group": symmetry.get("symbol"),
        },
        "mechanical_properties": {
            "bulk_modulus_vrh_GPa": k_vrh,
            "shear_modulus_vrh_GPa": g_vrh,
            "youngs_modulus_GPa": y_mod / 1e9
            if y_mod is not None
            else None,  # Convert Pa to GPa
            "poisson_ratio": poisson,
        },
        "elastic_tensor_ieee_GPa": ieee_tensor,
    }

    with open(json_file, "w") as f:
        json.dump(results_dict, f, indent=2)

    # Check for negative elastic constants
    has_negative_constants = False
    if ieee_tensor:
        tensor_array = np.array(ieee_tensor)
        diagonal = np.diag(tensor_array)
        has_negative_constants = np.any(diagonal < 0)

    # Save TXT summary
    txt_file = output_path / f"elastic_summary_{timestamp}.txt"
    with open(txt_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ELASTIC CONSTANTS CALCULATION RESULTS\n")
        f.write("=" * 80 + "\n\n")

        # Header information
        f.write("CALCULATION INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Formula:              {formula}\n")
        f.write(f"Crystal System:       {crystal_system}\n")
        f.write(
            f"Space Group:          {symmetry.get('symbol', 'N/A')} (#{symmetry.get('number', 'N/A')})\n"
        )
        f.write(f"Fitting Method:       {fitting_method}\n")
        f.write(f"Tensor Order:         {order}\n")
        f.write(f"Timestamp:            {timestamp}\n")
        f.write("\n")

        # Structure information
        f.write("STRUCTURE INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Number of Sites:      {nsites}\n")
        f.write(
            f"Volume:               {volume:.4f} ų\n"
            if volume
            else "Volume:               N/A\n"
        )
        f.write(
            f"Density:              {density:.4f} g/cm³\n"
            if density
            else "Density:              N/A\n"
        )
        f.write("\n")

        # Elastic tensor
        f.write("ELASTIC TENSOR (IEEE format, GPa)\n")
        f.write("-" * 80 + "\n")
        f.write("Voigt notation (6×6 symmetric matrix):\n\n")  # noqa: RUF001
        if ieee_tensor:
            tensor_array = np.array(ieee_tensor)
            f.write("      [1]       [2]       [3]       [4]       [5]       [6]\n")
            f.writelines(
                f"  [{i + 1}] " + "  ".join(f"{val:9.4f}" for val in row) + "\n"
                for i, row in enumerate(tensor_array)
            )
        else:
            f.write("  No tensor data available\n")
        f.write("\n")

        # Equilibrium stress
        if eq_stress:
            f.write("EQUILIBRIUM STRESS (GPa)\n")
            f.write("-" * 80 + "\n")
            stress_array = np.array(eq_stress)
            f.writelines(
                f"  [{i + 1}] " + "  ".join(f"{val:10.6f}" for val in row) + "\n"
                for i, row in enumerate(stress_array)
            )
            f.write("\n")

        # Check for negative elastic constants (indicates problems)
        has_negative = False
        negative_indices = []
        if ieee_tensor:
            tensor_array = np.array(ieee_tensor)
            diagonal = np.diag(tensor_array)
            has_negative = np.any(diagonal < 0)
            negative_indices = np.where(diagonal < 0)[0]

        # WARNING if negative constants found
        if has_negative:
            f.write("\n")
            f.write("!" * 80 + "\n")
            f.write("⚠️  WARNING: NEGATIVE ELASTIC CONSTANTS DETECTED\n")
            f.write("!" * 80 + "\n\n")
            f.write(
                "Diagonal elements with negative values: C_"
                + ", C_".join([f"{i + 1}{i + 1}" for i in negative_indices])
                + "\n\n"
            )
            f.write("Negative elastic constants indicate:\n")
            f.write(
                "  1. MECHANICAL INSTABILITY - Structure is not at a stable minimum\n"
            )
            f.write(
                "  2. INSUFFICIENT CONVERGENCE - Calculation parameters too loose\n"
            )
            f.write("  3. POOR INITIAL RELAXATION - Structure not at zero stress\n\n")
            f.write("RECOMMENDED FIXES:\n")
            f.write("-" * 80 + "\n")
            f.write("1. Increase k-point density:\n")
            f.write("   • Use at least 6×6×6 (better: 8×8×8) for elastic constants\n")  # noqa: RUF001
            f.write("   • Elastic properties require denser k-mesh than energies\n\n")
            f.write("2. Increase energy cutoff:\n")
            f.write("   • Use Mesh.Cutoff ≥ 300 Ry (better: 400 Ry)\n")
            f.write("   • Stress calculations need tighter convergence\n\n")
            f.write("3. Use better basis set:\n")
            f.write("   • Minimum: DZP (not SZ)\n")
            f.write("   • Recommended: DZDP or TZP for accurate forces/stresses\n\n")
            f.write("4. Ensure tight initial relaxation:\n")
            f.write("   • Forces < 0.01 eV/Å\n")
            f.write("   • Stress < 0.1 GPa on all components\n\n")
            f.write("5. Check structure stability:\n")
            f.write("   • Phonon dispersion (no imaginary modes)\n")
            f.write("   • Energy minimization completed properly\n\n")
            f.write("DO NOT TRUST THESE RESULTS UNTIL CONVERGENCE IS ACHIEVED!\n")
            f.write("=" * 80 + "\n\n")

        # Mechanical properties
        f.write("DERIVED MECHANICAL PROPERTIES\n")
        f.write("=" * 80 + "\n")

        f.write("Bulk Modulus (K) - Resistance to uniform compression\n")
        f.write("-" * 80 + "\n")
        if k_voigt is not None:
            f.write(f"  K_Voigt:            {k_voigt:10.4f} GPa  (upper bound)\n")
        if k_reuss is not None:
            f.write(f"  K_Reuss:            {k_reuss:10.4f} GPa  (lower bound)\n")
        if k_vrh is not None:
            f.write(
                f"  K_VRH:              {k_vrh:10.4f} GPa  ⭐ (recommended average)\n"
            )
        f.write("\n")

        f.write("Shear Modulus (G) - Resistance to shear deformation\n")
        f.write("-" * 80 + "\n")
        if g_voigt is not None:
            f.write(f"  G_Voigt:            {g_voigt:10.4f} GPa  (upper bound)\n")
        if g_reuss is not None:
            f.write(f"  G_Reuss:            {g_reuss:10.4f} GPa  (lower bound)\n")
        if g_vrh is not None:
            f.write(
                f"  G_VRH:              {g_vrh:10.4f} GPa  ⭐ (recommended average)\n"
            )
        f.write("\n")

        if y_mod is not None:
            f.write("Young's Modulus (E) - Stiffness in tension/compression\n")
            f.write("-" * 80 + "\n")
            # Convert from Pa to GPa (y_mod is in SI units = Pa)
            f.write(f"  E:                  {y_mod / 1e9:10.4f} GPa\n")
            f.write("\n")

        if poisson is not None:
            f.write("Poisson's Ratio (ν) - Lateral to axial strain ratio\n")  # noqa: RUF001
            f.write("-" * 80 + "\n")
            f.write(f"  ν:                  {poisson:10.6f}\n")  # noqa: RUF001
            f.write("\n")

        if universal_aniso is not None:
            f.write("Universal Anisotropy - Directional property variation\n")
            f.write("-" * 80 + "\n")
            f.write(f"  A_U:                {universal_aniso:10.6f}\n")
            f.write("  (0 = isotropic, >0 = anisotropic)\n")
            f.write("\n")

        # Ductility indicator
        if k_vrh and g_vrh and k_vrh != 0 and g_vrh != 0:
            pugh_ratio = k_vrh / g_vrh
            f.write("DUCTILITY INDICATOR (Pugh's Ratio)\n")
            f.write("=" * 80 + "\n")
            f.write(f"  K/G Ratio:          {pugh_ratio:10.4f}\n")
            if pugh_ratio > 1.75:
                f.write("  Prediction:         DUCTILE (deforms plastically)\n")
            else:
                f.write("  Prediction:         BRITTLE (fractures easily)\n")
            f.write("\n")

        # Interpretation guide
        f.write("=" * 80 + "\n")
        f.write("INTERPRETATION GUIDE\n")
        f.write("=" * 80 + "\n\n")

        f.write("Voigt-Reuss-Hill Averaging:\n")
        f.write("  The VRH values are arithmetic means of Voigt (upper bound) and\n")
        f.write(
            "  Reuss (lower bound) estimates. VRH is recommended for polycrystalline\n"
        )
        f.write("  materials as it provides a reasonable approximation.\n\n")

        f.write("Typical Property Ranges:\n")
        f.write("-" * 80 + "\n")
        f.write("Material Type       K (GPa)    G (GPa)    E (GPa)    ν\n")  # noqa: RUF001
        f.write("-" * 80 + "\n")
        f.write("Soft polymers       1-10       0.1-1      0.5-5      0.4-0.5\n")
        f.write("Metals (soft)       50-100     20-40      50-120     0.3-0.4\n")
        f.write("Metals (hard)       100-200    40-80      100-220    0.25-0.35\n")
        f.write("Semiconductors      80-120     50-80      120-200    0.2-0.3\n")
        f.write("Ceramics            200-400    100-200    250-500    0.2-0.3\n")
        f.write("Diamond             440        535        1050       0.07\n")
        f.write("\n")

        f.write("Property Meanings:\n")
        f.write("-" * 80 + "\n")
        f.write("• Bulk Modulus (K):\n")
        f.write("  Higher K = harder to compress uniformly\n")
        f.write("  Related to bond strength and atomic packing\n\n")

        f.write("• Shear Modulus (G):\n")
        f.write("  Higher G = more resistant to shape change\n")
        f.write("  Related to dislocation mobility and plastic deformation\n\n")

        f.write("• Young's Modulus (E):\n")
        f.write("  Higher E = stiffer material (resists elongation)\n")
        f.write("  Important for structural applications\n\n")

        f.write("• Poisson's Ratio (ν):\n")  # noqa: RUF001
        f.write("  ν → 0.5:  Nearly incompressible (rubber-like)\n")  # noqa: RUF001
        f.write("  ν ≈ 0.3:  Typical metals\n")  # noqa: RUF001
        f.write("  ν ≈ 0.2:  Ceramics/semiconductors\n")  # noqa: RUF001
        f.write("  ν → 0.0:  Cork-like behavior\n\n")  # noqa: RUF001

        f.write("• Pugh's Ratio (K/G):\n")
        f.write("  K/G > 1.75:  Ductile (bends without breaking)\n")
        f.write("  K/G < 1.75:  Brittle (shatters under stress)\n\n")

        f.write("• Universal Anisotropy (A_U):\n")
        f.write("  A_U = 0:    Isotropic (same properties in all directions)\n")
        f.write("  A_U > 0:    Anisotropic (directional properties)\n")
        f.write("  A_U > 1:    Highly anisotropic (e.g., layered materials)\n\n")

        # Important notes
        f.write("=" * 80 + "\n")
        f.write("IMPORTANT NOTES\n")
        f.write("=" * 80 + "\n\n")

        f.write("⚠ DFT Calculations at 0 K:\n")
        f.write("  These results are computed at 0 K. Elastic constants typically\n")
        f.write("  decrease with temperature. Expect 5-10% difference compared to\n")
        f.write("  room temperature (300 K) experimental measurements.\n\n")

        f.write("⚠ Convergence Requirements:\n")
        f.write("  Accurate elastic constants require:\n")
        f.write("  - Well-converged k-point mesh (denser than energy calculations)\n")
        f.write("  - High mesh cutoff (300+ Ry recommended)\n")
        f.write("  - Good basis set (DZP minimum, DZDP/TZP for high accuracy)\n")
        f.write(
            "  - Tight initial relaxation (forces < 0.01 eV/Å, stress < 0.1 GPa)\n\n"
        )

        if k_vrh and k_vrh < 0:
            f.write("⚠⚠ WARNING: NEGATIVE ELASTIC CONSTANTS DETECTED! ⚠⚠\n")
            f.write("  This indicates either:\n")
            f.write(
                "  1. Structure is mechanically unstable (imaginary phonon modes)\n"
            )
            f.write("  2. Calculation parameters need improvement:\n")
            f.write("     - Increase k-point density\n")
            f.write("     - Increase mesh cutoff\n")
            f.write("     - Use better basis set (upgrade to DZP/DZDP)\n")
            f.write("     - Ensure proper initial relaxation\n")
            f.write("  Please re-run with higher quality computational settings.\n\n")

        f.write("=" * 80 + "\n")
        f.write("REFERENCES\n")
        f.write("=" * 80 + "\n\n")

        f.write("[1] Voigt, W. (1928). Lehrbuch der Kristallphysik.\n")
        f.write("    Teubner, Leipzig.\n")
        f.write("    - Original work on upper bound elastic constant estimates\n\n")

        f.write(
            "[2] Reuss, A. (1929). Berechnung der Fließgrenze von Mischkristallen\n"
        )
        f.write("    auf Grund der Plastizitätsbedingung für Einkristalle.\n")
        f.write("    Z. Angew. Math. Mech. 9, 49-58.\n")
        f.write("    - Lower bound elastic constant estimates\n\n")

        f.write(
            "[3] Hill, R. (1952). The Elastic Behaviour of a Crystalline Aggregate.\n"
        )
        f.write("    Proc. Phys. Soc. A 65, 349-354.\n")
        f.write("    DOI: 10.1088/0370-1298/65/5/307\n")
        f.write("    - Voigt-Reuss-Hill averaging method\n\n")

        f.write("[4] Nye, J.F. (1985). Physical Properties of Crystals:\n")
        f.write("    Their Representation by Tensors and Matrices.\n")
        f.write("    Oxford University Press, Oxford.\n")
        f.write("    - Comprehensive reference on elastic tensor theory\n\n")

        f.write("[5] Pugh, S.F. (1954). Relations between the elastic moduli and\n")
        f.write("    the plastic properties of polycrystalline pure metals.\n")
        f.write("    Philos. Mag. 45, 823-843.\n")
        f.write("    DOI: 10.1080/14786440808520496\n")
        f.write("    - K/G ratio for ductile/brittle behavior prediction\n\n")

        f.write("[6] Ranganathan, S.I. & Ostoja-Starzewski, M. (2008).\n")
        f.write("    Universal elastic anisotropy index.\n")
        f.write("    Phys. Rev. Lett. 101, 055504.\n")
        f.write("    DOI: 10.1103/PhysRevLett.101.055504\n")
        f.write("    - Universal anisotropy measure\n\n")

        f.write("[7] Le Page, Y. & Saxe, P. (2002). Symmetry-general least-squares\n")
        f.write(
            "    extraction of elastic data for strained materials from ab initio\n"
        )
        f.write("    calculations of stress.\n")
        f.write("    Phys. Rev. B 65, 104104.\n")
        f.write("    DOI: 10.1103/PhysRevB.65.104104\n")
        f.write("    - Method for elastic constant calculations from DFT\n\n")

        f.write("[8] Mouhat, F. & Coudert, F.-X. (2014). Necessary and sufficient\n")
        f.write("    elastic stability conditions in various crystal systems.\n")
        f.write("    Phys. Rev. B 90, 224104.\n")
        f.write("    DOI: 10.1103/PhysRevB.90.224104\n")
        f.write("    - Born stability criteria for different crystal systems\n\n")

        f.write(
            "[9] de Jong, M. et al. (2015). Charting the complete elastic properties\n"
        )
        f.write("    of inorganic crystalline compounds.\n")
        f.write("    Sci. Data 2, 150009.\n")
        f.write("    DOI: 10.1038/sdata.2015.9\n")
        f.write("    - Large-scale elastic constants database and methodology\n\n")

        f.write("[10] Soler, J.M. et al. (2002). The SIESTA method for ab initio\n")
        f.write("     order-N materials simulation.\n")
        f.write("     J. Phys.: Condens. Matter 14, 2745-2779.\n")
        f.write("     DOI: 10.1088/0953-8984/14/11/302\n")
        f.write("     - SIESTA DFT code used for these calculations\n\n")

        # Add standard footer
        from atomate2.siesta.utils.text_output import get_standard_footer

        f.write(
            get_standard_footer(
                width=80,
                additional_info={
                    "Analysis type": "Elastic constants calculation",
                    "Crystal system": crystal_system,
                    "Formula": formula,
                },
            )
        )

    # Print warning to console if negative constants detected
    if has_negative_constants:
        from atomate2.siesta.utils.common import console

        if console:
            console.print(
                "\n[bold red]⚠️  WARNING: Negative elastic constants detected![/bold red]"
            )
            console.print(
                "[yellow]The elastic tensor contains negative diagonal elements.[/yellow]"
            )
            console.print(
                "[yellow]This indicates mechanical instability or poor convergence.[/yellow]"
            )
            console.print("\n[cyan]Recommendations:[/cyan]")
            console.print("  • Increase k-points: Use ≥6×6×6 (better: 8×8×8)")  # noqa: RUF001
            console.print(
                "  • Increase cutoff: Use Mesh.Cutoff ≥300 Ry (better: 400 Ry)"
            )
            console.print("  • Better basis: Use DZP minimum (DZDP/TZP recommended)")
            console.print("  • Tight relaxation: Forces <0.01 eV/Å, Stress <0.1 GPa")
            console.print(f"\n[cyan]See detailed warnings in: {txt_file}[/cyan]\n")

    # Generate all plots
    plot_files = {}

    # 1. Elastic tensor heatmap
    if ieee_tensor:
        heatmap_file = plot_elastic_tensor_heatmap(
            ieee_tensor, formula, output_path, timestamp
        )
        if heatmap_file:
            plot_files["elastic_tensor_heatmap"] = heatmap_file

    # 2. Mechanical properties bar chart
    if k_vrh is not None or g_vrh is not None:
        bar_file = plot_mechanical_properties_bar(
            k_vrh, g_vrh, y_mod, poisson, formula, output_path, timestamp
        )
        if bar_file:
            plot_files["mechanical_properties_bar"] = bar_file

    # 3. Stress-strain curves (if deformation data available)
    deformations = elastic_dict.get("deformations", [])
    stresses = elastic_dict.get("stresses", [])
    if deformations and stresses:
        ss_file = plot_stress_strain_curves(
            deformations, stresses, formula, output_path, timestamp
        )
        if ss_file:
            plot_files["stress_strain_curves"] = ss_file

    # 4. 3D Young's modulus surface
    if ieee_tensor:
        youngs_3d_file = plot_youngs_modulus_3d(
            ieee_tensor, formula, output_path, timestamp
        )
        if youngs_3d_file:
            plot_files["youngs_modulus_3d"] = youngs_3d_file

    # 5. 3D Linear compressibility
    if ieee_tensor:
        compress_3d_file = plot_linear_compressibility_3d(
            ieee_tensor, formula, output_path, timestamp
        )
        if compress_3d_file:
            plot_files["linear_compressibility_3d"] = compress_3d_file

    # 6. Pugh's ratio diagram
    if k_vrh is not None and g_vrh is not None:
        pugh_file = plot_pugh_ratio_diagram(
            k_vrh, g_vrh, formula, output_path, timestamp
        )
        if pugh_file:
            plot_files["pugh_ratio_diagram"] = pugh_file

    return {
        "json_file": str(json_file),
        "txt_file": str(txt_file),
        "output_folder": str(output_path),
        "has_warnings": has_negative_constants,
        "plot_files": plot_files,
    }


def plot_elastic_tensor_heatmap(
    ieee_tensor: list,
    formula: str,
    output_path: Path,
    timestamp: str,
) -> str | None:
    """
    Plot elastic tensor as a heatmap.

    Parameters
    ----------
    ieee_tensor : list
        6×6 elastic tensor in IEEE format (GPa)
    formula : str
        Chemical formula
    output_path : Path
        Output directory
    timestamp : str
        Timestamp for filename

    Returns
    -------
    str or None
        Path to saved plot file, or None if failed
    """  # noqa: RUF002
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        tensor = np.array(ieee_tensor)

        _fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap with diverging colormap centered at 0
        vmax = np.abs(tensor).max()
        vmin = -vmax if tensor.min() < 0 else 0

        im = ax.imshow(tensor, cmap="RdBu_r", aspect="equal", vmin=vmin, vmax=vmax)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Elastic Constant (GPa)", fontsize=12)

        # Set labels
        voigt_labels = ["1", "2", "3", "4", "5", "6"]
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels(voigt_labels, fontsize=11)
        ax.set_yticklabels(voigt_labels, fontsize=11)
        ax.set_xlabel("Voigt Index j", fontsize=12, fontweight="bold")
        ax.set_ylabel("Voigt Index i", fontsize=12, fontweight="bold")

        # Add values as text
        for i in range(6):
            for j in range(6):
                value = tensor[i, j]
                # Choose text color based on background
                text_color = "white" if abs(value) > vmax * 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                    fontweight="bold",
                )

        # Highlight diagonal elements
        for i in range(6):
            rect = plt.Rectangle(
                (i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="gold", linewidth=2
            )
            ax.add_patch(rect)

        ax.set_title(
            f"Elastic Tensor Cᵢⱼ (GPa) - {formula}\nIEEE Format (Voigt Notation)",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()
        output_file = output_path / f"elastic_tensor_heatmap_{timestamp}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        return str(output_file)

    except Exception as e:
        print(f"Warning: Could not create elastic tensor heatmap: {e}")  # noqa: T201
        return None


def plot_mechanical_properties_bar(
    k_vrh: float | None,
    g_vrh: float | None,
    y_mod: float | None,
    poisson: float | None,
    formula: str,
    output_path: Path,
    timestamp: str,
) -> str | None:
    """
    Plot mechanical properties as bar chart with reference materials.

    Parameters
    ----------
    k_vrh : float or None
        Bulk modulus (GPa)
    g_vrh : float or None
        Shear modulus (GPa)
    y_mod : float or None
        Young's modulus (Pa)
    poisson : float or None
        Poisson's ratio
    formula : str
        Chemical formula
    output_path : Path
        Output directory
    timestamp : str
        Timestamp for filename

    Returns
    -------
    str or None
        Path to saved plot file, or None if failed

    References
    ----------
    Reference material data from:

    - Simmons, G. & Wang, H. Single Crystal Elastic Constants and
      Calculated Aggregate Properties: A Handbook. MIT Press (1971).
    - de Jong, M. et al. Sci. Data 2, 150009 (2015). DOI: 10.1038/sdata.2015.9
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Reference materials for comparison (GPa)
        # Sources: Simmons & Wang (1971), Materials Project database
        references = {
            "Diamond": {"K": 442, "G": 535, "E": 1050},
            "Steel": {"K": 160, "G": 79, "E": 200},
            "Aluminum": {"K": 76, "G": 26, "E": 70},
            "Silicon": {"K": 98, "G": 67, "E": 163},
            "Copper": {"K": 137, "G": 46, "E": 117},
            "Gold": {"K": 180, "G": 27, "E": 78},
        }

        fig, axes = plt.subplots(1, 3, figsize=(15, 6))

        # Convert Young's modulus to GPa
        e_gpa = y_mod / 1e9 if y_mod is not None else None

        # Prepare data
        properties = [
            ("Bulk Modulus K", "K", k_vrh, "tab:blue"),
            ("Shear Modulus G", "G", g_vrh, "tab:green"),
            ("Young's Modulus E", "E", e_gpa, "tab:red"),
        ]

        for ax, (title, key, value, color) in zip(axes, properties, strict=False):
            if value is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=20)
                ax.set_title(title, fontsize=12, fontweight="bold")
                continue

            # Current material + references
            materials = [formula] + list(references.keys())
            values = [value] + [references[mat][key] for mat in references]

            # Create bars
            x = np.arange(len(materials))
            colors = [color] + ["lightgray"] * len(references)

            bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)

            # Highlight the calculated material
            bars[0].set_edgecolor(color)
            bars[0].set_linewidth(2)

            ax.set_xticks(x)
            ax.set_xticklabels(materials, rotation=45, ha="right", fontsize=9)
            ax.set_ylabel("GPa", fontsize=11)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)

            # Add value labels
            for bar, val in zip(bars, values, strict=False):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.02,
                    f"{val:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.suptitle(
            f"Mechanical Properties Comparison - {formula}",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )

        # Add citation footnote
        fig.text(
            0.5,
            -0.02,
            "Reference data: Simmons & Wang (1971), Materials Project database",
            ha="center",
            fontsize=8,
            style="italic",
            color="gray",
        )

        plt.tight_layout()

        output_file = output_path / f"mechanical_properties_bar_{timestamp}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        return str(output_file)

    except Exception as e:
        print(f"Warning: Could not create mechanical properties bar chart: {e}")  # noqa: T201
        return None


def plot_stress_strain_curves(
    deformations: list,
    stresses: list,
    formula: str,
    output_path: Path,
    timestamp: str,
) -> str | None:
    """
    Plot stress-strain curves for each deformation type.

    Parameters
    ----------
    deformations : list
        List of deformation matrices
    stresses : list
        List of stress tensors
    formula : str
        Chemical formula
    output_path : Path
        Output directory
    timestamp : str
        Timestamp for filename

    Returns
    -------
    str or None
        Path to saved plot file, or None if failed
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        if not deformations or not stresses:
            return None

        _fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        # Voigt notation mapping
        voigt_labels = ["σ₁₁", "σ₂₂", "σ₃₃", "σ₂₃", "σ₁₃", "σ₁₂"]
        strain_labels = ["ε₁₁", "ε₂₂", "ε₃₃", "ε₂₃", "ε₁₃", "ε₁₂"]

        # Group deformations by type
        strain_stress_data: dict[int, dict[str, list]] = {
            i: {"strains": [], "stresses": []} for i in range(6)
        }

        for deform, stress in zip(deformations, stresses, strict=False):
            deform_arr = np.array(deform)
            stress_arr = np.array(stress)

            # Convert deformation matrix to strain (Green-Lagrange)
            # ε = 0.5 * (F^T F - I)
            strain = 0.5 * (deform_arr.T @ deform_arr - np.eye(3))

            # Voigt notation: [11, 22, 33, 23, 13, 12]
            strain_voigt = [
                strain[0, 0],
                strain[1, 1],
                strain[2, 2],
                2 * strain[1, 2],
                2 * strain[0, 2],
                2 * strain[0, 1],
            ]
            stress_voigt = [
                stress_arr[0, 0],
                stress_arr[1, 1],
                stress_arr[2, 2],
                stress_arr[1, 2],
                stress_arr[0, 2],
                stress_arr[0, 1],
            ]

            # Find dominant strain component
            dominant_idx = np.argmax(np.abs(strain_voigt))

            strain_stress_data[dominant_idx]["strains"].append(
                strain_voigt[dominant_idx]
            )
            strain_stress_data[dominant_idx]["stresses"].append(
                stress_voigt[dominant_idx]
            )

        colors = plt.cm.tab10(np.linspace(0, 1, 6))

        for i, ax in enumerate(axes):
            data = strain_stress_data[i]
            if data["strains"]:
                strains = np.array(data["strains"])
                stresses_i = np.array(data["stresses"])

                # Sort by strain
                sort_idx = np.argsort(strains)
                strains = strains[sort_idx]
                stresses_i = stresses_i[sort_idx]

                ax.scatter(strains * 100, stresses_i, color=colors[i], s=60, zorder=3)
                ax.plot(
                    strains * 100, stresses_i, color=colors[i], linewidth=1.5, alpha=0.7
                )

                # Linear fit to estimate elastic constant
                if len(strains) > 1:
                    coeffs = np.polyfit(strains, stresses_i, 1)
                    fit_line = np.polyval(coeffs, strains)
                    ax.plot(
                        strains * 100,
                        fit_line,
                        "--",
                        color="gray",
                        linewidth=1,
                        alpha=0.7,
                        label=f"Slope: {coeffs[0]:.1f} GPa",
                    )
                    ax.legend(fontsize=9, loc="best")

            ax.set_xlabel(f"Strain {strain_labels[i]} (%)", fontsize=11)
            ax.set_ylabel(f"Stress {voigt_labels[i]} (GPa)", fontsize=11)
            ax.set_title(
                f"Component {i + 1}: {voigt_labels[i]} vs {strain_labels[i]}",
                fontsize=11,
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3)  # noqa: FBT003
            ax.axhline(y=0, color="black", linewidth=0.5)
            ax.axvline(x=0, color="black", linewidth=0.5)

        plt.suptitle(
            f"Stress-Strain Curves - {formula}\n"
            "Linear elastic response under applied deformations",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        output_file = output_path / f"stress_strain_curves_{timestamp}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        return str(output_file)

    except Exception as e:
        print(f"Warning: Could not create stress-strain curves: {e}")  # noqa: T201
        return None


def plot_youngs_modulus_3d(
    ieee_tensor: list,
    formula: str,
    output_path: Path,
    timestamp: str,
) -> str | None:
    """
    Plot 3D surface of directional Young's modulus.

    The directional Young's modulus E(n) for a unit vector n is:
    1/E(n) = Sᵢⱼₖₗ nᵢ nⱼ nₖ nₗ

    where S is the compliance tensor (inverse of elastic tensor).

    Parameters
    ----------
    ieee_tensor : list
        6×6 elastic tensor in IEEE format (GPa)
    formula : str
        Chemical formula
    output_path : Path
        Output directory
    timestamp : str
        Timestamp for filename

    Returns
    -------
    str or None
        Path to saved plot file, or None if failed
    """  # noqa: RUF002
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        tensor = np.array(ieee_tensor)

        # Check for singularity
        if np.linalg.det(tensor) == 0:
            print("Warning: Elastic tensor is singular, cannot compute compliance")  # noqa: T201
            return None

        # Compliance tensor (inverse of stiffness)
        compliance = np.linalg.inv(tensor)

        # Create spherical grid
        n_points = 50
        theta = np.linspace(0, np.pi, n_points)
        phi = np.linspace(0, 2 * np.pi, n_points)
        theta_grid, phi_grid = np.meshgrid(theta, phi)

        # Direction vectors
        n1 = np.sin(theta_grid) * np.cos(phi_grid)
        n2 = np.sin(theta_grid) * np.sin(phi_grid)
        n3 = np.cos(theta_grid)

        # Calculate directional Young's modulus
        # Using Voigt notation transformation
        E_dir = np.zeros_like(theta_grid)  # noqa: N806

        for i in range(n_points):
            for j in range(n_points):
                n = np.array([n1[i, j], n2[i, j], n3[i, j]])

                # Transform to Voigt indices
                # S'1111 = Sijkl ni nj nk nl
                s_inv = (
                    compliance[0, 0] * n[0] ** 4
                    + compliance[1, 1] * n[1] ** 4
                    + compliance[2, 2] * n[2] ** 4
                    + 2 * compliance[0, 1] * n[0] ** 2 * n[1] ** 2
                    + 2 * compliance[0, 2] * n[0] ** 2 * n[2] ** 2
                    + 2 * compliance[1, 2] * n[1] ** 2 * n[2] ** 2
                    + compliance[3, 3] * n[1] ** 2 * n[2] ** 2
                    + compliance[4, 4] * n[0] ** 2 * n[2] ** 2
                    + compliance[5, 5] * n[0] ** 2 * n[1] ** 2
                )

                if s_inv > 0:
                    E_dir[i, j] = 1.0 / s_inv
                else:
                    E_dir[i, j] = 0

        # Scale for 3D surface (radial distance = E)
        x = E_dir * n1
        y = E_dir * n2
        z = E_dir * n3

        # Create figure with two views
        fig = plt.figure(figsize=(14, 6))

        # 3D surface view
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.plot_surface(
            x,
            y,
            z,
            facecolors=plt.cm.viridis(E_dir / E_dir.max()),
            alpha=0.8,
            antialiased=True,
        )
        ax1.set_xlabel("X (GPa)", fontsize=10)
        ax1.set_ylabel("Y (GPa)", fontsize=10)
        ax1.set_zlabel("Z (GPa)", fontsize=10)
        ax1.set_title("3D Young's Modulus Surface", fontsize=12, fontweight="bold")

        # Add reference axes
        max_e = E_dir.max() * 1.1
        ax1.plot([0, max_e], [0, 0], [0, 0], "r-", linewidth=2, label="[100]")
        ax1.plot([0, 0], [0, max_e], [0, 0], "g-", linewidth=2, label="[010]")
        ax1.plot([0, 0], [0, 0], [0, max_e], "b-", linewidth=2, label="[001]")

        # 2D polar plots (cross-sections)
        ax2 = fig.add_subplot(122, projection="polar")

        # XY plane (theta = pi/2)
        mid_idx = n_points // 2
        e_xy = E_dir[mid_idx, :]
        ax2.plot(phi, e_xy, "b-", linewidth=2, label="XY plane (z=0)")

        # XZ plane (phi = 0)
        e_xz = E_dir[:, 0]
        ax2.plot(theta, e_xz, "r-", linewidth=2, label="XZ plane (y=0)")

        # YZ plane (phi = pi/2)
        phi_idx = n_points // 4
        e_yz = E_dir[:, phi_idx]
        ax2.plot(theta, e_yz, "g-", linewidth=2, label="YZ plane (x=0)")

        ax2.set_title("Polar Cross-Sections", fontsize=12, fontweight="bold")
        ax2.legend(loc="upper right", fontsize=9)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
        sm.set_array(E_dir)
        cbar = plt.colorbar(sm, ax=ax1, shrink=0.6, pad=0.1)
        cbar.set_label("E (GPa)", fontsize=10)

        plt.suptitle(
            f"Directional Young's Modulus - {formula}\n"
            f"E_min = {E_dir.min():.1f} GPa, E_max = {E_dir.max():.1f} GPa, "
            f"Anisotropy = {E_dir.max() / E_dir.min():.2f}",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()

        output_file = output_path / f"youngs_modulus_3d_{timestamp}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        return str(output_file)

    except Exception as e:
        print(f"Warning: Could not create 3D Young's modulus plot: {e}")  # noqa: T201
        return None


def plot_linear_compressibility_3d(
    ieee_tensor: list,
    formula: str,
    output_path: Path,
    timestamp: str,
) -> str | None:
    """
    Plot 3D surface of directional linear compressibility.

    Linear compressibility β(n) = Sᵢⱼₖₖ nᵢ nⱼ
    Negative linear compressibility (NLC) is a rare and interesting property.

    Parameters
    ----------
    ieee_tensor : list
        6×6 elastic tensor in IEEE format (GPa)
    formula : str
        Chemical formula
    output_path : Path
        Output directory
    timestamp : str
        Timestamp for filename

    Returns
    -------
    str or None
        Path to saved plot file, or None if failed
    """  # noqa: RUF002
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        tensor = np.array(ieee_tensor)

        # Check for singularity
        if np.linalg.det(tensor) == 0:
            print("Warning: Elastic tensor is singular, cannot compute compliance")  # noqa: T201
            return None

        # Compliance tensor
        compliance = np.linalg.inv(tensor)

        # Create spherical grid
        n_points = 50
        theta = np.linspace(0, np.pi, n_points)
        phi = np.linspace(0, 2 * np.pi, n_points)
        theta_grid, phi_grid = np.meshgrid(theta, phi)

        # Direction vectors
        n1 = np.sin(theta_grid) * np.cos(phi_grid)
        n2 = np.sin(theta_grid) * np.sin(phi_grid)
        n3 = np.cos(theta_grid)

        # Calculate directional linear compressibility
        # β(n) = S11 + S12 + S13 for direction [100], etc.
        # General: β(n) = Sij ni nj (summed over k for Siikk)
        beta = np.zeros_like(theta_grid)

        # Sum of compliance over hydrostatic component
        for i in range(n_points):
            for j in range(n_points):
                n = np.array([n1[i, j], n2[i, j], n3[i, j]])

                # β = ni nj (Si1 + Si2 + Si3) in Voigt notation
                beta_val = (
                    n[0] ** 2 * (compliance[0, 0] + compliance[0, 1] + compliance[0, 2])
                    + n[1] ** 2
                    * (compliance[1, 0] + compliance[1, 1] + compliance[1, 2])
                    + n[2] ** 2
                    * (compliance[2, 0] + compliance[2, 1] + compliance[2, 2])
                    + 2
                    * n[0]
                    * n[1]
                    * (compliance[5, 0] + compliance[5, 1] + compliance[5, 2])
                    + 2
                    * n[0]
                    * n[2]
                    * (compliance[4, 0] + compliance[4, 1] + compliance[4, 2])
                    + 2
                    * n[1]
                    * n[2]
                    * (compliance[3, 0] + compliance[3, 1] + compliance[3, 2])
                )
                beta[i, j] = beta_val * 1000  # Convert to TPa⁻¹

        # Check for negative linear compressibility (NLC)
        has_nlc = np.any(beta < 0)

        # Use absolute value for radial plot, color by sign
        beta_abs = np.abs(beta)

        # Scale for 3D surface
        x = beta_abs * n1
        y = beta_abs * n2
        z = beta_abs * n3

        # Create figure
        fig = plt.figure(figsize=(14, 6))

        # 3D surface view
        ax1 = fig.add_subplot(121, projection="3d")

        # Color by sign: positive = blue, negative = red
        colors = np.where(beta >= 0, beta, 0)
        colors_neg = np.where(beta < 0, -beta, 0)
        combined_colors = colors - colors_neg

        norm = plt.Normalize(vmin=combined_colors.min(), vmax=combined_colors.max())
        ax1.plot_surface(
            x,
            y,
            z,
            facecolors=plt.cm.coolwarm(norm(combined_colors)),
            alpha=0.8,
            antialiased=True,
        )

        ax1.set_xlabel("X (TPa⁻¹)", fontsize=10)
        ax1.set_ylabel("Y (TPa⁻¹)", fontsize=10)
        ax1.set_zlabel("Z (TPa⁻¹)", fontsize=10)
        ax1.set_title("3D Linear Compressibility", fontsize=12, fontweight="bold")

        # 2D polar plots
        ax2 = fig.add_subplot(122, projection="polar")

        mid_idx = n_points // 2
        beta_xy = beta[mid_idx, :]
        ax2.plot(phi, np.abs(beta_xy), "b-", linewidth=2, label="XY plane")

        beta_xz = beta[:, 0]
        ax2.plot(theta, np.abs(beta_xz), "r-", linewidth=2, label="XZ plane")

        phi_idx = n_points // 4
        beta_yz = beta[:, phi_idx]
        ax2.plot(theta, np.abs(beta_yz), "g-", linewidth=2, label="YZ plane")

        ax2.set_title("Polar Cross-Sections", fontsize=12, fontweight="bold")
        ax2.legend(loc="upper right", fontsize=9)

        # NLC warning
        nlc_text = ""
        if has_nlc:
            nlc_text = " ⚠️ NLC DETECTED!"

        plt.suptitle(
            f"Directional Linear Compressibility - {formula}{nlc_text}\n"
            f"β_min = {beta.min():.2f} TPa⁻¹, β_max = {beta.max():.2f} TPa⁻¹",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()

        output_file = output_path / f"linear_compressibility_3d_{timestamp}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        return str(output_file)

    except Exception as e:
        print(f"Warning: Could not create 3D compressibility plot: {e}")  # noqa: T201
        return None


def plot_pugh_ratio_diagram(
    k_vrh: float,
    g_vrh: float,
    formula: str,
    output_path: Path,
    timestamp: str,
) -> str | None:
    """
    Plot Pugh's ratio diagram for ductile/brittle classification.

    Pugh's criterion: K/G > 1.75 indicates ductile behavior.

    Parameters
    ----------
    k_vrh : float
        Bulk modulus VRH (GPa)
    g_vrh : float
        Shear modulus VRH (GPa)
    formula : str
        Chemical formula
    output_path : Path
        Output directory
    timestamp : str
        Timestamp for filename

    Returns
    -------
    str or None
        Path to saved plot file, or None if failed

    References
    ----------
    Reference material data from:

    - de Jong, M. et al. Sci. Data 2, 150009 (2015). DOI: 10.1038/sdata.2015.9
      (Materials Project elastic constants database)
    - Simmons, G. & Wang, H. Single Crystal Elastic Constants and
      Calculated Aggregate Properties: A Handbook. MIT Press (1971).
    - Every, A.G. & McCurdy, A.K. Landolt-Börnstein, Group III, Vol. 29a,
      Second and Higher Order Elastic Constants. Springer (1992).
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(10, 8))

        # Reference materials with elastic constants (GPa)
        # Sources:
        # - Diamond: McSkimin & Andreatch, J. Appl. Phys. 43, 2944 (1972)
        # - Si: Hall, Phys. Rev. 161, 756 (1967)
        # - Al2O3: Wachtman et al., J. Am. Ceram. Soc. 43, 334 (1960)
        # - SiC: Kamitani et al., J. Appl. Phys. 82, 3152 (1997)
        # - Fe, Cu, Al, Au, Ag, Pb: Simmons & Wang, MIT Press (1971)
        # - Also cross-referenced with Materials Project (materialsproject.org)
        references: dict[str, dict[str, Any]] = {
            "Diamond": {"K": 442, "G": 535, "type": "brittle"},
            "Si": {"K": 98, "G": 67, "type": "brittle"},
            "Al₂O₃": {"K": 252, "G": 163, "type": "brittle"},
            "SiC": {"K": 226, "G": 196, "type": "brittle"},
            "Fe": {"K": 170, "G": 82, "type": "ductile"},
            "Cu": {"K": 137, "G": 46, "type": "ductile"},
            "Al": {"K": 76, "G": 26, "type": "ductile"},
            "Au": {"K": 180, "G": 27, "type": "ductile"},
            "Ag": {"K": 103, "G": 30, "type": "ductile"},
            "Pb": {"K": 46, "G": 6, "type": "ductile"},
        }

        # Plot reference materials
        for name, data in references.items():
            marker = "s" if data["type"] == "brittle" else "o"
            color = "tab:red" if data["type"] == "brittle" else "tab:blue"
            ax.scatter(
                data["G"],
                data["K"],
                marker=marker,
                s=100,
                c=color,
                alpha=0.6,
                edgecolors="black",
                linewidths=0.5,
            )
            ax.annotate(
                name,
                (data["G"], data["K"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
            )

        # Plot calculated material
        pugh_ratio = k_vrh / g_vrh if g_vrh != 0 else float("inf")
        is_ductile = pugh_ratio > 1.75

        marker_calc = "o" if is_ductile else "s"
        color_calc = "tab:blue" if is_ductile else "tab:red"

        ax.scatter(
            g_vrh,
            k_vrh,
            marker=marker_calc,
            s=300,
            c=color_calc,
            edgecolors="black",
            linewidths=2,
            zorder=5,
            label=f"{formula} (K/G = {pugh_ratio:.2f})",
        )
        ax.annotate(
            formula,
            (g_vrh, k_vrh),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="black"),
        )

        # Ductile/Brittle boundary line (K/G = 1.75)
        g_range = np.linspace(0, max(g_vrh * 1.5, 600), 100)
        k_boundary = 1.75 * g_range
        ax.plot(
            g_range, k_boundary, "k--", linewidth=2, label="K/G = 1.75 (Pugh boundary)"
        )

        # Shade regions
        ax.fill_between(
            g_range,
            k_boundary,
            k_boundary.max() * 1.5,
            alpha=0.1,
            color="blue",
            label="Ductile region",
        )
        ax.fill_between(
            g_range, 0, k_boundary, alpha=0.1, color="red", label="Brittle region"
        )

        # Labels and legend
        ax.set_xlabel("Shear Modulus G (GPa)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Bulk Modulus K (GPa)", fontsize=12, fontweight="bold")

        # Set limits
        ax.set_xlim(0, max(g_vrh * 1.5, max(d["G"] for d in references.values()) * 1.2))
        ax.set_ylim(0, max(k_vrh * 1.5, max(d["K"] for d in references.values()) * 1.2))

        # Add text for prediction
        prediction = "DUCTILE" if is_ductile else "BRITTLE"
        prediction_color = "blue" if is_ductile else "red"
        ax.text(
            0.98,
            0.02,
            f"Prediction: {prediction}\nK/G = {pugh_ratio:.2f}",
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            color=prediction_color,
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)  # noqa: FBT003

        ax.set_title(
            f"Pugh's Ductility Criterion - {formula}\n"
            "K/G > 1.75 → Ductile | K/G < 1.75 → Brittle",
            fontsize=13,
            fontweight="bold",
        )

        # Add citation footnote
        fig.text(
            0.5,
            0.01,
            "Reference data: Simmons & Wang (1971), Materials Project, "
            "Pugh criterion: Philos. Mag. 45, 823 (1954)",
            ha="center",
            fontsize=8,
            style="italic",
            color="gray",
        )

        plt.tight_layout(rect=[0, 0.03, 1, 1])  # Leave room for footnote

        output_file = output_path / f"pugh_ratio_diagram_{timestamp}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        return str(output_file)

    except Exception as e:
        print(f"Warning: Could not create Pugh's ratio diagram: {e}")  # noqa: T201
        return None
