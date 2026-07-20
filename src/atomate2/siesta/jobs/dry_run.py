"""Core dry-run infrastructure for previewing workflows without calculations.

This module provides the fundamental building blocks for dry-run mode across
all SIESTA jobs and flows. Dry-run mode allows users to:

- Generate and save structures without running expensive SIESTA calculations
- Preview workflow outputs before committing computational resources
- Verify structure transformations and parameters
- Rapidly iterate on workflow design

All job-level dry-runs use these reusable functions to maintain consistency.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from jobflow import job

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


@job
def dry_run_save_structure(
    structure: Structure,
    output_dir: str | Path = "dry_run_output",
    output_format: str = "cif",
    label: str = "structure",
    metadata: dict | None = None,
) -> dict:
    """
    Save structure to file without running calculations (fundamental dry-run operation).

    This is the atomic unit of dry-run operations. All job-level dry-runs
    use this function to save their output structures consistently.

    Parameters
    ----------
    structure : Structure
        Pymatgen Structure object to save.
    output_dir : str | Path
        Directory to save structure file (default: "dry_run_output").
    output_format : str
        Output format: 'cif', 'xyz', 'xsf', 'POSCAR', 'json', etc.
        Any format supported by pymatgen Structure.to() (default: "cif").
    label : str
        Descriptive label for filename (default: "structure").
    metadata : dict, optional
        Additional metadata to include in output (e.g., maker_name, tier, etc.).

    Returns
    -------
    dict
        Dry-run output containing:
        - dry_run: bool (always True)
        - label: str (structure label)
        - structure_file: str (path to saved file)
        - formula: str (reduced formula)
        - num_atoms: int (number of atoms)
        - lattice: dict (lattice parameters)
        - metadata: dict (user-provided metadata)

    Examples
    --------
    Save a structure in CIF format:
        >>> from pymatgen.core import Structure
        >>> structure = Structure.from_file("input.cif")
        >>> job = dry_run_save_structure(structure, label="relaxation_input")
        >>> result = run_locally(job)

    Save with custom metadata:
        >>> job = dry_run_save_structure(
        ...     structure,
        ...     label="eos_volume_0.95",
        ...     metadata={"volume_scale": 0.95, "maker": "EOSMaker"},
        ... )

    Save in XYZ format:
        >>> job = dry_run_save_structure(structure, output_format="xyz")
    """
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create filename
    filename = output_dir / f"{label}.{output_format}"

    # Save structure
    try:
        if output_format == "cif":
            from atomate2.siesta.sets.utils.structure_io import write_cif_with_ghost

            write_cif_with_ghost(structure, filename)
        else:
            structure.to(filename=str(filename), fmt=output_format)
        logger.info(f"[DRY RUN] Saved {label}: {filename}")
    except Exception as e:
        logger.error(f"[DRY RUN] Failed to save {label}: {e}")
        raise

    # Collect structure information
    result = {
        "dry_run": True,
        "label": label,
        "structure_file": str(filename),
        "formula": structure.composition.reduced_formula,
        "num_atoms": len(structure),
        "lattice": {
            "a": float(structure.lattice.a),
            "b": float(structure.lattice.b),
            "c": float(structure.lattice.c),
            "alpha": float(structure.lattice.alpha),
            "beta": float(structure.lattice.beta),
            "gamma": float(structure.lattice.gamma),
            "volume": float(structure.volume),
        },
        "metadata": metadata or {},
    }

    return result


@job
def dry_run_save_multiple_structures(
    structures: list[Structure],
    output_dir: str | Path = "dry_run_output",
    output_format: str = "cif",
    label_prefix: str = "structure",
    metadata_list: list[dict] | None = None,
) -> dict:
    """
    Save multiple structures to files (for workflows with many structures).

    Used by workflows that generate many structures: EOS (multiple volumes),
    phonon (many displacements), NEB (multiple images), surfaces (many slabs), etc.

    Parameters
    ----------
    structures : list[Structure]
        List of pymatgen Structure objects to save.
    output_dir : str | Path
        Directory to save structure files (default: "dry_run_output").
    output_format : str
        Output format for all structures (default: "cif").
    label_prefix : str
        Prefix for filenames. Files named as: {label_prefix}_000.{format},
        {label_prefix}_001.{format}, etc. (default: "structure").
    metadata_list : list[dict], optional
        List of metadata dicts (one per structure). If None, empty dicts used.

    Returns
    -------
    dict
        Summary containing:
        - dry_run: bool (always True)
        - num_structures: int (number of structures saved)
        - structure_files: list[dict] (info for each structure)
        - output_dir: str (path to output directory)

    Examples
    --------
    Save EOS volumes:
        >>> structures = [scale_structure(struct, s) for s in [0.95, 1.0, 1.05]]
        >>> metadata = [{"scale": s} for s in [0.95, 1.0, 1.05]]
        >>> job = dry_run_save_multiple_structures(
        ...     structures, label_prefix="eos_volume", metadata_list=metadata
        ... )

    Save phonon displacements:
        >>> displaced = generate_displacements(structure, supercell)
        >>> job = dry_run_save_multiple_structures(
        ...     displaced, label_prefix="phonon_displacement"
        ... )
    """
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_list = metadata_list or [{}] * len(structures)

    if len(metadata_list) != len(structures):
        logger.warning(
            f"metadata_list length ({len(metadata_list)}) != "
            f"structures length ({len(structures)}). Using empty metadata."
        )
        metadata_list = [{}] * len(structures)

    structure_files = []

    for i, (structure, metadata) in enumerate(
        zip(structures, metadata_list, strict=False)
    ):
        label = f"{label_prefix}_{i:03d}"
        filename = output_dir / f"{label}.{output_format}"

        try:
            if output_format == "cif":
                from atomate2.siesta.sets.utils.structure_io import write_cif_with_ghost

                write_cif_with_ghost(structure, filename)
            else:
                structure.to(filename=str(filename), fmt=output_format)
            logger.info(f"[DRY RUN] Saved {label}: {filename}")

            structure_files.append(
                {
                    "index": i,
                    "label": label,
                    "file": str(filename),
                    "formula": structure.composition.reduced_formula,
                    "num_atoms": len(structure),
                    "volume": float(structure.volume),
                    "metadata": metadata,
                }
            )
        except Exception as e:
            logger.error(f"[DRY RUN] Failed to save {label}: {e}")
            raise

    logger.info(f"[DRY RUN] Saved {len(structure_files)} structures to {output_dir}")

    return {
        "dry_run": True,
        "num_structures": len(structures),
        "structure_files": structure_files,
        "output_dir": str(output_dir),
    }


@job
def dry_run_workflow_summary(
    job_outputs: list[dict],
    workflow_type: str,
    output_dir: str | Path = "dry_run_output",
    **workflow_metadata,
) -> dict:
    """
    Create comprehensive summary of dry-run workflow results.

    Aggregates all job outputs from a workflow and creates a human-readable
    summary file with workflow metadata and structure information.

    Parameters
    ----------
    job_outputs : list[dict]
        List of dry-run job outputs (from dry_run_save_structure calls).
    workflow_type : str
        Type of workflow (e.g., "adsorption_scan", "eos", "phonon").
    output_dir : str | Path
        Directory to save summary file (default: "dry_run_output").
    **workflow_metadata
        Additional workflow metadata to include in summary (e.g.,
        grid_size, height, supercell_matrix, etc.).

    Returns
    -------
    dict
        Summary output containing:
        - dry_run: bool (always True)
        - summary_file: str (path to summary text file)
        - num_jobs: int (number of jobs processed)
        - workflow_type: str (workflow type)
        - timestamp: str (ISO format timestamp)

    Examples
    --------
    Create summary for EOS workflow:
        >>> eos_jobs = [job1.output, job2.output, ...]
        >>> summary = dry_run_workflow_summary(
        ...     eos_jobs, workflow_type="eos", num_volumes=10, volume_range="0.90-1.10"
        ... )

    Create summary for adsorption scan:
        >>> scan_jobs = [slab.output, ads.output, *site_jobs.outputs]
        >>> summary = dry_run_workflow_summary(
        ...     scan_jobs,
        ...     workflow_type="adsorption_scan",
        ...     grid_size=(5, 5),
        ...     height=2.0,
        ...     placement="top",
        ... )
    """
    from datetime import datetime
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now()

    # Create summary file
    summary_file = output_dir / "dry_run_summary.txt"

    with open(summary_file, "w") as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write(f"DRY RUN SUMMARY: {workflow_type}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Workflow parameters
        if workflow_metadata:
            f.write("Workflow Parameters:\n")
            f.write("-" * 80 + "\n")
            f.writelines(
                f"  {key}: {val}\n" for key, val in sorted(workflow_metadata.items())
            )
            f.write("\n")

        # Job outputs
        f.write(f"Generated {len(job_outputs)} job outputs:\n")
        f.write("-" * 80 + "\n")

        for i, output in enumerate(job_outputs, 1):
            if isinstance(output, dict):
                # Handle both single structure and multiple structure outputs
                if output.get("dry_run"):
                    if "structure_files" in output:
                        # Multiple structures output
                        num_structs = output.get("num_structures", 0)
                        f.write(f"  {i:2d}. Multiple structures: {num_structs} files\n")
                    else:
                        # Single structure output
                        label = output.get("label", "unknown")
                        formula = output.get("formula", "N/A")
                        num_atoms = output.get("num_atoms", 0)
                        f.write(
                            f"  {i:2d}. {label:30s} {formula:15s} ({num_atoms:3d} atoms)\n"
                        )
                else:
                    f.write(f"  {i:2d}. Non-dry-run output (skipped)\n")
            else:
                f.write(f"  {i:2d}. Unknown output type\n")

        # Footer with instructions
        f.write("\n" + "=" * 80 + "\n")
        f.write("Next Steps:\n")
        f.write("-" * 80 + "\n")
        f.write("  1. Review structures in visualization software:\n")
        f.write("     - VESTA: File → Open → {output_dir}/*.cif\n")
        f.write("     - Avogadro: Open files from {output_dir}/\n")
        f.write("     - XCrySDen: xcrysden --xsf {output_dir}/*.xsf\n")
        f.write("\n")
        f.write("  2. Verify parameters and structure transformations\n")
        f.write("     - Check formulas, atom counts, lattice parameters\n")
        f.write("     - Verify workflow logic and transformations\n")
        f.write("\n")
        f.write("  3. If satisfied, run full workflow:\n")
        f.write("     - Set dry_run=False in your maker\n")
        f.write("     - Submit to HPC or run locally\n")
        f.write("\n")
        f.write("  4. If adjustments needed:\n")
        f.write("     - Modify parameters (height, grid_size, etc.)\n")
        f.write("     - Run dry-run again to preview changes\n")
        f.write("\n")

        # Add standard footer
        from atomate2.siesta.utils.text_output import get_standard_footer

        f.write(
            get_standard_footer(
                width=80,
                additional_info={
                    "Analysis type": "Dry-run preview",
                    "Workflow type": workflow_type,
                    "Number of jobs": str(len(job_outputs)),
                },
            )
        )

    logger.info(f"[DRY RUN] Summary saved: {summary_file}")

    return {
        "dry_run": True,
        "summary_file": str(summary_file),
        "num_jobs": len(job_outputs),
        "workflow_type": workflow_type,
        "timestamp": timestamp.isoformat(),
    }
