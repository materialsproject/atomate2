"""Jobs for the VASP Raman workflow."""

from __future__ import annotations

import contextlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from jobflow import Flow, Response, job
from monty.json import MontyEncoder
from pymatgen.core import Structure

from atomate2.common.jobs.phonons import generate_phonon_displacements
from atomate2.utils import raman as rm
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.schemas.raman import RamanDoc, RamanModeData
from atomate2.vasp.sets.raman import RamanDielectricSetGenerator

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from atomate2.vasp.sets.base import VaspInputGenerator

logger = logging.getLogger(__name__)


@dataclass
class RamanDielectricMaker(BaseVaspMaker):
    """
    Maker for LEPSILON runs of structures displaced along a normal mode.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .VaspInputGenerator
        A generator used to make the input set.
    """

    name: str = "raman dielectric"
    input_set_generator: VaspInputGenerator = field(
        default_factory=RamanDielectricSetGenerator
    )


@job(data=[Structure])
def generate_magnetic_phonon_displacements(
    structure: Structure,
    supercell_matrix: np.ndarray,
    displacement: float,
    sym_reduce: bool,
    symprec: float,
    use_symmetrized_structure: str | None,
    kpath_scheme: str,
    code: str,
) -> list[Structure]:
    """
    Generate displaced supercells that keep the magnetic moments of structure.

    Same displacements as :obj:`generate_phonon_displacements`, with the magmom
    site property of each atom copied to its images in the supercell.
    """
    displaced = generate_phonon_displacements.original(
        structure=structure,
        supercell_matrix=supercell_matrix,
        displacement=displacement,
        sym_reduce=sym_reduce,
        symprec=symprec,
        use_symmetrized_structure=use_symmetrized_structure,
        kpath_scheme=kpath_scheme,
        code=code,
    )
    if structure.site_properties.get("magmom") is None:
        return displaced
    magmoms = rm.supercell_magmoms(structure, supercell_matrix)
    for supercell in displaced:
        supercell.add_site_property("magmom", magmoms)
    return displaced


@job
def get_gamma_normal_modes(
    structure: Structure,
    force_constants: Any,
    supercell_matrix: Sequence[Sequence[float]],
    primitive_matrix: Sequence[Sequence[float]] | str | None = "auto",
    symprec: float = 1e-4,
    step_size: float = 0.005,
    mode_selection: Literal["raman_active", "all"] = "raman_active",
    modes: Sequence[int] | None = None,
    degeneracy_tolerance: float = 1e-3,
    magnetic: bool = False,
    max_imaginary_freq: float | None = 0.1,
    magnetic_structure: Structure | None = None,
) -> dict[str, Any]:
    """
    Get Gamma normal modes, select modes and build the displaced structures.

    Parameters
    ----------
    structure
        Relaxed structure used by the phonon flow.
    force_constants
        Force constants from the phonon flow.
    supercell_matrix
        Supercell matrix of the phonon flow.
    primitive_matrix
        Primitive matrix of the phonon flow (ignored if magnetic).
    symprec
        Symmetry tolerance.
    step_size
        Displacement along each mode in Angstrom.
    mode_selection
        "raman_active" or "all" optical modes. Ignored if modes is given.
    modes
        Explicit 0-based mode indices.
    degeneracy_tolerance
        Frequency tolerance in THz for degenerate modes.
    magnetic
        Use the magnetic structure (magmom site property) for the primitive cell
        and the symmetry.
    max_imaginary_freq
        If any Gamma frequency is below -max_imaginary_freq (THz), no modes are
        selected and no dielectric runs follow. None disables the check.
    magnetic_structure
        Structure with the magmom site property, used if magnetic and structure
        has no moments (the phonon document stores the structure without them).
    """
    if isinstance(structure, dict):
        structure = Structure.from_dict(structure)
    if isinstance(magnetic_structure, dict):
        magnetic_structure = Structure.from_dict(magnetic_structure)
    if (
        magnetic
        and structure.site_properties.get("magmom") is None
        and magnetic_structure is not None
    ):
        structure = rm.transfer_magmoms(structure, magnetic_structure)
    phonon = rm.build_phonopy(
        structure,
        force_constants,
        supercell_matrix,
        primitive_matrix,
        symprec,
        magnetic=magnetic,
    )
    nm = rm.gamma_normal_modes(phonon)
    nmodes = len(nm["frequencies_thz"])
    acoustic = rm.acoustic_mode_indices(nm["frequencies_thz"])
    magmoms = structure.site_properties.get("magmom") if magnetic else None
    order = rm.magnetic_order(magmoms)

    label_phonon = phonon
    if magnetic and order == "FM":
        label_phonon = rm.build_phonopy(
            structure, force_constants, supercell_matrix, "auto", symprec
        )
    rules = None
    same_cell = len(label_phonon.primitive) == nmodes // 3
    if same_cell:
        diff = (
            label_phonon.primitive.scaled_positions - phonon.primitive.scaled_positions
        )
        same_cell = np.allclose(diff - np.rint(diff), 0, atol=1e-5) and np.allclose(
            label_phonon.primitive.cell, phonon.primitive.cell
        )
    if same_cell:
        try:
            rules = rm.gamma_selection_rules(
                label_phonon, nmodes, degeneracy_tolerance, nm["eigenvecs"]
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Irreps failed (%s); computing all optical modes.", exc)
    else:
        logger.warning("Cells for labels and modes differ; computing all modes.")
    if rules is None:
        rules = {
            "raman_active": [True] * nmodes,
            "ir_active": [True] * nmodes,
            "irrep_labels": [None] * nmodes,
            "degenerate_sets": [[k] for k in range(nmodes)],
            "pointgroup": str(phonon.primitive_symmetry.pointgroup_symbol),
        }

    optical = [k for k in range(nmodes) if k not in acoustic]
    if modes is not None:
        selected = sorted({int(m) for m in modes})
        if bad := [m for m in selected if not 0 <= m < nmodes]:
            raise ValueError(f"Mode indices {bad} out of range (3N = {nmodes})")
    elif mode_selection == "raman_active":
        selected = [k for k in optical if rules["raman_active"][k]]
    elif mode_selection == "all":
        selected = optical
    else:
        raise ValueError(f"Unknown mode_selection={mode_selection!r}")

    frequencies_cm = nm["frequencies_thz"] * rm.THZ_TO_CM
    unstable = rm.gamma_instabilities(nm["frequencies_thz"], max_imaginary_freq)
    stop_reason = None
    if unstable:
        freqs = ", ".join(f"{nm['frequencies_thz'][k]:.3f}" for k in unstable)
        stop_reason = (
            f"Imaginary Gamma modes {unstable} ({freqs} THz) below "
            f"-{abs(max_imaginary_freq)} THz; no dielectric runs."
        )
        logger.error(stop_reason)
        selected = []
    for k in selected:
        if frequencies_cm[k] < 0:
            logger.warning(
                "Mode %d is UNSTABLE. Freq = %.2f cm^-1", k, frequencies_cm[k]
            )

    displaced = [
        {
            "mode": int(k),
            "displacement": float(disp),
            "structure": rm.displace_along_mode(
                nm["structure"], nm["eigenvecs"][k], nm["masses"], disp
            ),
        }
        for k in selected
        for disp in (-step_size, step_size)
    ]

    prim_magmoms = nm["structure"].site_properties.get("magmom") if magnetic else None
    return {
        "structure": nm["structure"],
        "masses": nm["masses"].tolist(),
        "frequencies_thz": nm["frequencies_thz"].tolist(),
        "frequencies_cm": frequencies_cm.tolist(),
        "eigenvals": nm["eigenvals"].tolist(),
        "eigenvecs": nm["eigenvecs"].tolist(),
        "pattern_norms": [
            rm.mode_displacement_pattern(nm["eigenvecs"][k], nm["masses"])[1]
            for k in range(nmodes)
        ],
        "acoustic_modes": acoustic,
        "selected_modes": selected,
        "step_size": step_size,
        "displaced": displaced,
        "gamma_unstable_modes": unstable,
        "max_imaginary_freq": max_imaginary_freq,
        "stop_reason": stop_reason,
        "magnetic_order": order if magnetic else None,
        "magmoms": None if prim_magmoms is None else np.asarray(prim_magmoms).tolist(),
        **rules,
    }


@job
def run_raman_displacements(
    modes_data: dict[str, Any],
    raman_maker: BaseVaspMaker | None = None,
    prev_dir: str | Path | None = None,
    bandgap: float | None = None,
    bandgap_thresholds: Sequence[float] | None = (1.0, 2.0),
    kpoints_lengths: Sequence[int] = (40, 30, 20),
) -> Response:
    """
    Run one dielectric calculation per mode and displacement.

    Parameters
    ----------
    modes_data
        Output of :obj:`get_gamma_normal_modes`.
    raman_maker
        Maker for the displaced dielectric runs.
    prev_dir
        Optional previous directory.
    bandgap
        Band gap in eV used to choose Rk of the k-mesh.
    bandgap_thresholds
        Band-gap limits in eV for the Rk rule. None keeps the Rk of raman_maker.
    kpoints_lengths
        Rk below, between and above the thresholds.
    """
    raman_maker = raman_maker or RamanDielectricMaker()
    generator = getattr(raman_maker, "input_set_generator", None)
    if bandgap_thresholds is not None and hasattr(generator, "kpoints_length"):
        rk = rm.kpoints_length_from_bandgap(
            bandgap, bandgap_thresholds, kpoints_lengths
        )
        raman_maker = replace(
            raman_maker, input_set_generator=replace(generator, kpoints_length=rk)
        )
        if bandgap is None:
            logger.warning("Band gap unknown; using Rk = %s.", rk)
    kpoints_length = getattr(
        getattr(raman_maker, "input_set_generator", None), "kpoints_length", None
    )
    jobs = []
    outputs: dict[str, list] = defaultdict(list)
    ntot = len(modes_data["displaced"])

    for idx, item in enumerate(modes_data["displaced"]):
        structure = item["structure"]
        if isinstance(structure, dict):
            structure = Structure.from_dict(structure)
        mode, disp = item["mode"], item["displacement"]

        kwargs = {"prev_dir": prev_dir} if prev_dir is not None else {}
        eps_job = raman_maker.make(structure, **kwargs)
        eps_job.append_name(f" mode {mode} disp {disp:+.4f} ({idx + 1}/{ntot})")
        info = {
            "mode": mode,
            "displacement": disp,
            "step_size": modes_data["step_size"],
        }
        with contextlib.suppress(Exception):
            eps_job.update_maker_kwargs(
                {"_set": {"write_additional_data->raman_info:json": info}},
                dict_mod=True,
            )

        jobs.append(eps_job)
        outputs["mode"].append(mode)
        outputs["displacement"].append(disp)
        outputs["epsilon"].append(
            eps_job.output.calcs_reversed[0].output.epsilon_static
        )
        outputs["uuids"].append(eps_job.output.uuid)
        outputs["dirs"].append(eps_job.output.dir_name)

    outputs = dict(outputs)
    outputs["bandgap"] = bandgap
    outputs["kpoints_length"] = kpoints_length
    if not jobs:
        return Response(output=outputs)
    return Response(replace=Flow(jobs, outputs, name="raman dielectric runs"))


@job(output_schema=RamanDoc)
def calculate_raman_tensors(
    modes_data: dict[str, Any],
    raman_data: dict[str, list],
    mp_id: str | None = None,
    lorentzian_width: float | None = 4.0,
) -> RamanDoc:
    """
    Compute Raman tensors from finite differences of the dielectric tensor.

    Parameters
    ----------
    modes_data
        Output of :obj:`get_gamma_normal_modes`.
    raman_data
        Output of :obj:`run_raman_displacements`.
    mp_id
        Optional Materials Project id.
    lorentzian_width
        Line width in cm^-1 of the stored spectrum, None to skip.
    """
    structure = modes_data["structure"]
    if isinstance(structure, dict):
        structure = Structure.from_dict(structure)
    freqs_cm = np.asarray(modes_data["frequencies_cm"])

    per_mode: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for mode, disp, eps, uuid, dir_name in zip(
        raman_data.get("mode", []),
        raman_data.get("displacement", []),
        raman_data.get("epsilon", []),
        raman_data.get("uuids", []),
        raman_data.get("dirs", []),
        strict=True,
    ):
        if eps is None:
            raise ValueError(f"No dielectric tensor for mode {mode} in {dir_name}")
        d = per_mode[int(mode)]
        d["displacements"].append(float(disp))
        d["epsilons"].append(np.asarray(eps, dtype=float).tolist())
        d["uuids"].append(uuid)
        d["dirs"].append(dir_name)

    raman_tensor: dict[str, list[list[float]]] = {}
    mode_docs: list[RamanModeData] = []
    for mode in sorted(per_mode):
        d = per_mode[mode]
        tensor = rm.raman_tensor_from_epsilons(
            d["displacements"],
            d["epsilons"],
            structure.volume,
            modes_data["pattern_norms"][mode],
        )
        raman_tensor[str(mode)] = tensor.tolist()
        mode_docs.append(
            RamanModeData(
                mode=mode,
                frequency_cm=float(freqs_cm[mode]),
                irrep=modes_data["irrep_labels"][mode],
                raman_tensor=tensor.tolist(),
                raman_activity=rm.raman_activity(tensor),
                intensity=rm.crd_intensity(tensor),
                displacements=d["displacements"],
                epsilons=d["epsilons"],
                task_uuids=d["uuids"],
                task_dirs=d["dirs"],
            )
        )

    spectrum_x = spectrum_y = None
    stable = [m for m in mode_docs if m.frequency_cm > 0]
    if lorentzian_width and stable:
        grid, spec = rm.lorentzian_spectrum(
            [m.frequency_cm for m in stable],
            [m.intensity for m in stable],
            width=lorentzian_width,
        )
        spectrum_x, spectrum_y = grid.tolist(), spec.tolist()

    stop_reason = modes_data.get("stop_reason")
    state = "unstable" if stop_reason else "successful"
    doc = RamanDoc(
        structure=structure,
        formula_pretty=structure.composition.reduced_formula,
        mp_id=mp_id,
        bandgap=raman_data.get("bandgap"),
        kpoints_length=raman_data.get("kpoints_length"),
        pointgroup=modes_data.get("pointgroup"),
        magnetic_order=modes_data.get("magnetic_order"),
        magmoms=modes_data.get("magmoms"),
        step_size=modes_data["step_size"],
        frequencies=freqs_cm.tolist(),
        frequencies_thz=modes_data["frequencies_thz"],
        normalmodes={
            "eigenvals": modes_data["eigenvals"],
            "eigenvecs": modes_data["eigenvecs"],
        },
        irrep_labels=modes_data["irrep_labels"],
        raman_active=modes_data["raman_active"],
        ir_active=modes_data["ir_active"],
        acoustic_modes=modes_data["acoustic_modes"],
        computed_modes=sorted(per_mode),
        raman_tensor=raman_tensor,
        modes=mode_docs,
        unstable_modes=[m.mode for m in mode_docs if m.frequency_cm < 0],
        gamma_unstable_modes=modes_data.get("gamma_unstable_modes", []),
        max_imaginary_freq=modes_data.get("max_imaginary_freq"),
        spectrum_frequencies=spectrum_x,
        spectrum_intensities=spectrum_y,
        lorentzian_width=lorentzian_width if spectrum_x else None,
        state=state,
        stop_reason=stop_reason,
    )

    with open("raman.json", "w") as f:
        json.dump(
            {
                "structure": structure.as_dict(),
                "formula_pretty": doc.formula_pretty,
                "normalmodes": doc.normalmodes,
                "frequencies": doc.frequencies,
                "raman_tensor": raman_tensor,
                "MP_id": mp_id,
                "state": state,
            },
            f,
            cls=MontyEncoder,
        )
    return doc
