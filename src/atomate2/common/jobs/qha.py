"""Jobs for running qha calculations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from jobflow import Flow, Response, job

from atomate2.common.schemas.phonons import PhononBSDOSDoc
from atomate2.common.schemas.qha import PhononQHADoc
from atomate2.common.utils import get_supercell_matrix

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure

    from atomate2.common.flows.phonons import BasePhononMaker

logger = logging.getLogger(__name__)


@job
def get_supercell_size(
    eos_output: dict,
    min_length: float,
    max_length: float,
    prefer_90_degrees: bool,
    allow_orthorhombic: bool = False,
    **kwargs,
) -> list[list[float]]:
    """
    Job to get the supercell size from an eos output.

    Parameters
    ----------
    eos_output: dict
        output from eos state job
    min_length: float
        minimum length of cell in Angstrom
    max_length: float
        maximum length of cell in Angstrom
    prefer_90_degrees: bool
        if True, the algorithm will try to find a cell with 90 degree angles first
    allow_orthorhombic: bool
        if True, orthorhombic supercells are allowed
    **kwargs:
        Additional parameters that can be set.
    """
    return get_supercell_matrix(
        eos_output["relax"]["structure"][0],
        min_length,
        max_length,
        prefer_90_degrees,
        allow_orthorhombic,
        **kwargs,
    )


@job(data=[PhononBSDOSDoc])
def get_phonon_jobs(
    phonon_maker: BasePhononMaker, eos_output: dict, supercell_matrix: list[list[float]]
) -> Flow:
    """
    Start all relevant phonon jobs.

    Parameters
    ----------
    phonon_maker: .BasePhononMaker
        Maker to start harmonic phonon runs.
    eos_output: dict
        Output from EOSMaker
    supercell_matrix:
        Supercell matrix to be passed into the phonon runs.
    """
    phonon_jobs = []
    outputs = []
    for istructure, structure in enumerate(eos_output["relax"]["structure"]):
        if eos_output["relax"]["dir_name"][istructure] is not None:
            phonon_job = phonon_maker.make(
                structure,
                prev_dir=eos_output["relax"]["dir_name"][istructure],
                supercell_matrix=supercell_matrix,
            )
        else:
            phonon_job = phonon_maker.make(structure, supercell_matrix=supercell_matrix)
        phonon_job.append_name(f" eos deformation {istructure + 1}")
        phonon_jobs.append(phonon_job)
        outputs.append(phonon_job.output)
    replace_flow = Flow(phonon_jobs, outputs)
    return Response(replace=replace_flow)


@job(
    output_schema=PhononQHADoc,
    data=["free_energies", "heat_capacities", "entropies", "helmholtz_volume"],
)
def analyze_free_energy(
    phonon_outputs: list[PhononBSDOSDoc],
    structure: Structure,
    t_max: float = None,
    pressure: float = None,
    ignore_imaginary_modes: bool = False,
    eos_type: str = "vinet",
    **kwargs,
) -> Flow:
    """Analyze the free energy from all phonon runs.

    Parameters
    ----------
    phonon_outputs: list[PhononBSDOSDoc]
        list of PhononBSDOSDoc objects
    structure: Structure object
        Corresponding structure object.
    t_max: float
        Max temperature for QHA in Kelvin.
    pressure: float
        Pressure for QHA in GPa.
    ignore_imaginary_modes: bool
        If True, all free energies will be used
        for EOS fit
    kwargs: dict
        Additional keywords to pass to this job
    """
    # only add free energies if there are no imaginary modes
    # tolerance has to be tested
    electronic_energies: list[list[float]] = []
    free_energies: list[list[float]] = []
    heat_capacities: list[list[float]] = []
    entropies: list[list[float]] = []
    temperatures: list[float] = []
    formula_units: list[int] = []
    volume: list[float] = [
        output.volume_per_formula_unit * output.formula_units
        for output in phonon_outputs
    ]
    supercell_matrix: list[list[float]] = phonon_outputs[0].supercell_matrix

    for itemp, temp in enumerate(phonon_outputs[0].temperatures):
        temperatures.append(float(temp))
        sorted_volume = []
        electronic_energies.append([])
        free_energies.append([])
        heat_capacities.append([])
        entropies.append([])

        for _, output in sorted(zip(volume, phonon_outputs, strict=True)):
            # check if imaginary modes
            if (not output.has_imaginary_modes) or ignore_imaginary_modes:
                electronic_energies[itemp].append(output.total_dft_energy)
                # convert from J/mol in kJ/mol
                free_energies[itemp].append(output.free_energies[itemp] / 1000.0)
                heat_capacities[itemp].append(output.heat_capacities[itemp])
                entropies[itemp].append(output.entropies[itemp])
                sorted_volume.append(output.volume_per_formula_unit)
                formula_units.append(output.formula_units)

    # potentially implement a space group check in the future

    if len(set(formula_units)) != 1:
        raise ValueError("There should be only one formula unit.")

    return PhononQHADoc.from_phonon_runs(
        volumes=sorted_volume,
        free_energies=free_energies,
        electronic_energies=electronic_energies,
        entropies=entropies,
        heat_capacities=heat_capacities,
        temperatures=temperatures,
        structure=structure,
        t_max=t_max,
        pressure=pressure,
        formula_units=next(iter(set(formula_units))),
        eos_type=eos_type,
        supercell_matrix=supercell_matrix,
        **kwargs,
    )
