"""Jobs for running qha calculations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from jobflow import Flow, Response, job

from atomate2.common.schemas.qha import PhononQHADoc

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure

    from atomate2.common.flows.phonons import BasePhononMaker
    from atomate2.common.schemas.phonons import PhononBSDOSDoc

logger = logging.getLogger(__name__)


@job
def get_phonon_jobs(phonon_maker: BasePhononMaker, eos_output: dict) -> Flow:
    """
    Start all relevant phonon jobs.

    Parameters
    ----------
    phonon_maker: .BasePhononMaker
        Maker to start harmonic phonon runs.
    eos_output: dict
        Output from EOSMaker

    """
    phonon_jobs = []
    outputs = []
    for structure in eos_output["relax"]["structure"]:
        phonon_job = phonon_maker.make(structure)
        phonon_jobs.append(phonon_job)
        outputs.append(phonon_job.output)

    return Response(replace=phonon_jobs, output=outputs)


@job(
    output_schema=PhononQHADoc,
)
def analyze_free_energy(
    phonon_outputs: list[PhononBSDOSDoc],
    structure: Structure,
    t_max: float = None,
    pressure: float = None,
    ignore_imaginary_modes: bool = False,
) -> Flow:
    """Analyze the free energy from all phonon runs.

    Parameters
    ----------
    total_dft_energy_per_formula_unit: float
        Total DFT energy in eV per formula unit.
    structure: Structure object
        Corresponding structure object.
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

    for itemp, temp in enumerate(phonon_outputs[0].temperatures):
        temperatures.append(float(temp))
        sorted_volume = []
        electronic_energies.append([])
        free_energies.append([])
        heat_capacities.append([])
        entropies.append([])

        for _, output in sorted(zip(volume, phonon_outputs)):
            # check if imaginary modes
            if (not output.has_imaginary_modes) or ignore_imaginary_modes:
                electronic_energies[itemp].append(output.total_dft_energy)
                # convert from J/mol in kJ/mol
                free_energies[itemp].append(output.free_energies[itemp] / 1000.0)
                heat_capacities[itemp].append(output.heat_capacities[itemp])
                entropies[itemp].append(output.entropies[itemp])
                sorted_volume.append(output.volume_per_formula_unit)
                formula_units.append(output.formula_units)

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
    )
