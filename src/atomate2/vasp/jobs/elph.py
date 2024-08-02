"""Jobs for performing electron phonon calculations in VASP."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from jobflow import Flow, Response, job

from atomate2.vasp.jobs.base import BaseVaspMaker, vasp_job
from atomate2.vasp.jobs.core import TransmuterMaker
from atomate2.vasp.schemas.elph import ElectronPhononRenormalisationDoc
from atomate2.vasp.sets.core import ElectronPhononSetGenerator

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core import Structure
    from pymatgen.electronic_structure.bandstructure import BandStructure


DEFAULT_ELPH_TEMPERATURES = (0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
DEFAULT_MIN_SUPERCELL_LENGTH = 15
logger = logging.getLogger(__name__)


@dataclass
class SupercellElectronPhononDisplacedStructureMaker(TransmuterMaker):
    """
    Maker to run electron phonon VASP jobs to generate displaced structures.

    This job:

    1. Generates a close to cubic supercell with cell lengths > 15 Å.
    2. Performs an IBRION = 6 finite-displacement calculation to calculate the phonon
       eigenvalues and eigenvectors.
    3. Displaces the atoms to simulate a range of temperatures.

    .. warning::
        Electron phonon properties should be converged with respect to supercell size.
        Typically, cells with all lattice vectors greater than 15 Å should be a
        reasonable starting point.

    .. note::
        The input structure should be well relaxed to avoid imaginary modes. For
        example, using :obj:`TightRelaxMaker`.

    .. note::
        Requires VASP 6.0 and higher. See https://www.vasp.at/wiki/index.php/Electron-
        phonon_interactions_from_Monte-Carlo_sampling
        for more details.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .VaspInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_vasp_input_set`.
    copy_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_vasp_outputs`.
    run_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_vasp`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDoc.from_directory`.
    stop_children_kwargs : dict
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data : dict
        Additional data to write to the current directory. Given as a dict of
        {filename: data}. Note that if using FireWorks, dictionary keys cannot contain
        the "." character which is typically used to denote file extensions. To avoid
        this, use the ":" character, which will automatically be converted to ".". E.g.
        ``{"my_file:txt": "contents of the file"}``.
    """

    name: str = "supercell electron phonon displacements"
    input_set_generator: ElectronPhononSetGenerator = field(
        default_factory=ElectronPhononSetGenerator
    )
    transformations: tuple[str, ...] = ("SupercellTransformation",)
    transformation_params: tuple[dict, ...] = None
    temperatures: tuple[float, ...] = DEFAULT_ELPH_TEMPERATURES
    min_supercell_length: float = DEFAULT_MIN_SUPERCELL_LENGTH

    @vasp_job
    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
    ) -> Response:
        """Run a transmuter VASP job.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.
        """
        dim = self.min_supercell_length / np.array(structure.lattice.abc)
        scaling_matrix = np.diag(np.ceil(dim).astype(int)).tolist()
        if self.transformation_params is None:
            # only overwrite transformation params if it is not set
            self.transformation_params = ({"scaling_matrix": scaling_matrix},)

        # update temperatures
        self.input_set_generator.temperatures = self.temperatures

        return super().make.original(self, structure, prev_dir)


@job
def run_elph_displacements(
    temperatures: list[float],
    structures: list[Structure],
    vasp_maker: BaseVaspMaker,
    prev_dir: str | Path | None = None,
    original_structure: Structure = None,
    supercell_structure: Structure = None,
) -> Response:
    """
    Run electron phonon displaced structures.

    Note, this job will replace itself with N displacement calculations.

    Parameters
    ----------
    temperatures : list of float
        Temperatures at which electron phonon structures were generated.
    structures : list of Structure
        Electron phonon displaced structures for each temperature.
    vasp_maker : BaseVaspMaker
        A maker to generate VASP calculations on the displaced structures.
    prev_dir : str or Path or None
        A previous VASP directory to use for copying VASP outputs.
    original_structure : Structure
        The original structure before supercell is made and before electron phonon
        displacements.
    """
    if len(temperatures) != len(structures):
        raise ValueError(
            f"Number of temperatures ({len(temperatures)}) does not equal number of "
            f"structures ({len(structures)})."
        )

    jobs = []
    outputs: dict[str, list] = {
        "temperatures": [],
        "band_structures": [],
        "structures": [],
        "uuids": [],
        "dirs": [],
    }
    for temp, structure in zip(temperatures, structures):
        # create the job
        elph_job = vasp_maker.make(structure, prev_dir=prev_dir)
        elph_job.append_name(f" T={temp}")

        # write details of the electron phonon temperature and structure elph_info.json
        # file. this file will automatically get added to the task document and allow
        # the elph builder to reconstruct the elph document. note the ":" is
        # automatically converted to a "." in the filename.
        info = {
            "temperature": temp,
            "original_structure": original_structure,
            "supercell_structure": supercell_structure,
        }
        elph_job.update_maker_kwargs(
            {"_set": {"write_additional_data->elph_info:json": info}}, dict_mod=True
        )

        jobs.append(elph_job)

        # extract the outputs we want
        outputs["temperatures"].append(temp)
        outputs["band_structures"].append(elph_job.output.vasp_objects["bandstructure"])
        outputs["structures"].append(elph_job.output.structure)
        outputs["dirs"].append(elph_job.output.dir_name)
        outputs["uuids"].append(elph_job.output.uuid)

    disp_flow = Flow(jobs, outputs)
    return Response(replace=disp_flow)


@job(output_schema=ElectronPhononRenormalisationDoc)
def calculate_electron_phonon_renormalisation(
    temperatures: list[float],
    displacement_band_structures: list[BandStructure],
    displacement_structures: list[Structure],
    displacement_uuids: list[str],
    displacement_dirs: list[str],
    bulk_band_structure: BandStructure,
    bulk_structure: Structure,
    bulk_uuid: str,
    bulk_dir: str,
    elph_uuid: str,
    elph_dir: str,
    original_structure: Structure,
) -> ElectronPhononRenormalisationDoc:
    """
    Calculate the electron-phonon renormalisation of the band gap.

    Parameters
    ----------
    temperatures : list of float
        The temperatures at which electron phonon properties were calculated.
    displacement_band_structures : list of BandStructure
        The electron-phonon displaced band structures.
    displacement_structures : list of Structure
        The electron-phonon displaced structures.
    displacement_uuids : list of str
        The UUIDs of the electron-phonon displaced band structure calculations.
    displacement_dirs : list of str
        The calculation directories of the electron-phonon displaced band structure
        calculations.
    bulk_band_structure : BandStructure
        The band structure of the bulk undisplaced supercell calculation.
    bulk_structure : Structure
        The structure of the bulk undisplaced supercell.
    bulk_uuid : str
        The UUID of the bulk undisplaced supercell band structure calculation.
    bulk_dir : str
        The directory of the bulk undisplaced supercell band structure calculation.
    elph_uuid : str
        The UUID of electron-phonon calculation that generated the displaced structures.
    elph_dir : str
        The directory of electron-phonon calculation that generated the displaced
        structures.
    original_structure : Structure
        The original primitive structure for which electron-phonon calculations
        were performed.
    """
    if bulk_structure is None:
        raise ValueError(
            "Bulk (undisplaced) supercell band structure calculation failed. Cannot "
            "calculate electron-phonon renormalisation."
        )

    # filter band structures that are None (i.e., the displacement calculation failed)
    keep = [idx for idx, b in enumerate(displacement_band_structures) if b is not None]
    temperatures = [temperatures[i] for i in keep]
    displacement_band_structures = [displacement_band_structures[i] for i in keep]
    displacement_structures = [displacement_structures[i] for i in keep]
    displacement_uuids = [displacement_uuids[i] for i in keep]
    displacement_dirs = [displacement_dirs[i] for i in keep]

    logger.info("Calculating electron-phonon renormalisation")

    return ElectronPhononRenormalisationDoc.from_band_structures(
        temperatures,
        displacement_band_structures,
        displacement_structures,
        displacement_uuids,
        displacement_dirs,
        bulk_band_structure,
        bulk_structure,
        bulk_uuid,
        bulk_dir,
        elph_uuid,
        elph_dir,
        original_structure,
    )
