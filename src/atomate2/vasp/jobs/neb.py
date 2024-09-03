"""
Define NEB VASP jobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from os import mkdir, symlink
from pathlib import Path
from typing import TYPE_CHECKING

from custodian.vasp.validators import VasprunXMLValidator
from pymatgen.core import Structure
from pymatgen.io.vasp import Kpoints

from atomate2.vasp.jobs.base import BaseVaspMaker, vasp_job
from atomate2.vasp.sets.core import NebSetGenerator

if TYPE_CHECKING:
    from atomate2.vasp.sets.base import VaspInputGenerator

@dataclass
class NebFromImagesMaker(BaseVaspMaker):
    """
    Maker to create VASP NEB jobs.

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
    name: str = "NEB"
    input_set_generator: VaspInputGenerator = field(
        default_factory = NebSetGenerator
    )
    run_vasp_kwargs: dict = field(
        default_factory=lambda : {
            "job_type": "neb",
            "vasp_job_kwargs": {
                "output_file": "vasp.out",
                "stderr_file": "std_err.txt",
            },
        }
    )
    lclimb: bool = True
    kpoints_kludge: Kpoints | None = None

    @vasp_job
    def make(
        self,
        images: list[Structure],
        prev_dir: str | Path | None = None,
    ):

        #self.task_document_kwargs = {
        #    "outcar_file": f"{Path.cwd()}/01/OUTCAR",
        #    "contcar_file": f"{Path.cwd()}/01/CONTCAR",
        #}

        num_images = len(images) - 2
        self.input_set_generator.num_images = num_images
        
        for iimage in range(len(images)):
            image_dir = f"{iimage:02}"
            mkdir(image_dir)

            # emmet TaskDoc stores at most 9 relaxation jobs because reasons
            if iimage < 8:
                symlink(image_dir, f"relax{iimage}")
                
            images[iimage].to(f"{image_dir}/POSCAR")

            """
            Bug in VASP 6 compiled with HDF5 support:
            https://www.vasp.at/forum/viewtopic.php?f=3&t=18721&p=23430&hilit=neb+hdf5+images#p23430

            Seems like there's a validation check of whether the KPOINTS file used in the head directory
            is the same as in each IMAGE subdirectory.

            Using KSPACING in INCAR gets around this issue; a kludge to use KPOINTS is to simply copy it
            to each subdirectory, as we do here
            """
            if isinstance(self.kpoints_kludge, Kpoints):
                self.kpoints_kludge.write_file(f"{image_dir}/KPOINTS")

        return super().make.original(self, images[0], prev_dir = prev_dir)


@dataclass
class NEBMaker(BaseVaspMaker):
    name: str = "NEB"
    input_set_generator: VaspInputGenerator = field(
        default_factory= NebSetGenerator
    )
    run_vasp_kwargs: dict = field(default_factory=dict)

    @vasp_job
    def make(
        self,
        endpoint1: Structure,
        endpoint2: Structure,
        lclimb: bool = True,
        nimages: int = 1,
        interp_sort_tol: float = 0.5,
        KPOINTS_KLUDGE: Kpoints | None = None,
        **kwargs,
    ):
        # can't let user overwrite this
        self.run_vasp_kwargs = {
            "job_type": "neb",
            "vasp_job_kwargs": {
                "output_file": "vasp.out",
                "stderr_file": "std_err.txt",  # ,"auto_npar": True
            },
            "validators": (VasprunXMLValidator(), NEBFilesValidator()),
        }

        self.task_document_kwargs = {
            "outcar_file": f"{Path.cwd()}/01/OUTCAR",
            "contcar_file": f"{Path.cwd()}/01/CONTCAR",
        }

        self.input_set_generator.nimage = nimages
        self.input_set_generator.lclimb = lclimb

        images = endpoint1.interpolate(
            endpoint2, nimages=nimages + 1, autosort_tol=interp_sort_tol
        )

        for iimage in range(nimages + 2):
            tdir = f"{iimage:02}"
            mkdir(tdir)
            images[iimage].to(tdir + "/POSCAR")

            """
                Bug in VASP 6 compiled with HDF5 support:
                    https://www.vasp.at/forum/viewtopic.php?f=3&t=18721&p=23430&hilit=neb+hdf5+images#p23430

                Seems like there's a validation check of whether the KPOINTS file used in the head directory
                is the same as in each IMAGE subdirectory.

                Using KSPACING in INCAR gets around this issue; a kludge to use KPOINTS is to simply copy it
                to each subdirectory, as we do here
            """
            if KPOINTS_KLUDGE:
                KPOINTS_KLUDGE.write_file(f"{tdir}/KPOINTS")

        return super().make.original(self, images[1], **kwargs)
