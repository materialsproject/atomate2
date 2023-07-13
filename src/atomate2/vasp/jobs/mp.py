"""
Module defining jobs for Materials Project r2SCAN workflows.

Reference: https://doi.org/10.1103/PhysRevMaterials.6.013801
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from monty.serialization import loadfn
from pkg_resources import resource_filename

from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.sets.base import VaspInputGenerator

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Outcar, Vasprun

__all__ = ["MPPreRelaxMaker", "MPMetaGGARelaxMaker", "MPMetaGGAStaticMaker"]

_BASE_MP_R2SCAN_RELAX_SET = loadfn(
    resource_filename("atomate2.vasp.sets", "BaseMPR2SCANRelaxSet.yaml")
)


class MPMetaGGARelaxGenerator(VaspInputGenerator):
    config_dict: dict = field(default_factory=lambda: _BASE_MP_R2SCAN_RELAX_SET)
    # Tolerance for metallic bandgap. If bandgap < bandgap_tol, KSPACING will be 0.22,
    # otherwise it will increase with bandgap up to a max of 0.44.
    bandgap_tol: float = 1e-4
    bandgap_override: float | None = None

    def get_incar_updates(
        self,
        structure: Structure,
        prev_incar: dict = None,
        bandgap: float = None,
        vasprun: Vasprun = None,
        outcar: Outcar = None,
    ) -> dict:
        """
        Get updates to the INCAR for a relaxation job.

        Parameters
        ----------
        structure
            A structure.
        prev_incar
            An incar from a previous calculation.
        bandgap
            The band gap.
        vasprun
            A vasprun from a previous calculation.
        outcar
            An outcar from a previous calculation.

        Returns
        -------
        dict
            A dictionary of updates to apply.
        """
        updates = {"EDIFFG": -0.05, "METAGGA": None, "GGA": "PS"}
        bandgap = self.bandgap_override or bandgap

        if bandgap < self.bandgap_tol:  # metallic
            return {"KSPACING": 0.22, "ISMEAR": 2, "SIGMA": 0.2, **updates}

        rmin = 25.22 - 2.87 * bandgap
        kspacing = 2 * np.pi * 1.0265 / (rmin - 1.0183)
        return {"KSPACING": min(kspacing, 0.44), "ISMEAR": -5, "SIGMA": 0.05, **updates}


@dataclass
class MPPreRelaxMaker(BaseVaspMaker):
    """
    Maker to create VASP pre-relaxation job using PBEsol by default.

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

    name: str = "MP pre-relax"
    input_set_generator: VaspInputGenerator = field(
        default_factory=MPMetaGGARelaxGenerator
    )


@dataclass
class MPMetaGGARelaxMaker(BaseVaspMaker):
    """
    Maker to create VASP relaxation job using r2SCAN by default.

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

    name: str = "MP Relax"
    input_set_generator: VaspInputGenerator = field(
        default_factory=MPMetaGGARelaxGenerator
    )


@dataclass
class MPMetaGGAStaticMaker(BaseVaspMaker):
    """
    Maker to create VASP static job using r2SCAN by default.

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

    name: str = "MP Static"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: MPMetaGGARelaxGenerator(
            user_incar_settings={"NSW": 0, "ISMEAR": -5, "LREAL": False}
        )
    )
