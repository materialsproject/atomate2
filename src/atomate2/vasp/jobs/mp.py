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
from atomate2.vasp.jobs.core import StaticSetGenerator
from atomate2.vasp.sets.base import VaspInputGenerator

if TYPE_CHECKING:
    from pymatgen.core import Structure

__all__ = ["MPPreRelaxMaker", "MPRelaxMaker", "MPStaticMaker"]

_BASE_MP_R2SCAN_RELAX_SET = loadfn(
    resource_filename("atomate2.vasp.sets", "BaseMPR2SCANRelaxSet.yaml")
)


class MPMetaRelaxGenerator(VaspInputGenerator):
    config_dict: dict = _BASE_MP_R2SCAN_RELAX_SET


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

    name: str = "MP PreRelax"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: MPMetaRelaxGenerator(
            user_incar_settings={"EDIFFG": -0.05, "METAGGA": None, "GGA": "PS"}
        )
    )


@dataclass
class MPMetaRelaxMaker(BaseVaspMaker):
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
        default_factory=MPMetaRelaxGenerator
    )

    def make(
        self, structure: Structure, bandgap: float = 0.0, bandgap_tol: float = 1e-4
    ):
        """Set correct k-point density, smearing and sigma based on bandgap estimate.

        Parameters
        ----------
        structure : pymatgen.Structure
            The structure to relax.
        bandgap : float
            The bandgap of the material in eV. Used to determine the k-point density.
        bandgap_tol : float
            The tolerance for the bandgap. If the bandgap is less than this value, the
            k-point density will be set to 0.22, otherwise it will be set to a value
            based on the bandgap.

        Returns
        -------
        MPRelaxMaker
            The maker.
        """
        self.input_set_generator.config_dict["INCAR"].update(
            _get_kspacing_params(bandgap, bandgap_tol)
        )

        return super().make(structure=structure)


@dataclass
class MPMetaStaticMaker(BaseVaspMaker):
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
        default_factory=lambda: MPMetaRelaxGenerator(
            user_incar_settings={"NSW": 0, "ISMEAR": -5, "LREAL": False}
        )
    )

    def make(
        self, structure: Structure, bandgap: float = 0.0, bandgap_tol: float = 1e-4
    ):
        """Set correct k-point density, smearing and sigma based on bandgap estimate.

        Parameters
        ----------
        structure : pymatgen.Structure
            The structure to relax.
        bandgap : float
            The bandgap of the material in eV. Used to determine the k-point density.
        bandgap_tol : float
            The tolerance for the bandgap. If the bandgap is less than this value, the
            k-point density will be set to 0.22, otherwise it will be set to a value
            based on the bandgap.

        Returns
        -------
            Response: A response object containing the output, detours and stop
                commands of the VASP run.
        """
        self.input_set_generator.config_dict["INCAR"].update(
            _get_kspacing_params(bandgap, bandgap_tol)
        )

        return super().make(structure=structure)


def _get_kspacing_params(bandgap: float, bandgap_tol: float) -> dict[str, int | float]:
    """Get the k-point density, smearing and sigma based on bandgap estimate.

    Parameters
    ----------
    bandgap : float
        The bandgap of the material in eV. Used to determine the k-point density.
    bandgap_tol : float
        Tolerance for metallic bandgap. If bandgap < bandgap_tol, KSPACING will be 0.22,
        otherwise it will increase with bandgap up to a max of 0.44.

    Returns
    -------
    Dict
        {"KSPACING": float, "ISMEAR": int, "SIGMA": float}
    """
    if bandgap < bandgap_tol:  # metallic
        return {"KSPACING": 0.22, "ISMEAR": 2, "SIGMA": 0.2}

    rmin = 25.22 - 2.87 * bandgap
    kspacing = 2 * np.pi * 1.0265 / (rmin - 1.0183)
    return {"KSPACING": min(kspacing, 0.44), "ISMEAR": -5, "SIGMA": 0.05}
