"""Core jobs for running QChem calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from atomate2.qchem.jobs.base import BaseQChemMaker
from atomate2.qchem.sets.base import QChemInputGenerator
from atomate2.qchem.sets.core import (
    ForceSetGenerator,
    FreqSetGenerator,
    OptSetGenerator,
    PESScanSetGenerator,
    SinglePointSetGenerator,
    TransitionStateSetGenerator,
)

# from custodian.qchem.handlers import (
#     QChemErrorHandler,
# )


logger = logging.getLogger(__name__)

__all__ = [
    "SinglePointMaker",
    "OptMaker",
    "ForceMaker",
    "TransitionStateMaker",
    "FreqMaker",
    "PESScanMaker",
]


@dataclass
class SinglePointMaker(BaseQChemMaker):
    """
    Maker to create QChem single point calculation jobs.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .QChemInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_qchem_input_set`.
    copy_qchem_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_qchem_outputs`.
    run_qchem_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_qchem`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDocument.from_directory`.
    stop_children_kwargs : dict
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data : dict
        Additional data to write to the current directory. Given as a dict of
        {filename: data}. Note that if using FireWorks, dictionary keys cannot contain
        the "." character which is typically used to denote file extensions. To avoid
        this, use the ":" character, which will automatically be converted to ".". E.g.
        ``{"my_file:txt": "contents of the file"}``.
    """

    name: str = "single_point"
    input_set_generator: QChemInputGenerator = field(
        default_factory=SinglePointSetGenerator
    )


@dataclass
class OptMaker(BaseQChemMaker):
    """
    Maker to create QChem optimization jobs.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .QChemInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_qchem_input_set`.
    copy_qchem_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_qchem_outputs`.
    run_qchem_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_qchem`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDocument.from_directory`.
    stop_children_kwargs : dict
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data : dict
        Additional data to write to the current directory. Given as a dict of
        {filename: data}. Note that if using FireWorks, dictionary keys cannot contain
        the "." character which is typically used to denote file extensions. To avoid
        this, use the ":" character, which will automatically be converted to ".". E.g.
        ``{"my_file:txt": "contents of the file"}``.
    """

    name: str = "optimization"
    input_set_generator: QChemInputGenerator = field(default_factory=OptSetGenerator)


@dataclass
class ForceMaker(BaseQChemMaker):
    """
    Maker to create QChem job to converge electron density and calculate the gradient and atomic forces

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .QChemInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_qchem_input_set`.
    copy_qchem_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_qchem_outputs`.
    run_qchem_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_qchem`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDocument.from_directory`.
    stop_children_kwargs : dict
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data : dict
        Additional data to write to the current directory. Given as a dict of
        {filename: data}. Note that if using FireWorks, dictionary keys cannot contain
        the "." character which is typically used to denote file extensions. To avoid
        this, use the ":" character, which will automatically be converted to ".". E.g.
        ``{"my_file:txt": "contents of the file"}``.
    """

    name: str = "force"
    input_set_generator: QChemInputGenerator = field(default_factory=ForceSetGenerator)


@dataclass
class TransitionStateMaker(BaseQChemMaker):
    """
     Maker to create QChem transition state structure optimization jobs.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .QChemInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_qchem_input_set`.
    copy_qchem_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_qchem_outputs`.
    run_qchem_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_qchem`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDocument.from_directory`.
    stop_children_kwargs : dict
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data : dict
        Additional data to write to the current directory. Given as a dict of
        {filename: data}. Note that if using FireWorks, dictionary keys cannot contain
        the "." character which is typically used to denote file extensions. To avoid
        this, use the ":" character, which will automatically be converted to ".". E.g.
        ``{"my_file:txt": "contents of the file"}``.
    """

    name: str = "transition_state"
    input_set_generator: QChemInputGenerator = field(
        default_factory=TransitionStateSetGenerator
    )


@dataclass
class FreqMaker(BaseQChemMaker):
    """
     Maker to create QChem job for frequency calculations.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .QChemInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_qchem_input_set`.
    copy_qchem_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_qchem_outputs`.
    run_qchem_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_qchem`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDocument.from_directory`.
    stop_children_kwargs : dict
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data : dict
        Additional data to write to the current directory. Given as a dict of
        {filename: data}. Note that if using FireWorks, dictionary keys cannot contain
        the "." character which is typically used to denote file extensions. To avoid
        this, use the ":" character, which will automatically be converted to ".". E.g.
        ``{"my_file:txt": "contents of the file"}``.
    """

    name: str = "frequency"
    input_set_generator: QChemInputGenerator = field(default_factory=FreqSetGenerator)


@dataclass
class PESScanMaker(BaseQChemMaker):
    """
     Maker to create a QChem job which performa a potential energy surface scan by varying bond lengths, angles,
    and/or dihedral angles in a molecule.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .QChemInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_qchem_input_set`.
    copy_qchem_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_qchem_outputs`.
    run_qchem_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_qchem`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDocument.from_directory`.
    stop_children_kwargs : dict
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data : dict
        Additional data to write to the current directory. Given as a dict of
        {filename: data}. Note that if using FireWorks, dictionary keys cannot contain
        the "." character which is typically used to denote file extensions. To avoid
        this, use the ":" character, which will automatically be converted to ".". E.g.
        ``{"my_file:txt": "contents of the file"}``.
    """

    name: str = "pes_scan"
    input_set_generator: QChemInputGenerator = field(
        default_factory=PESScanSetGenerator
    )
