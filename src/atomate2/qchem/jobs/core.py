"""Core jobs for running QChem calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atomate2.qchem.sets.base import QCInputGenerator

from atomate2.qchem.jobs.base import BaseQCMaker
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


@dataclass
class SinglePointMaker(BaseQCMaker):
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

    name: str = "single point"
    input_set_generator: QCInputGenerator = field(
        default_factory=SinglePointSetGenerator
    )


@dataclass
class OptMaker(BaseQCMaker):
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
    input_set_generator: QCInputGenerator = field(default_factory=OptSetGenerator)


@dataclass
class ForceMaker(BaseQCMaker):
    """
    QChem Maker for a Force Job.

    Maker to create QChem job to converge electron density
    and calculate the gradient and atomic forces.

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
    input_set_generator: QCInputGenerator = field(default_factory=ForceSetGenerator)


@dataclass
class TransitionStateMaker(BaseQCMaker):
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

    name: str = "transition state"
    input_set_generator: QCInputGenerator = field(
        default_factory=TransitionStateSetGenerator
    )


@dataclass
class FreqMaker(BaseQCMaker):
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
    input_set_generator: QCInputGenerator = field(default_factory=FreqSetGenerator)


@dataclass
class PESScanMaker(BaseQCMaker):
    """
    Maker for a PES Scan job.

    Maker to create a QChem job which perform a potential energy surface
    scan by varying bond lengths, angles,
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

    name: str = "PES scan"
    input_set_generator: QCInputGenerator = field(default_factory=PESScanSetGenerator)
