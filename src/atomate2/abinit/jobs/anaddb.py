"""ANADDB jobs for analyzing DDB files from ABINIT DFPT calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import jobflow
from jobflow import Maker, Response, job
from pymatgen.phonon import PhononBandStructureSymmLine, PhononDos

from atomate2.abinit.files import write_anaddb_input_set
from atomate2.abinit.jobs.base import setup_job
from atomate2.abinit.run import run_anaddb
from atomate2.abinit.schemas.anaddb import AnaddbTaskDoc
from atomate2.abinit.schemas.outfiles import AbinitStoredFile
from atomate2.abinit.sets.anaddb import (
    AnaddbDfptDteInputGenerator,
    AnaddbInputGenerator,
    AnaddbPhbandsDOSInputGenerator,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pymatgen.core.structure import Structure

    from atomate2.abinit.utils.history import JobHistory

logger = logging.getLogger(__name__)

__all__ = ["AnaddbDfptDteMaker", "AnaddbMaker", "AnaddbPhBandsDOSMaker", "anaddb_job"]


_DATA_OBJECTS = [
    AbinitStoredFile,
    PhononDos,
    PhononBandStructureSymmLine,
]


def anaddb_job(method: Callable) -> job:
    """
    Decorate the ``make`` method of ANADDB job makers.

    This is a thin wrapper around :obj:`~jobflow.core.job.job` that configures common
    settings for all anaddb jobs. For example, it ensures that large data objects
    (band structures, density of states, DDB, etc) are all stored in the
    atomate2 data store. It also configures the output schema to be an Abinit
    :obj:`.TaskDocument`.

    Any makers that return Anaddb jobs (not flows) should decorate the ``make`` method
    with @anaddb_job. For example:

    .. code-block:: python

        class MyAnaddbMaker(AnaddbMaker):
            @anaddb_job
            def make(structure):
                # code to run abinit job.
                pass

    Parameters
    ----------
    method : callable
        A AnaddbMaker.make method. This should not be specified directly and is
        implied by the decorator.

    Returns
    -------
    callable
        A decorated version of the make function that will generate Anaddb jobs.
    """
    return job(method, data=_DATA_OBJECTS, output_schema=AnaddbTaskDoc)


@dataclass
class AnaddbMaker(Maker):
    """
    Maker to create a job to analyze DDB files using ANADDB.

    ANADDB (ANAlysis of Derivative DataBase) is an ABINIT utility that
    post-processes derivative database (DDB) files from DFPT calculations
    to extract physical properties.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : AnaddbInputGenerator
        Generator for ANADDB input files. Defaults to AnaddbInputGenerator.
    factory_kwargs : dict
        Additional keyword arguments passed to the input set generator
        factory methods.
    task_document_kwargs : dict
        Additional keyword arguments passed to AnaddbTaskDoc.from_directory().
    """

    name: str = "Anaddb"
    input_set_generator: AnaddbInputGenerator = field(
        default_factory=AnaddbInputGenerator
    )
    factory_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)

    @property
    def calc_type(self) -> str:
        """Get the type of calculation for this maker."""
        return self.input_set_generator.calc_type

    @anaddb_job
    def make(
        self,
        structure: Structure,
        prev_outputs: str | list[str] | None = None,
        history: JobHistory | None = None,
    ) -> jobflow.Response:
        """
        Create an ANADDB job to analyze a DDB file.

        Parameters
        ----------
        structure : Structure
            A pymatgen Structure object. Required for ANADDB analysis to
            interpret the DDB file and generate outputs like phonon bands.
        prev_outputs : str or list[str] or None
            Path(s) to previous calculation directories containing DDB files
            to analyze. Can be a single path or a list of paths.
        history : JobHistory or None
            A JobHistory object containing the history of previous jobs in
            the workflow.

        Returns
        -------
        Response
            A jobflow Response containing an AnaddbTaskDoc with the analysis
            results.
        """
        # Setup job and get general job configuration
        config = setup_job(
            structure=None,
            prev_outputs=prev_outputs,
            restart_from=None,
            history=history,
            wall_time=None,
        )

        # Write anaddb input set
        write_anaddb_input_set(
            structure=structure,
            input_set_generator=self.input_set_generator,
            prev_outputs=prev_outputs,
            directory=config.workdir,
        )

        # Run anaddb
        run_anaddb(
            start_time=config.start_time,
        )

        # Parse ANADDB output
        task_doc = AnaddbTaskDoc.from_directory(
            Path.cwd(),
            **self.task_document_kwargs,
        )
        task_doc.task_label = self.name

        return Response(output=task_doc)


@dataclass
class AnaddbDfptDteMaker(AnaddbMaker):
    """
    Maker to extract DFPT properties including DTE tensors from a merged DDB.

    This maker uses ANADDB to analyze merged DDB files from DFPT calculations
    that include derivative with respect to electric field (DTE).
    It extracts the static SHG and dielectric tensors.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : AnaddbInputGenerator
        Generator for ANADDB input files. Defaults to
        AnaddbDfptDteInputGenerator, which is configured for DTE analysis.
    factory_kwargs : dict
        Additional keyword arguments passed to the input set generator
        factory methods.
    task_document_kwargs : dict
        Additional keyword arguments passed to AnaddbTaskDoc.from_directory().
    """

    name: str = "Anaddb"
    input_set_generator: AnaddbInputGenerator = field(
        default_factory=AnaddbDfptDteInputGenerator
    )


@dataclass
class AnaddbPhBandsDOSMaker(AnaddbMaker):
    """
    Maker to compute phonon band structure and density of states from a DDB.

    This maker uses ANADDB to interpolate phonon frequencies from a merged
    DDB file, generating phonon band structures along high-symmetry paths
    and phonon density of states. It can also compute thermodynamic
    properties such as heat capacity, entropy, and free energy.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : AnaddbInputGenerator
        Generator for ANADDB input files. Defaults to
        AnaddbPhbandsDOSInputGenerator, which is configured for phonon
        band structure and DOS analysis.
    factory_kwargs : dict
        Additional keyword arguments passed to the input set generator
        factory methods.
    task_document_kwargs : dict
        Additional keyword arguments passed to AnaddbTaskDoc.from_directory().
    """

    name: str = "Anaddb PhbandsDOS"
    input_set_generator: AnaddbInputGenerator = field(
        default_factory=AnaddbPhbandsDOSInputGenerator
    )
