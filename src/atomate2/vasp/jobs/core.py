"""Core jobs for running VASP calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from jobflow import job
from pymatgen.core.structure import Structure

from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.schemas.task import TaskDocument
from atomate2.vasp.sets.base import VaspInputSetGenerator
from atomate2.vasp.sets.core import (
    HSEBSSetGenerator,
    HSERelaxSetGenerator,
    NonSCFSetGenerator,
    RelaxSetGenerator,
    StaticSetGenerator,
)

logger = logging.getLogger(__name__)

__all__ = ["StaticMaker", "RelaxMaker", "NonSCFMaker", "DielectricMaker", "HSEBSMaker"]


@dataclass
class StaticMaker(BaseVaspMaker):
    """Maker to create VASP static jobs."""

    name: str = "static"
    input_set_generator: VaspInputSetGenerator = field(
        default_factory=StaticSetGenerator
    )


@dataclass
class RelaxMaker(BaseVaspMaker):
    """Maker to create VASP relaxation jobs."""

    name: str = "relax"
    input_set_generator: VaspInputSetGenerator = field(
        default_factory=RelaxSetGenerator
    )


@dataclass
class NonSCFMaker(BaseVaspMaker):
    """Maker to create non self consistent field VASP jobs."""

    name: str = "non-scf"
    input_set_generator: VaspInputSetGenerator = field(
        default_factory=NonSCFSetGenerator
    )

    @job(output_schema=TaskDocument)
    def make(
        self,
        structure: Structure,
        prev_vasp_dir: Union[str, Path],
        mode: str = "uniform",
    ):
        """
        Run a non-scf VASP job.

        Parameters
        ----------
        structure
            A pymatgen structure object.
        prev_vasp_dir
            A previous VASP calculation directory to copy output files from.
        mode
            Type of band structure calculation. Options are:
            - "line": Full band structure along symmetry lines.
            - "uniform": Uniform mesh band structure.
        """
        self.input_set_generator.mode = mode

        if "parse_dos" not in self.task_document_kwargs:
            # parse DOS only for uniform band structure
            self.task_document_kwargs["parse_dos"] = mode == "uniform"

        if "parse_bandstructure" not in self.task_document_kwargs:
            self.task_document_kwargs["parse_bandstructure"] = mode

        # copy previous inputs
        if "additional_vasp_files" not in self.copy_vasp_kwargs:
            self.copy_vasp_kwargs["additional_vasp_files"] = ("CHGCAR",)

        return super().make.original(self, structure, prev_vasp_dir)


@dataclass
class HSERelaxMaker(BaseVaspMaker):
    """Maker to create DFPT VASP jobs."""

    name: str = "hse relax"
    input_set_generator: VaspInputSetGenerator = field(
        default_factory=HSERelaxSetGenerator
    )


@dataclass
class HSEBSMaker(BaseVaspMaker):
    """Maker to create HSE06 band structure jobs."""

    name: str = "hse band structure"
    input_set_generator: VaspInputSetGenerator = field(
        default_factory=HSEBSSetGenerator
    )

    @job(output_schema=TaskDocument)
    def make(
        self,
        structure: Structure,
        prev_vasp_dir: Union[str, Path] = None,
        mode="uniform",
    ):
        """
        Run a HSE06 band structure VASP job.

        Parameters
        ----------
        structure
            A pymatgen structure object.
        prev_vasp_dir
            A previous VASP calculation directory to copy output files from.
        mode
            Type of band structure calculation. Options are:
            - "line": Full band structure along symmetry lines.
            - "uniform": Uniform mesh band structure.
            - "gap": Get the energy at the CBM and VBM.
        """
        self.input_set_generator.mode = mode

        if mode == "gap" and prev_vasp_dir is None:
            logger.warning(
                "HSE band structure in 'gap' mode requires a previous VASP calculation "
                "directory from which to extract the VBM and CBM k-points. This "
                "calculation will instead be a standard uniform calculation."
            )

        if "parse_dos" not in self.task_document_kwargs:
            # parse DOS only for uniform band structure
            self.task_document_kwargs["parse_dos"] = mode == "uniform"

        if "parse_bandstructure" not in self.task_document_kwargs:
            parse_bandstructure = "uniform" if mode == "gap" else mode
            self.task_document_kwargs["parse_bandstructure"] = parse_bandstructure

        # copy previous inputs
        if (
            prev_vasp_dir is not None
            and "additional_vasp_files" not in self.copy_vasp_kwargs
        ):
            self.copy_vasp_kwargs["additional_vasp_files"] = ("CHGCAR",)

        return super().make.original(self, structure, prev_vasp_dir)


@dataclass
class DielectricMaker(BaseVaspMaker):
    """Maker to create dielectric calculation VASP jobs."""

    name: str = "dielectric"
    input_set_generator: StaticSetGenerator = field(
        default_factory=lambda: StaticSetGenerator(lepsilon=True)
    )
