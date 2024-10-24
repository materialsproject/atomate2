"""Core jobs for running VASP calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pymatgen.io.vasp.sets import MVLGWSet

from atomate2.vasp.jobs.base import BaseVaspMaker, vasp_job

if TYPE_CHECKING:
    from pathlib import Path

    from jobflow import Response
    from pymatgen.core.structure import Structure

    from atomate2.vasp.sets.base import VaspInputGenerator


logger = logging.getLogger(__name__)


@dataclass
class MVLStaticMaker(BaseVaspMaker):
    """
    Maker to create a static calculation compatible with Materials Virtual Lab GW jobs.

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

    name: str = "MVL static"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: MVLGWSet(mode="STATIC")
    )

    @vasp_job
    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
    ) -> Response:
        """
        Run a static calculation compatible with later Materials Virtual Lab GW jobs.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.
        """
        return super().make.original(self, structure, prev_dir)


@dataclass
class MVLNonSCFMaker(BaseVaspMaker):
    """
    Maker to create a non-scf calculation compatible with Materials Virtual Lab GW jobs.

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

    name: str = "MVL nscf"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: MVLGWSet(mode="DIAG")
    )

    @vasp_job
    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
    ) -> Response:
        """
        Run a static calculation compatible with later Materials Virtual Lab GW jobs.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.
        """
        self.copy_vasp_kwargs.setdefault("additional_vasp_files", ("CHGCAR",))

        return super().make.original(self, structure, prev_dir)


@dataclass
class MVLGWMaker(BaseVaspMaker):
    """
    Maker to create Materials Virtual Lab GW jobs.

    This class can make the jobs for the typical three stapes of the GW calculation.

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

    name: str = "MVL G0W0"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: MVLGWSet(mode="GW")
    )

    @vasp_job
    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
    ) -> Response:
        """
        Run a Materials Virtual Lab GW band structure VASP job.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure object.
        prev_dir : str or Path or None
            A previous VASP calculation directory to copy output files from.
        """
        self.copy_vasp_kwargs.setdefault(
            "additional_vasp_files", ("CHGCAR", "WAVECAR", "WAVEDER")
        )

        return super().make.original(self, structure, prev_dir)
