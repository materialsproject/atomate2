"""Task Document for LAMMPS calculations."""

import os
import warnings
from glob import glob
from pathlib import Path
from typing import Literal, Optional

from emmet.core.structure import StructureMetadata
from emmet.core.vasp.calculation import StoreTrajectoryOption
from emmet.core.vasp.task_valid import TaskState
from pydantic import Field
from pymatgen.core import Composition, Structure
from pymatgen.core.trajectory import Trajectory
from pymatgen.io.lammps.generators import LammpsData, LammpsInputFile
from pymatgen.io.lammps.outputs import parse_lammps_log

from atomate2.lammps.files import DumpConvertor
from atomate2.utils.datetime import datetime_str


class LammpsTaskDocument(StructureMetadata):
    """Task Document for LAMMPS calculations."""

    dir_name: str = Field(None, description="Directory where the task was run")

    task_label: str = Field(None, description="Label for the task")

    last_updated: str = Field(
        datetime_str(), description="Timestamp for the last time the task was updated"
    )

    trajectories: Optional[list[Trajectory]] = Field(
        None, description="Pymatgen trajectories output from lammps run"
    )

    composition: Composition | None = Field(
        None, description="Composition of the system"
    )

    state: TaskState = Field(None, description="State of the calculation")

    reduced_formula: str | None = Field(
        None, description="Reduced formula of the system"
    )

    dump_files: Optional[dict] = Field(
        None, description="Dump files produced by lammps run"
    )

    structure: Optional[Structure] = Field(
        None, description="Final structure of the system, taken from the last dump file"
    )

    metadata: Optional[dict] = Field(None, description="Metadata for the task")

    raw_log_file: str = Field(None, description="Log file output from lammps run")

    thermo_log: list = Field(
        None,
        description="Parsed log output from lammps run, with a focus on thermo data",
    )

    inputs: dict = Field(None, description="Input files for the task")

    additional_outputs: Optional[dict] = Field(
        None,
        description="Additional outputs written out by the lammps run that \
            do not end with .dump or .log",
    )

    @classmethod
    def from_directory(
        cls: type["LammpsTaskDocument"],
        dir_name: str | Path,
        task_label: str,
        store_trajectory: StoreTrajectoryOption = StoreTrajectoryOption.NO,
        trajectory_format: Literal["pmg", "ase"] = "pmg",
        output_file_pattern: str | None = None,
        parse_additional_outputs: Optional[list] = None,
    ) -> "LammpsTaskDocument":
        """
        Create a LammpsTaskDocument from a directory where LAMMPS was run.

        dir_name: str | Path
            Directory where the task was run
        task_label: str
            Label for the task
        store_trajectory: Literal["no", "partial", "full"]
            Whether to store the trajectory output from the lammps run.
            Default is 'no', which does not parse and store any trajectory data.
        trajectory_format: Literal["pmg", "ase"]
            Format of the trajectory output. Default is 'pmg'
        output_file_pattern: str
            Pattern for the output file, written to disk in dir_name. Default is None.
        additional_outputs: Optional[list]
            Additional outputs to be stored in the task document that
            do not end with .dump or .log. Default is None. Provide a list of filenames
            that need to be parsed (as raw text) and stored in the task document
            under extra_outputs.
        """
        log_file = os.path.join(dir_name, "log.lammps")
        try:
            with open(log_file) as f:
                raw_log = f.read()
            thermo_log = parse_lammps_log(log_file)
            state = TaskState.ERROR if "ERROR" in raw_log else TaskState.SUCCESS
        except ValueError:
            raise ValueError(
                f"Error parsing log file for {dir_name}, incomplete job!"
            ) from None

        if state == TaskState.ERROR:
            return LammpsTaskDocument(
                dir_name=str(dir_name),
                task_label=task_label,
                raw_log_file=raw_log,
                thermo_log=thermo_log,
                state=state,
            )

        dump_file_keys = glob("*dump*", root_dir=dir_name)
        dump_files = {}
        if dump_file_keys:
            for dump_file in dump_file_keys:
                with open(os.path.join(dir_name, dump_file)) as f:
                    dump_files[dump_file] = f.read()

            if store_trajectory != StoreTrajectoryOption.NO:
                warnings.warn(
                    "Trajectory data might be large, only store if \
                        absolutely necessary. Consider manually \
                            parsing the dump files instead.",
                    stacklevel=1,
                )
                if output_file_pattern is None:
                    output_file_pattern = "trajectory"
                trajectories = [
                    DumpConvertor(
                        store_md_outputs=store_trajectory,
                        dumpfile=os.path.join(dir_name, dump_file),
                    ).save(
                        filename=f"{output_file_pattern}{i}.traj", fmt=trajectory_format
                    )
                    for i, dump_file in enumerate(dump_files)
                ]

            final_structure = (
                DumpConvertor(
                    dumpfile=os.path.join(dir_name, dump_file_keys[-1]), read_index=-1
                )
                .to_pymatgen_trajectory()
                .get_structure(-1)
            )
        else:
            warnings.warn(
                "No dump files found, no trajectory data stored", stacklevel=1
            )
            final_structure = None

        try:
            input_file = LammpsInputFile.from_file(
                os.path.join(dir_name, "in.lammps"), ignore_comments=True
            )
        except FileNotFoundError:
            warnings.warn(f"Input file not found for {dir_name}", stacklevel=1)
            input_file = None
        try:
            data_files = [
                LammpsData.from_file(
                    os.path.join(dir_name, file),
                    atom_style=input_file.get_args("atom_style"),
                )
                for file in glob("*.data*", root_dir=dir_name)
            ]
        except FileNotFoundError:
            warnings.warn(f"Data file not found for {dir_name}", stacklevel=1)
            data_files = None

        if parse_additional_outputs is not None:
            additional_outputs = {}
            for output_file in parse_additional_outputs:
                output_path = os.path.join(dir_name, output_file)
                if os.path.exists(output_path):
                    with open(output_path) as f:
                        additional_outputs[output_file] = f.read()
                else:
                    warnings.warn(
                        f"Additional output file {output_file} not found in {dir_name}",
                        stacklevel=1,
                    )

        inputs = {"in.lammps": input_file, "data_files": data_files}
        composition = final_structure.composition if final_structure else None
        reduced_formula = composition.reduced_formula if final_structure else None

        return LammpsTaskDocument(
            dir_name=str(dir_name),
            task_label=task_label,
            raw_log_file=raw_log,
            thermo_log=thermo_log,
            dump_files=dump_files,
            trajectories=trajectories
            if store_trajectory != StoreTrajectoryOption.NO
            else None,
            structure=final_structure,
            composition=composition,
            reduced_formula=reduced_formula,
            inputs=inputs,
            state=state,
            additional_outputs=additional_outputs if parse_additional_outputs else None,
        )
