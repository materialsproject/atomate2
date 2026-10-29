"""File I/O functions for LAMMPS input files."""

from pathlib import Path
from typing import Any, Literal

from ase.io import Trajectory as AseTrajectory
from ase.io import read
from emmet.core.vasp.calculation import StoreTrajectoryOption
from monty.serialization import dumpfn
from numpy.typing import ArrayLike
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.trajectory import Trajectory as PmgTrajectory
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.lammps.data import CombinedData, LammpsBox, LammpsData
from pymatgen.io.lammps.generators import BaseLammpsGenerator

from atomate2.common.files import copy_files, gunzip_files


def write_lammps_input_set(
    data: Structure | Molecule | LammpsData | CombinedData,
    input_set_generator: BaseLammpsGenerator,
    box_or_lattice: Lattice | LammpsBox | ArrayLike | None = None,
    additional_data: LammpsData | CombinedData | None = None,
    directory: str | Path = ".",
) -> None:
    """Write LAMMPS input set to a directory."""
    input_set = input_set_generator.get_input_set(
        data=data, additional_data=additional_data, box_or_lattice=box_or_lattice
    )
    input_set.write_input(directory)


def copy_lammps_restart_file(prev_dir: str | Path) -> str:
    """Copy the restart file of a previous LAMMPS run to the current directory.

    The copied file is gunzipped if needed; the previous directory is left untouched.

    Parameters
    ----------
    prev_dir : str or Path
        The directory of the previous LAMMPS run.

    Returns
    -------
    str
        The name of the (gunzipped) restart file in the current directory.
    """
    prev_dir = Path(prev_dir)
    # the plain and gzipped versions of a restart file count as a single file
    restart_files = {
        file.name.removesuffix(".gz"): file
        for file in prev_dir.iterdir()
        if "restart" in file.name
    }
    if len(restart_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one restart file in {prev_dir}, found "
            f"{len(restart_files)}. It should have the extension '.restart'."
        )

    ((restart_name, restart_file),) = restart_files.items()
    copy_files(prev_dir, include_files=[restart_file.name])
    if restart_file.suffix == ".gz":
        gunzip_files(include_files=[restart_file.name], force=True)
    return restart_name


class DumpConvertor:
    """
    Class to convert LAMMPS dump files to pymatgen or ase Trajectory objects.

    args:
        dumpfile : str
            Path to the LAMMPS dump file
        store_md_outputs : StoreTrajectoryOption
            Option to store MD outputs in the Trajectory object
        read_index : str | int
            Index of the frame to read from the dump file
            (default is ':', i.e. read all frames).
            Use an integer to read a specific frame (practical for large files).

    """

    def __init__(
        self,
        dumpfile: str,
        store_md_outputs: StoreTrajectoryOption = StoreTrajectoryOption.NO,
        read_index: str | int = ":",
    ) -> None:
        self.store_md_outputs = store_md_outputs
        self.traj = (
            read(dumpfile, index=read_index)
            if isinstance(read_index, str)
            else [read(dumpfile, index=read_index)]
        )
        self.is_periodic = any(self.traj[0].pbc)
        self.frame_properties_keys = ["forces", "velocities"]

    def to_ase_trajectory(self, filename: str | None = None) -> AseTrajectory:
        """Convert to ASE trajectory object."""
        for idx, atoms in enumerate(self.traj):
            with AseTrajectory(
                filename, "a" if idx > 0 else "w", atoms=atoms
            ) as file:  # check logic here
                file.write()
        return AseTrajectory(filename, "r")

    def to_pymatgen_trajectory(self, filename: str | None = None) -> PmgTrajectory:
        """Convert to pymatgen trajectory object."""
        species = AseAtomsAdaptor.get_structure(
            self.traj[0], cls=Structure if self.is_periodic else Molecule
        ).species

        frames = []
        frame_properties = []

        for atoms in self.traj:
            if self.store_md_outputs == StoreTrajectoryOption.FULL:
                frame_properties.append(
                    {
                        key: getattr(atoms, f"get_{key}")()
                        for key in self.frame_properties_keys
                    }
                )

            if self.is_periodic:
                frames.append(
                    Structure(
                        lattice=atoms.get_cell(),
                        species=species,
                        coords=atoms.get_positions(),
                        coords_are_cartesian=True,
                    )
                )
            else:
                frames.append(
                    Molecule(
                        species=species,
                        coords=atoms.get_positions(),
                        charge=atoms.get_charges(),
                        properties={"box": atoms.get_cell().tolist()},
                    )
                )
        traj_method = "from_structures" if self.is_periodic else "from_molecules"
        pmg_traj = getattr(PmgTrajectory, traj_method)(
            frames,
            frame_properties=frame_properties or None,
            constant_lattice=False,
        )

        if filename:
            dumpfn(pmg_traj, filename)

        return pmg_traj

    def save(
        self, filename: str | None = None, fmt: Literal["pmg", "ase"] = "pmg"
    ) -> Any:
        """Save the trajectory to a file."""
        filename = str(filename) if filename is not None else None
        if fmt == "pmg" and filename:
            return self.to_pymatgen_trajectory(filename=filename)
        if fmt == "ase" and filename:
            return self.to_ase_trajectory(filename=filename)
        return None
