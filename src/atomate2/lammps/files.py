"""File I/O functions for LAMMPS input files."""

from pathlib import Path
from typing import Literal

from ase.io import Trajectory as AseTrajectory
from ase.io import read
from emmet.core.vasp.calculation import StoreTrajectoryOption
from monty.serialization import dumpfn
from pymatgen.core import Molecule, Structure
from pymatgen.core.trajectory import Trajectory as PmgTrajectory
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.lammps.data import CombinedData, LammpsData
from pymatgen.io.lammps.generators import BaseLammpsGenerator


def write_lammps_input_set(
    data: Structure | LammpsData | CombinedData,
    input_set_generator: BaseLammpsGenerator,
    additional_data: LammpsData | CombinedData | None = None,
    directory: str | Path = ".",
) -> None:
    """Write LAMMPS input set to a directory."""
    input_set = input_set_generator.get_input_set(
        data,
        additional_data,
    )
    input_set.write_input(directory)


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
                    )
                )
        traj_method = "from_structures" if self.is_periodic else "from_molecules"
        pmg_traj = getattr(PmgTrajectory, traj_method)(
            frames,
            frame_properties=frame_properties if frame_properties else None,
            constant_lattice=False,
        )

        if filename:
            dumpfn(pmg_traj, filename)

        return pmg_traj

    def save(
        self, filename: str | None = None, fmt: Literal["pmg", "ase"] = "pmg"
    ) -> PmgTrajectory | AseTrajectory | None:
        """Save the trajectory to a file."""
        filename = str(filename) if filename is not None else None
        if fmt == "pmg" and filename:
            return self.to_pymatgen_trajectory(filename=filename)
        if fmt == "ase" and filename:
            return self.to_ase_trajectory(filename=filename)
        return None
