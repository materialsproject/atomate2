from pathlib import Path
from pymatgen.core import Structure, Molecule
from pymatgen.io.lammps.generators import BaseLammpsGenerator
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.lammps.data import LammpsData
from pymatgen.core.trajectory import Trajectory as PmgTrajectory
from ase.io import Trajectory as AseTrajectory
from typing import Literal
from monty.serialization import dumpfn
from monty.json import MSONable
from emmet.core.vasp.calculation import StoreTrajectoryOption
import warnings

def write_lammps_input_set(
    structure: Structure | Path | None,
    input_set_generator: BaseLammpsGenerator,
    directory: str | Path = ".",
    **kwargs,
):
    data = structure
    if input_set_generator.interchange:
        warnings.warn("Interchange is experimental and may not work as expected. Use with caution. Ensure FF units are consistent with LAMMPS.")
        #write unit convertor here
        input_set_generator.interchange.to_lammps_datafile("interchange_data.lmp")
        data = LammpsData.from_file("interchange_data.lmp", atom_style=input_set_generator.atom_style)
        #validate data here: ff coeffs style, atom_style, etc. have to be updated into the input_set_generator.settings
    input_set = input_set_generator.get_input_set(data, **kwargs)
    input_set.write_input(directory)


class DumpConvertor(MSONable):    
    def __init__(self, dumpfile, store_md_outputs : StoreTrajectoryOption  = StoreTrajectoryOption.NO) -> None:
        self.store_md_outputs = store_md_outputs
        self.traj = read(dumpfile, index=':')
        self.is_periodic = any(self.traj[0].pbc)
        self.frame_properties_keys = ['forces', 'velocities']
        
    def to_ase_trajectory(self, filename : str | None = None):
        for idx, atoms in enumerate(self.traj):
            with AseTrajectory(filename, 'a' if idx > 0 else 'w', atoms = atoms) as file: #check logic here
                file.write()
        return AseTrajectory(filename, 'r')
    
    def to_pymatgen_trajectory(self, filename : str | None = None) -> PmgTrajectory:
                
        species = AseAtomsAdaptor.get_structure(self.traj[0], cls = Structure if self.is_periodic else Molecule).species

        frames  = []
        frame_properties = []
        
        for atoms in self.traj:
            if self.store_md_outputs == StoreTrajectoryOption.FULL:
                frame_properties.append({key : getattr(atoms, f'get_{key}')() for key in self.frame_properties_keys})
                
            if self.is_periodic:
                frames.append(Structure(lattice = atoms.get_cell(), 
                                        species = species, 
                                        coords = atoms.get_positions(), 
                                        coords_are_cartesian = True)
                                )
            else:
                frames.append(Molecule(species = species, 
                                        coords = atoms.get_positions(), 
                                        charge = atoms.get_charges(),
                                        )
                                  )
        traj_method = 'from_structures' if self.is_periodic else 'from_molecules'
        pmg_traj = getattr(PmgTrajectory, traj_method)(
            frames,
            frame_properties=frame_properties if frame_properties else None,
            constant_lattice=False,
        )
        
        if filename:
            dumpfn(pmg_traj, filename)
        
        return pmg_traj
    
    def save(self, filename : str | None = None, fmt : Literal["pmg", "ase"] = "pmg"):
        filename = str(filename) if filename is not None else None
        if fmt == "pmg" and filename:
            return self.to_pymatgen_trajectory(filename=filename) 
        if fmt == "ase" and filename:
            return self.to_ase_trajectory(filename=filename)