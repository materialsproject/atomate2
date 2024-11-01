from pathlib import Path
import numpy as np
from pymatgen.core import Structure, Molecule
from pymatgen.io.lammps.generators import BaseLammpsGenerator
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.trajectory import Trajectory as PmgTrajectory
from ase.io import Trajectory as AseTrajectory
from typing import Literal
from monty.serialization import dumpfn
from monty.json import MSONable

def write_lammps_input_set(
    structure: Structure | Path,
    input_set_generator: BaseLammpsGenerator,
    directory: str | Path = ".",
    **kwargs,
):
    input_set = input_set_generator.get_input_set(structure, **kwargs)
    input_set.write_input(directory)


class DumpConvertor(MSONable):    
    def __init__(self, dumpfile, store_md_outputs : Literal["no", "partial", "full"]  = 'partial') -> None:
        self.store_md_outputs = store_md_outputs
        self.traj = read(dumpfile, index=':')
        self.is_periodic = any(self.traj[0].pbc)
        self.frame_properties_keys = ['forces', 'velocities']
        
    def to_ase_trajectory(self, filename : str | None = None):
        for idx, atoms in enumerate(self.traj):
            with AseTrajectory(filename, 'a' if idx > 0 else 'w', atoms = atoms) as file: #check logic here
                file.write()
        return AseTrajectory(filename, 'r')
    
    def to_pymatgen_trajectory(self, filename : str | None = 'trajectory.json.gz') -> PmgTrajectory:
                
        species = AseAtomsAdaptor.get_structure(self.traj[0], cls = Structure if self.is_periodic else Molecule).species

        frames  = []
        frame_properties = []
        
        for atoms in self.traj:
            if self.store_md_outputs:
                if self.store_md_outputs == 'full':
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
        if fmt == "pmg":
            return self.to_pymatgen_trajectory(filename=filename) 
        elif fmt == "ase":
            return self.to_ase_trajectory(filename=filename)

        