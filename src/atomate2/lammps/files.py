from pathlib import Path

from pymatgen.core import Structure

from pymatgen.io.lammps.generators import BaseLammpsGenerator
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.trajectory import Trajectory as PmgTrajectory
from ase.io import Trajectory as AseTrajectory

def write_lammps_input_set(
    structure: Structure | Path,
    input_set_generator: BaseLammpsGenerator,
    directory: str | Path = ".",
    **kwargs,
):
    input_set = input_set_generator.get_input_set(structure, **kwargs)
    input_set.write_input(directory)

    
def trajectory_from_lammps_dump(dump_path, trajectory_format : str = 'pmg'):
    '''
    Read a LAMMPS dump file and return a pymatgen Trajectory object.
    
    Args:
        dump_path (str): Path to the LAMMPS dump file.
        
    Returns:
        Trajectory: Pymatgen Trajectory object.
    '''
    
    lammps_traj = read(dump_path, index=':')
    
    if trajectory_format == 'ase':
        return AseTrajectory(lammps_traj)
    
    if trajectory_format == 'pmg':
        
        frames = [AseAtomsAdaptor.get_structure(atoms) for atoms in lammps_traj]
        veclocities = [atom.get_velocities() for atom in lammps_traj]
        forces = [atom.get_forces() for atom in lammps_traj]
        #TODO: add velocities and forces to the frame properties
        #TODO: account for molecules instead of structures
    return PmgTrajectory.from_structures(frames)