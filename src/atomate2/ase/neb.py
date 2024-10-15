""" Create NEB jobs with ASE. """

from __future__ import annotations

from dataclasses import dataclass, field

from ase.mep.neb import NEB
from pymatgen.core import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from atomate2.ase.jobs import AseMaker
from atomate2.common.schemas.neb import NebResult

# Parameters chosen for consistency with atomate2.vasp.sets.core.NebSetGenerator
_DEFAULT_NEB_KWARGS = {
    "k": 5., 
    "climb": True,
    "method": "improvedtangent"
}


@dataclass
class AseNebMaker(AseMaker):
    """Define ASE NEB jobs."""

    name: str = "ASE NEB maker"
    neb_kwargs : dict = field(default_factory=dict)

    def run_ase(
        self,
        images: list[Structure | Molecule],
        prev_dir: str | Path | None = None,
    ) -> NebResult:
        """
        Run ASE, method to be implemented in subclasses.

        This method exists to permit subclasses to redefine `make`
        for different output schemas.

        Parameters
        ----------
        mol_or_struct: .Molecule or .Structure
            pymatgen molecule or structure
        prev_dir : str or Path or None
            A previous calculation directory to copy output files from. Unused, just
                added to match the method signature of other makers.
        """

        self.neb_kwargs = self.neb_kwargs or _DEFAULT_NEB_KWARGS

        is_mol_calc = all(isinstance(image,Molecule) for image in images)
        
        images = [
            image.to_ase_atoms() for image in images
        ]
        
        neb_calc = NEB(images,**self.neb_kwargs)
        for image in images:
            image.calc = self.calculator

        with contextlib.redirect_stdout(sys.stdout if verbose else io.StringIO()):
            obs = TrajectoryObserver(atoms)
            if self.relax_cell and (not is_mol):
                atoms = cell_filter(atoms)
            optimizer = self.opt_class(atoms, **kwargs)
            optimizer.attach(obs, interval=interval)
            t_i = time.perf_counter()
            optimizer.run(fmax=fmax, steps=steps)
            t_f = time.perf_counter()
            obs()
        if traj_file is not None:
            obs.save(traj_file)
        if isinstance(atoms, cell_filter):
            atoms = atoms.atoms

        struct = self.ase_adaptor.get_structure(
            atoms, cls=Molecule if is_mol else Structure
        )
        traj = obs.to_pymatgen_trajectory(None)
        is_force_conv = all(
            np.linalg.norm(traj.frame_properties[-1]["forces"][idx]) < abs(fmax)
            for idx in range(len(struct))
        )
        return AseResult(
            final_mol_or_struct=struct,
            trajectory=traj,
            is_force_converged=is_force_conv,
            energy_downhill=traj.frame_properties[-1]["energy"]
            < traj.frame_properties[0]["energy"],
            dir_name=os.getcwd(),
            elapsed_time=t_f - t_i,
        )
        