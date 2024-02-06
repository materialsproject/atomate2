"""Flow for calculating surface adsorption energies."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker

from atomate2.vasp.jobs.adsorption import (
    get_boxed_molecule,
    AdslabRelaxMaker,
    SlabStaticMaker,
    MolStaticMaker,
    MoleculeRelaxMaker,
    generate_slab,
    generate_adslabs,
    run_adslabs_job,
    adsorption_calculations
)

from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.core import TightRelaxMaker

if TYPE_CHECKING:
    from pathlib import Path
    from pymatgen.core.structure import Structure, Molecule
    from atomate2.vasp.jobs.base import BaseVaspMaker

@dataclass
class AdsorptionMaker(Maker):
    """Makes a flow for calculating adsorption energies."""
    name: str = "adsorption"
    get_supercell_size_kwargs: dict = field(default_factory=dict)


    bulk_relax_maker: BaseVaspMaker = field(
        default_factory=lambda: DoubleRelaxMaker.from_relax_maker(TightRelaxMaker())
    )
    mol_relax_maker: BaseVaspMaker = field(default_factory=MoleculeRelaxMaker())

    adslab_relax_maker: BaseVaspMaker = field(default_factory=AdslabRelaxMaker())

    static_energy_maker: BaseVaspMaker = field(default_factory=StaticMaker())

    def __init__(self):
        self.molecule_dft_energy = None

    def make(
            self,
            molecule: Molecule,
            structure: Structure,
            min_vacuum: float = 20.0,
            min_slab_size: float = 10.0,
            min_lw: float = 10.0,
            surface_idx = (0, 0, 1),
            prev_dir: str | Path | None = None,
            molecule_dft_energy: float | None = None,
            slab_dft_energy: float | None = None,
    ) -> Flow:

        jobs = []

        molecule_structure = get_boxed_molecule(molecule)

        if self.molecule_dft_energy is None:
            mol_optimize_job = self.mol_relax_maker.make(molecule_structure, prev_dir=prev_dir)
            mol_optimize_job.append_name(f"molecule relaxation job")
            jobs.append(mol_optimize_job)
            optimized_molecule = mol_optimize_job.output.structure

            mol_static_job = MolStaticMaker.make(molecule_structure)
            mol_static_job.append_name(f"molecule static job")
            jobs.append(mol_static_job)
            self.molecule_dft_energy = mol_static_job.output.output.energy

        bulk_optimize_job = self.bulk_relax_maker.make(structure, prev_dir=prev_dir)
        bulk_optimize_job.append_name(f"bulk relaxation job")
        jobs.append(bulk_optimize_job)
        optimized_bulk = bulk_optimize_job.output.structure

        generate_slab_structure = generate_slab(
            bulk_structure=optimized_bulk,
            min_slab_size=self.min_slab_size,
            surface_idx=self.surface_idx,
            min_vacuum_size=self.min_vacuum,
            min_lw=self.min_lw,
        )
        jobs.append(generate_slab_structure)

        if self.slab_dft_energy is None:
            slab_optimize_job = self.adslab_relax_maker.make(generate_slab_structure, prev_dir=prev_dir)
            slab_optimize_job.append_name(f"slab relaxation job")
            jobs.append(slab_optimize_job)
            optimized_slab = slab_optimize_job.output.structure

            slab_static_job = SlabStaticMaker.make(optimized_slab)
            slab_static_job.append_name(f"slab static job")
            jobs.append(slab_static_job)
            self.slab_dft_energy = slab_static_job.output.output.energy

        generate_adslabs_structures = generate_adslabs(
            bulk_structure=optimized_bulk,
            molecule_structure=optimized_molecule,
            min_slab_size=self.min_slab_size,
            surface_idx=self.surface_idx,
            min_vacuum_size=self.min_vacuum,
            min_lw=self.min_lw,
        )
        jobs.append(generate_adslabs_structures)

        vasp_adslabs_calcs = run_adslabs_job(
            adslab_structures=generate_adslabs_structures.output,
            relax_maker = self.adslab_relax_maker,
            prev_dir=prev_dir,
        )
        jobs.append(vasp_adslabs_calcs)

        adsorption_calc = adsorption_calculations(
            bulk_structure=optimized_bulk,
            molecule_structure=optimized_molecule,
            surface_idx=self.surface_idx,
            adslab_structures=generate_adslabs_structures.output,
            adslabs_data=vasp_adslabs_calcs.output,
            molecule_dft_energy=self.molecule_dft_energy,
            slab_dft_energy=self.slab_dft_energy,
        )
        jobs.append(adsorption_calc)

        return Flow(jobs, adsorption_calc.output)