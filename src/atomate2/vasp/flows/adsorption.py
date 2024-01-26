"""Flow for calculating surface adsorption energies."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker

from atomate2.vasp.jobs.adsorption import (
    get_boxed_molecule,
    adslabRelaxMaker,
    StaticMaker,
    moleculeRelaxMaker,
    run_adslab_jobs,
    run_adsorption_calculations
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
    min_ads_distance: float = 2.0
    min_vacuum: float = 20.0
    min_slab_size: float = 10.0
    min_lw: float = 10.0
    prefer_90_degrees: bool = True
    get_supercell_size_kwargs: dict = field(default_factory=dict)

    bulk_relax_maker: BaseVaspMaker = field(
        default_factory=lambda: DoubleRelaxMaker.from_relax_maker(TightRelaxMaker())
    )
    mol_relax_maker: BaseVaspMaker = field(default_factory=moleculeRelaxMaker())

    adslab_relax_maker: BaseVaspMaker = field(default_factory=adslabRelaxMaker())

    static_energy_maker: BaseVaspMaker = field(default_factory=StaticMaker())

    def make(
            self,
            molecule: Molecule,
            structure: Structure,
            supercell_vector: [],
            surface_vector: [],
            prev_dir: str | Path | None = None,
            molecule_dft_energy: float | None = None,
    ) -> Flow:

        jobs = []

        molecule_structure = get_boxed_molecule(molecule)

        if self.molecule_dft_energy is None:
            molOptimize = self.mol_relax_maker.make(molecule_structure, prev_dir=prev_dir)
            jobs.append(molOptimize)
            optimized_molecule = molOptimize.output.structure

            mol_static_job = StaticMaker.make(molecule_structure)
            jobs.append(mol_static_job)
            self.molecule_dft_energy = mol_static_job.output.output.energy

        bulkOptimize = self.bulk_relax_maker.make(structure, prev_dir=prev_dir)
        jobs.append(bulkOptimize)
        optimized_bulk = bulkOptimize.output.structure

        generate_slab = generate_slab()

        generate_adslabs = generate_adslabs()


        adslab_jobs = run_adslab_jobs(
            optimized_bulk,
            optimized_molecule,
            self.supercell_idx,
            self.surface_idx,
            self.prefer_90_degrees,
            self.min_vacuum,
            self.min_ads_length,
            self.include_slab
        )
        jobs.append(adslab_jobs)

        adsorption_calc = run_adsorption_calculations()
        jobs.append(adsorption_calc)

        return Flow(jobs, adsorption_calc.output)

