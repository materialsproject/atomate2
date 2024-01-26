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
    generate_slab,
    generate_adslabs,
    run_slab_job,
    run_adslabs_job,
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
            min_vacuum: float = 20.0,
            min_slab_size: float = 10.0,
            min_lw: float = 10.0.
            surface_idx,
            prev_dir: str | Path | None = None,
            molecule_dft_energy: float | None = None,
    ) -> Flow:

        jobs = []
        outputs: dict[str, list] = {
            "configuration_number": [],
            "adsorption_energy": [],
            "dirs": [],
        }

        molecule_structure = get_boxed_molecule(molecule)

        if self.molecule_dft_energy is None:
            molOptimize = self.mol_relax_maker.make(molecule_structure, prev_dir=prev_dir)
            molOptimize.append_name(f"molecule relaxation job")
            jobs.append(molOptimize)
            optimized_molecule = molOptimize.output.structure

            mol_static_job = StaticMaker.make(molecule_structure)
            mol_static_job.append_name(f"molecule static job")
            jobs.append(mol_static_job)
            self.molecule_dft_energy = mol_static_job.output.output.energy

        bulkOptimize = self.bulk_relax_maker.make(structure, prev_dir=prev_dir)
        bulkOptimize.append_name(f"bulk relaxation job")
        jobs.append(bulkOptimize)
        optimized_bulk = bulkOptimize.output.structure

        generate_slab_structure = generate_slab(
            bulk_structure=optimized_bulk,
            min_slab_size=self.min_slab_size,
            surface_idx=self.surface_idx,
            min_vacuum_size=self.min_vacuum,
            min_lw=self.min_lw,
        )
        jobs.append(generate_slab_structure)

        generate_adslabs_structures = generate_adslabs(
            bulk_structure=optimized_bulk,
            molecule_structure=optimized_molecule,
            min_slab_size=self.min_slab_size,
            surface_idx=self.surface_idx,
            min_vacuum_size=self.min_vacuum,
            min_lw=self.min_lw,
        )
        jobs.append(generate_adslabs_structures)


        vasp_slab_calcs = run_slab_job(
            optimized_bulk,
            optimized_molecule,
            self.supercell_idx,
            self.surface_idx,
            self.prefer_90_degrees,
            self.min_vacuum,
            self.min_ads_length,
            self.include_slab
        )
        jobs.append(vasp_slab_calcs)

        vasp_adslabs_calcs = run_adslabs_job(
            optimized_bulk,
            optimized_molecule,
            self.supercell_idx,
            self.surface_idx,
            self.prefer_90_degrees,
            self.min_vacuum,
            self.min_ads_length,
            self.include_slab
        )
        jobs.append(vasp_adslabs_calcs)

        adsorption_calc = run_adsorption_calculations()
        jobs.append(adsorption_calc)

        return Flow(jobs, adsorption_calc.output)

