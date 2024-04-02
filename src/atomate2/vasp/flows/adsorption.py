"""Flow for calculating surface adsorption energies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Job, Maker

from atomate2.vasp.jobs.adsorption import (
    BulkRelaxMaker,
    MolRelaxMaker,
    MolStaticMaker,
    SlabRelaxMaker,
    SlabStaticMaker,
    adsorption_calculations,
    generate_adslabs,
    generate_slab,
    get_boxed_molecule,
    run_adslabs_job,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core.structure import Molecule, Structure

    from atomate2.vasp.jobs.base import BaseVaspMaker


@dataclass
class AdsorptionMaker(Maker):
    """
    Workflow that calculates the adsorption energy of a molecule on a surface.

    The flow consists of the following steps:
    1. Optimize the molecule structure and calculate its static energy.
    2. Optimize the bulk structure.
    3. Generate a slab structure using the optimized bulk structure and calculate its static energy.
    4. Generate adsorption sites on the slab and calculate corresponding static energy.
    5. Calculate the adsorption energy by calculating the energy difference between the slab with
    adsorbed molecule and the sum of the slab without the adsorbed molecule and the molecule.

    Parameters
    ----------
    name: str
        Name of the flow.
    bulk_relax_maker: BaseVaspMaker
        Maker for bulk relaxation.
    mol_relax_maker: BaseVaspMaker
        Maker for molecule relaxation.
    slab_relax_maker: BaseVaspMaker
        Maker for slab relaxation with adsorption.
    slab_static_maker: BaseVaspMaker
        Maker for slab static energy calculation.
    mol_static_energy_maker: BaseVaspMaker
        Maker for molecule static energy calculation.
    """  # noqa: E501

    name: str = "adsorption workflow"

    mol_relax_maker: BaseVaspMaker = field(default_factory=MolRelaxMaker)

    mol_static_energy_maker: BaseVaspMaker = field(default_factory=MolStaticMaker)

    bulk_relax_maker: BaseVaspMaker = field(default_factory=BulkRelaxMaker)

    slab_relax_maker: BaseVaspMaker = field(default_factory=SlabRelaxMaker)

    slab_static_maker: BaseVaspMaker = field(default_factory=SlabStaticMaker)

    def make(
        self,
        molecule: Molecule,
        structure: Structure,
        min_vacuum: float = 20.0,
        min_slab_size: float = 10.0,
        min_lw: float = 10.0,
        surface_idx: tuple[int, int, int] = (0, 0, 1),
        prev_dir_mol: str | Path | None = None,
        prev_dir_bulk: str | Path | None = None,
    ) -> Flow:
        """
        Generate a flow for calculating adsorption energies.

        Parameters
        ----------
        molecule: Molecule
            A pymatgen molecule object. The molecule to be adsorbed.
        structure: Structure
            A pymatgen structure object. The bulk structure to be used for slab generation.
        min_vacuum: float
            The minimum size of the vacuum region. In Angstroms or number of hkl planes.
        min_slab_size: float
            The minimum size of layers of the slab. In Angstroms or number of hkl planes.
        min_lw: float
            Minimum length and width of the slab
        surface_idx: tuple
            Miller index [h, k, l] of plane parallel to surface.
        prev_dir_mol: str or Path or None
            A previous VASP calculation directory to copy output files from.
        prev_dir_bulk: str or Path or None
            A previous VASP calculation directory to copy output files from.

        Returns
        -------
        Flow
            A flow object for calculating adsorption energies.
        """  # noqa: E501
        jobs: list[Job] = []
        molecule_structure = get_boxed_molecule(molecule)

        if self.mol_relax_maker:
            mol_optimize_job = self.mol_relax_maker.make(
                molecule_structure, prev_dir=prev_dir_mol
            )
            mol_optimize_job.append_name("molecule relaxation job")
            jobs += [mol_optimize_job]

            optimized_molecule = mol_optimize_job.output.structure
            prev_dir = mol_optimize_job.output.dir_name
        else:
            prev_dir = prev_dir_mol
            optimized_molecule = molecule_structure

        mol_static_job = self.mol_static_energy_maker.make(
            molecule_structure, prev_dir=prev_dir
        )
        mol_static_job.append_name("molecule static job")
        jobs += [mol_static_job]

        molecule_dft_energy = mol_static_job.output.output.energy

        if self.bulk_relax_maker:
            bulk_optimize_job = self.bulk_relax_maker.make(
                structure, prev_dir=prev_dir_bulk
            )
            bulk_optimize_job.append_name("bulk relaxation job")
            jobs += [bulk_optimize_job]

            optimized_bulk = bulk_optimize_job.output.structure
        else:
            optimized_bulk = structure

        generate_slab_structure = generate_slab(
            bulk_structure=optimized_bulk,
            min_slab_size=min_slab_size,
            surface_idx=surface_idx,
            min_vacuum_size=min_vacuum,
            min_lw=min_lw,
        )
        jobs += [generate_slab_structure]

        generate_adslabs_structures = generate_adslabs(
            bulk_structure=optimized_bulk,
            molecule_structure=optimized_molecule,
            min_slab_size=min_slab_size,
            surface_idx=surface_idx,
            min_vacuum_size=min_vacuum,
            min_lw=min_lw,
        )
        jobs += [generate_adslabs_structures]

        if self.slab_relax_maker is None:
            raise ValueError("adslab_relax_maker shouldn't be Null.")

        # slab relaxation without adsoprtion
        slab_optimize_job = self.slab_relax_maker.make(generate_slab_structure)
        slab_optimize_job.append_name("slab relaxation job")
        jobs += [slab_optimize_job]

        optimized_slab = slab_optimize_job.output.structure
        prev_dir = slab_optimize_job.output.dir_name

        slab_static_job = self.slab_static_maker.make(optimized_slab, prev_dir=prev_dir)
        slab_static_job.append_name("slab static job")
        jobs += [slab_static_job]

        slab_dft_energy = slab_static_job.output.output.energy

        vasp_adslabs_calcs = run_adslabs_job(
            adslab_structures=generate_adslabs_structures,
            relax_maker=self.slab_relax_maker,
            static_maker=self.slab_static_maker,
        )
        jobs += [vasp_adslabs_calcs]

        adsorption_calc = adsorption_calculations(
            # bulk_structure=optimized_bulk,
            # molecule_structure=optimized_molecule,
            # surface_idx=surface_idx,
            adslab_structures=generate_adslabs_structures,
            adslabs_data=vasp_adslabs_calcs.output,
            molecule_dft_energy=molecule_dft_energy,
            slab_dft_energy=slab_dft_energy,
        )
        jobs += [adsorption_calc]

        return Flow(jobs, output=adsorption_calc.output, name=self.name)
