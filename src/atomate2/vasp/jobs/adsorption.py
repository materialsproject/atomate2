"""Jobs used in the calculation of surface adsorption energy."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Response, job
from pymatgen.core import Structure, Molecule, Element

from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.vasp import Kpoints

from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.sets.core import StaticSetGenerator

if TYPE_CHECKING:
    from pathlib import Path
    import numpy as np
    from atomate2.vasp.sets.base import VaspInputGenerator

logger = logging.getLogger(__name__)


@job
def get_boxed_molecule(molecule: Molecule) -> Structure:
    """Get the molecule structure."""
    return molecule.get_boxed_structure(10, 10, 10, offset=np.array([5, 5, 5]))


@job
def removeAdsorbate(slab):
    """
    Remove adsorbate from the given slab.

    Parameters:
    slab (Slab): A pymatgen Slab object, potentially with adsorbates.

    Returns:
    Slab: The modified slab with adsorbates removed.
    """
    adsorbate_indices = []
    for i, site in enumerate(slab):
        if site.properties.get('surface_properties') == 'adsorbate':
            adsorbate_indices.append(i)
    # Reverse the indices list to avoid index shifting after removing sites
    adsorbate_indices.reverse()
    # Remove the adsorbate sites
    for idx in adsorbate_indices:
        slab.remove_sites([idx])
    return slab

@job(data=[Structure])
def generate_slab(
    bulk_structure: Structure,
    min_slab_size: int,
    surface_idx,
    min_vacuum_size: int,
    min_lw: float,
) -> Structure:
    """Generate the adsorption slabs."""

    H = Molecule([Element("H")], [[0, 0, 0]])
    slab_generator = SlabGenerator(bulk_structure, surface_idx, min_slab_size, min_vacuum_size, primitive=False,
                                   center_slab=True)
    temp_slab = slab_generator.get_slab()
    ads_slabs = AdsorbateSiteFinder(temp_slab).generate_adsorption_structures(H, translate=True, min_lw=min_lw)
    slabOnly = removeAdsorbate(ads_slabs[0])

    return slabOnly

@job(data=[Structure])
def generate_adslabs(
    bulk_structure: Structure,
    molecule_structure: Structure,
    min_slab_size: int,
    surface_idx,
    min_vacuum_size: int,
    min_lw: float,
) -> list[Structure]:
    """Generate the adsorption slabs."""

    slab_generator = SlabGenerator(bulk_structure, surface_idx, min_slab_size, min_vacuum_size, primitive=False,
                                   center_slab=True)
    slab = slab_generator.get_slab()
    ads_slabs = AdsorbateSiteFinder(slab).generate_adsorption_structures(molecule_structure, translate=True,
                                                                         min_lw=min_lw)
    return ads_slabs

@job
def run_adslabs_job(
    adslab_structures: list[Structure],
    relax_maker: AdslabRelaxMaker,
    prev_dir: str | Path | None = None,
    ) -> Flow:

    adsorption_jobs = []
    ads_outputs: dict[str, list] = {
        "job_name": "",
        "configuration_number": [],
        "adsorption_energy": [],
        "dirs": [],
    }

    for i, ad_structure in enumerate(adslab_structures):
        if prev_dir is not None:
            ads_job = relax_maker.make(ad_structure, prev_dir=prev_dir)
        else:
            ads_job = relax_maker.make(ad_structure)
        ads_job.append_name(f"configuration {i}")

        adsorption_jobs.append(ads_job)
        ads_outputs["configuration_number"].append(i)
        ads_outputs["relaxed_structures"].append(i)
        ads_outputs["adsorption_energy"].append(ads_job.output.structure)
        ads_outputs["dirs"].append(ads_job.output.dir_name)

    adsorption_flow = Flow(ads_job, ads_outputs)
    return Response(replace=adsorption_flow)

@job
def adsorption_calculations(
        bulk_structure: Structure,
        molecule_structure: Structure,
        surface_idx,
        adslab_structures: list[Structure],
        ads_outputs: dict[str, list],
        molecule_dft_energy: float,
        slab_dft_energy: float,
):
    """Calculating the adsorption energies."""

    bulk_composition = bulk_structure.composition
    bulk_reduced_formula = bulk_composition.reduced_formula
    molecule_composition = molecule_structure.composition
    molecule_reduced_formula = molecule_composition.reduced_formula
    flow_name = f"{bulk_reduced_formula}_{molecule_reduced_formula}_{surface_idx}"

    outputs: dict[str, list] = {
        "job_name": f"{flow_name}_adsorption_calculations",
        "adsorption_configuration": [],
        "configuration_number": [],
        "adsorption_energy": [],
        "dirs": [],
    }

    for i in ads_outputs:
        outputs["adsorption_configuration"].append(adslab_structures[i])
        outputs["configuration_number"].append(ads_outputs["configuration_number"][i])
        ads_energy = ads_outputs["adsorption_energy"][i] - molecule_dft_energy - slab_dft_energy
        outputs["adsorption_energy"].append(ads_energy)
        outputs["dirs"].append(ads_outputs["dirs"][i])

    return outputs


@dataclass
class MoleculeRelaxMaker(BaseVaspMaker):
    name: str = "adsorption relaxation"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: StaticSetGenerator(
            user_kpoints_settings=Kpoints.from_dict({
                'nkpoints': 0,
                'generation_style': 'Gamma',
                'kpoints': [[11, 11, 11]],
                'usershift': [0, 0, 0],
                'comment': 'Automatic mesh'}),
            user_incar_settings={
                "ALGO": "Normal",
                "IBRION": 2,
                "ISIF": 2,
                "ENCUT": 700,
                "GGA": "RP",
                "EDIFF": 1e-5,
                "LAECHG": False,
                "LREAL": False,
                "LCHARG": False,
                "LDAU": False,
                "NSW": 300,
                "NELM": 500,
            },
            auto_ispin=True,
        )
    )

@dataclass
class AdslabRelaxMaker(BaseVaspMaker):
    name: str = "adsorption relaxation"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: StaticSetGenerator(
            user_kpoints_settings=Kpoints.from_dict({
                'nkpoints': 0,
                'generation_style': 'Gamma',
                'kpoints': [[3, 3, 1]],
                'usershift': [0, 0, 0],
                'comment': 'Automatic mesh'}),
            user_incar_settings={
                "ALGO": "Normal",
                "IBRION": 2,
                "ISIF": 2,
                "ENCUT": 700,
                "GGA": "RP",
                "EDIFF": 1e-5,
                "LAECHG": False,
                "LREAL": False,
                "LCHARG": False,
                "LDAU": False,
                "NSW": 300,
                "NELM": 500,
            },
            auto_ispin=True,
        )
    )

@dataclass
class SlabStaticMaker(BaseVaspMaker):
    name: str = "adsorption static calculation"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: StaticSetGenerator(
            user_kpoints_settings=Kpoints.from_dict({
                'nkpoints': 0,
                'generation_style': 'Gamma',
                'kpoints': [[3, 3, 1]],
                'usershift': [0, 0, 0],
                'comment': 'Automatic mesh'}),
            user_incar_settings={
                "ALGO": "Normal",
                "ENCUT": 700,
                "GGA": "RP",
                "EDIFF": 1e-7,
                "LAECHG": False,
                "LREAL": False,
                "LCHARG": False,
                "LDAU": False,
                "NELM": 500,
            },
            auto_ispin=True,
        )
    )

@dataclass
class MolStaticMaker(BaseVaspMaker):
    name: str = "adsorption static calculation"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: StaticSetGenerator(
            user_kpoints_settings=Kpoints.from_dict({
                'nkpoints': 0,
                'generation_style': 'Gamma',
                'kpoints': [[11, 11, 11]],
                'usershift': [0, 0, 0],
                'comment': 'Automatic mesh'}),
            user_incar_settings={
                "ALGO": "Normal",
                "ENCUT": 700,
                "GGA": "RP",
                "EDIFF": 1e-7,
                "LAECHG": False,
                "LREAL": False,
                "LCHARG": False,
                "LDAU": False,
                "NELM": 500,
            },
            auto_ispin=True,
        )
    )