"""Jobs used in the calculation of surface adsorption energy."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from jobflow import Flow, Response, job
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Element, Molecule, Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.vasp import Kpoints

from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.sets.core import StaticSetGenerator

if TYPE_CHECKING:
    from atomate2.vasp.sets.base import VaspInputGenerator

from collections import defaultdict

logger = logging.getLogger(__name__)


@job
def get_boxed_molecule(molecule: Molecule) -> Structure:
    """Get the molecule structure.

    Parameters
    ----------
    molecule: Molecule
        The molecule to be adsorbed.

    Returns
    -------
    Structure
        The molecule structure.
    """
    return molecule.get_boxed_structure(10, 10, 10, offset=np.array([5, 5, 5]))


@job
def remove_adsorbate(slab: Structure) -> Structure:
    """
    Remove adsorbate from the given slab.

    Parameters
    ----------
    slab: Structure
        A pymatgen Slab object, potentially with adsorbates.

    Returns
    -------
    Structure
        The modified slab with adsorbates removed.
    """
    adsorbate_indices = []
    for i, site in enumerate(slab):
        if site.properties.get("surface_properties") == "adsorbate":
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
    min_slab_size: float,
    surface_idx: tuple,
    min_vacuum_size: float,
    min_lw: float,
) -> Structure:
    """Generate the adsorption slabs without adsorbates.

    Parameters
    ----------
    bulk_structure: Structure
        The bulk/unit cell structure.
    min_slab_size: float
        The minimum size of the slab. In Angstroms or number of hkl planes.
        See pymatgen.core.surface.SlabGenerator for more details.
    surface_idx: tuple
        The Miller index [h, k, l] of the surface.
    min_vacuum_size: float
        The minimum size of the vacuum region. In Angstroms or number of hkl planes.
        See pymatgen.core.surface.SlabGenerator for more details.
    min_lw: float
        The minimum length and width of the slab.

    Returns
    -------
    Structure
        The slab structure without adsorbates.
    """
    hydrogen = Molecule([Element("H")], [[0, 0, 0]])
    slab_generator = SlabGenerator(
        bulk_structure,
        surface_idx,
        min_slab_size,
        min_vacuum_size,
        primitive=False,
        center_slab=True,
    )
    temp_slab = slab_generator.get_slab()
    ads_slabs = AdsorbateSiteFinder(temp_slab).generate_adsorption_structures(
        hydrogen, translate=True, min_lw=min_lw
    )
    slab_only = remove_adsorbate(ads_slabs[0])

    return slab_only  # noqa: RET504


@job(data=[Structure])
def generate_adslabs(
    bulk_structure: Structure,
    molecule_structure: Structure,
    min_slab_size: float,
    surface_idx: tuple,
    min_vacuum_size: float,
    min_lw: float,
) -> list[Structure]:
    """Generate the adsorption slabs with adsorbates.

    Parameters
    ----------
    bulk_structure: Structure
        The bulk/unit cell structure.
    molecule_structure: Structure
        The molecule to be adsorbed.
    min_slab_size: float
        The minimum size of the slab. In Angstroms or number of hkl planes.
    surface_idx: tuple
        The Miller index [h, k, l] of the surface.
    min_vacuum_size: float
        The minimum size of the vacuum region. In Angstroms or number of hkl planes.
    min_lw: float
        The minimum length and width of the slab.

    Returns
    -------
    list[Structure]
        The list of all possible configurations of slab structures with adsorbates.
    """
    slab_generator = SlabGenerator(
        bulk_structure,
        surface_idx,
        min_slab_size,
        min_vacuum_size,
        primitive=False,
        center_slab=True,
    )
    slab = slab_generator.get_slab()
    ads_slabs = AdsorbateSiteFinder(slab).generate_adsorption_structures(
        molecule_structure, translate=True, min_lw=min_lw
    )
    return ads_slabs  # noqa: RET504


@job
def run_adslabs_job(
    adslab_structures: list[Structure],
    relax_maker: SlabRelaxMaker,
    static_maker: SlabStaticMaker,
) -> Flow:
    """Workflow of running the adsorption slab calculations.

    Parameters
    ----------
    adslab_structures: list[Structure]
        The list of all possible configurations of slab structures with adsorbates.
    relax_maker: AdslabRelaxMaker
        The relaxation maker for the adsorption slab structures.
    static_maker: SlabStaticMaker
        The static maker for the adsorption slab structures.

    Returns
    -------
    Flow
        The flow of the adsorption slab calculations.
    """
    adsorption_jobs = []
    ads_outputs = defaultdict(list)

    for i, ad_structure in enumerate(adslab_structures):
        ads_job = relax_maker.make(ad_structure)
        ads_job.append_name(f"configuration {i}")

        adsorption_jobs.append(ads_job)
        ads_outputs["configuration_number"].append(i)
        ads_outputs["relaxed_structures"].append(ads_job.output.structure)

        static_job = static_maker.make(ads_job.output.structure)
        static_job.append_name(f"static configuration {i}")
        adsorption_jobs.append(static_job)

        ads_outputs["static_energy"].append(static_job.output.energy)
        ads_outputs["dirs"].append(ads_job.output.dir_name)

    adsorption_flow = Flow(adsorption_jobs, ads_outputs)
    return Response(replace=adsorption_flow)


@job
def adsorption_calculations(
    # bulk_structure: Structure,
    # molecule_structure: Structure,
    # surface_idx: tuple,
    adslab_structures: list[Structure],
    ads_outputs: dict[str, list],
    molecule_dft_energy: float,
    slab_dft_energy: float,
) -> list:
    """Calculate the adsorption energies by subtracting the energies of
    the adsorbate, slab, and adsorbate-slab.

    Parameters
    ----------
    bulk_structure: Structure
        The bulk/unit cell structure.
    molecule_structure: Structure
        The molecule to be adsorbed.
    surface_idx: tuple
        The Miller index [h, k, l] of the surface.
    adslab_structures: list[Structure]
        The list of all possible configurations of slab structures with adsorbates.
    ads_outputs: dict[str, list]
        The outputs of the adsorption calculations.
    molecule_dft_energy: float
        The static energy of the molecule.
    slab_dft_energy: float
        The static energy of the slab.

    Returns
    -------
    list
        The list of (adsorption configurations, configuration number,
        adsorption energy, directories) sorted by adsorption energy.
    """  # noqa: D205
    # bulk_composition = bulk_structure.composition
    # bulk_reduced_formula = bulk_composition.reduced_formula
    # molecule_composition = molecule_structure.composition
    # molecule_reduced_formula = molecule_composition.reduced_formula
    # flow_name = f"{bulk_reduced_formula}_{molecule_reduced_formula}_{surface_idx}"

    outputs: dict[str, list] = {
        "adsorption_configuration": [],
        "configuration_number": [],
        "adsorption_energy": [],
        "dirs": [],
    }

    for i, ad_structure in enumerate(adslab_structures):
        outputs["adsorption_configuration"].append(ad_structure)
        outputs["configuration_number"].append(i)
        ads_energy = (
            ads_outputs["static_energy"][i] - molecule_dft_energy - slab_dft_energy
        )
        outputs["adsorption_energy"].append(ads_energy)
        outputs["dirs"].append(ads_outputs["dirs"][i])

        sorted_outputs = sorted(
            zip(
                outputs["adsorption_configuration"],
                outputs["configuration_number"],
                outputs["adsorption_energy"],
                outputs["dirs"],
            ),
            key=lambda x: x[2],
        )

    return sorted_outputs


@dataclass
class BulkRelaxMaker(BaseVaspMaker):
    """Maker for molecule relaxation.

    Parameters
    ----------
    name: str
        The name of the flow produced by the maker.
    input_set_generator: VaspInputGenerator
        The input set generator for the relaxation calculation.
    """

    name: str = "bulk relaxation"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: StaticSetGenerator(
            user_kpoints_settings=Kpoints.from_dict(
                {
                    "nkpoints": 0,
                    "generation_style": "Gamma",
                    "kpoints": [[11, 11, 11]],
                    "usershift": [0, 0, 0],
                    "comment": "Automatic mesh",
                }
            ),
            user_incar_settings={
                "ALGO": "Normal",
                "IBRION": 2,
                "ISIF": 3,
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
class MolRelaxMaker(BaseVaspMaker):
    """Maker for molecule relaxation.

    Parameters
    ----------
    name: str
        The name of the flow produced by the maker.
    input_set_generator: VaspInputGenerator
        The input set generator for the relaxation calculation.
    """

    name: str = "molecule relaxation"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: StaticSetGenerator(
            user_kpoints_settings=Kpoints.from_dict(
                {
                    "nkpoints": 0,
                    "generation_style": "Gamma",
                    "kpoints": [[11, 11, 11]],
                    "usershift": [0, 0, 0],
                    "comment": "Automatic mesh",
                }
            ),
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
class SlabRelaxMaker(BaseVaspMaker):
    """Maker for adsorption slab relaxation.

    Parameters
    ----------
    name: str
        The name of the flow produced by the maker.
    input_set_generator: VaspInputGenerator
        The input set generator for the relaxation calculation.
    """

    name: str = "adsorption relaxation"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: StaticSetGenerator(
            user_kpoints_settings=Kpoints.from_dict(
                {
                    "nkpoints": 0,
                    "generation_style": "Gamma",
                    "kpoints": [[3, 3, 1]],
                    "usershift": [0, 0, 0],
                    "comment": "Automatic mesh",
                }
            ),
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
    """Maker for slab static energy calculation.

    Parameters
    ----------
    name: str
        The name of the flow produced by the maker.
    input_set_generator: VaspInputGenerator
        The input set generator for the static energy calculation.
    """

    name: str = "adsorption static calculation"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: StaticSetGenerator(
            user_kpoints_settings=Kpoints.from_dict(
                {
                    "nkpoints": 0,
                    "generation_style": "Gamma",
                    "kpoints": [[3, 3, 1]],
                    "usershift": [0, 0, 0],
                    "comment": "Automatic mesh",
                }
            ),
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
    """Maker for molecule static energy calculation.

    Parameters
    ----------
    name: str
        The name of the flow produced by the maker.
    input_set_generator: VaspInputGenerator
        The input set generator for the static energy calculation.
    """

    name: str = "molecule static calculation"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: StaticSetGenerator(
            user_kpoints_settings=Kpoints.from_dict(
                {
                    "nkpoints": 0,
                    "generation_style": "Gamma",
                    "kpoints": [[11, 11, 11]],
                    "usershift": [0, 0, 0],
                    "comment": "Automatic mesh",
                }
            ),
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
