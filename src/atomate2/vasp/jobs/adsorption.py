"""Jobs used in the calculation of surface adsorption energy."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Response, job
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Element, Molecule, Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.vasp import Kpoints
from pymatgen.io.vasp.sets import MPRelaxSet

from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.schemas.adsorption import AdsorptionDocument

# from atomate2.vasp.sets.core import RelaxSetGenerator, StaticSetGenerator

if TYPE_CHECKING:
    from atomate2.vasp.sets.base import VaspInputGenerator

logger = logging.getLogger(__name__)


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

    name: str = "bulk_relax_maker__"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: MPRelaxSet(
            user_kpoints_settings={"reciprocal_density": 60},
            user_incar_settings={
                "ISIF": 3,
                "ENCUT": 700,
                "GGA": "RP",
                "EDIFF": 1e-5,
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

    name: str = "mol_relax_maker__"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: MPRelaxSet(
            user_kpoints_settings=Kpoints.from_dict(
                {
                    "nkpoints": 0,
                    "generation_style": "Gamma",
                    "kpoints": [[1, 1, 1]],
                    "usershift": [0, 0, 0],
                    "comment": "Automatic mesh",
                }
            ),
            user_incar_settings={
                "ISIF": 2,
                "ENCUT": 700,
                "GGA": "RP",
                "EDIFF": 1e-5,
                "LDAU": False,
                "NSW": 300,
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

    name: str = "mol_static_maker__"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: MPRelaxSet(
            user_kpoints_settings=Kpoints.from_dict(
                {
                    "nkpoints": 0,
                    "generation_style": "Gamma",
                    "kpoints": [[1, 1, 1]],
                    "usershift": [0, 0, 0],
                    "comment": "Automatic mesh",
                }
            ),
            user_incar_settings={
                "ENCUT": 700,
                "IBRION": -1,
                "GGA": "RP",
                "EDIFF": 1e-7,
                "LDAU": False,
                "NSW": 0,
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

    name: str = "slab_relax_maker__"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: MPRelaxSet(
            user_kpoints_settings={"reciprocal_density": 60},
            user_incar_settings={
                "ISIF": 2,
                "ENCUT": 700,
                "GGA": "RP",
                "EDIFF": 1e-5,
                "LDAU": False,
                "NSW": 300,
                "NELM": 500,
            },
            auto_ispin=False,
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

    name: str = "slab_static_maker__"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: MPRelaxSet(
            user_kpoints_settings={"reciprocal_density": 100},
            user_incar_settings={
                "ENCUT": 700,
                "IBRION": -1,
                "GGA": "RP",
                "EDIFF": 1e-7,
                "LDAU": False,
                "NSW": 0,
                "NELM": 500,
            },
            auto_ispin=False,
        )
    )


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
    return molecule.get_boxed_structure(10, 10, 10)


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


@job
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
    return remove_adsorbate(ads_slabs[0])


@job
def generate_adslabs(
    bulk_structure: Structure,
    molecule_structure: Molecule,
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

    return AdsorbateSiteFinder(slab).generate_adsorption_structures(
        molecule_structure, translate=True, min_lw=min_lw
    )


@job
def run_adslabs_job(
    adslab_structures: list[Structure],
    relax_maker: SlabRelaxMaker,
    static_maker: SlabStaticMaker,
) -> Response:
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
        ads_job = relax_maker.make(structure=ad_structure, prev_dir=None)
        ads_job.append_name(f"adsconfig_{i}")

        adsorption_jobs.append(ads_job)
        ads_outputs["configuration_number"].append(i)
        ads_outputs["relaxed_structures"].append(ads_job.output.structure)

        prev_dir_ads = ads_job.output.dir_name

        static_job = static_maker.make(
            structure=ads_job.output.structure, prev_dir=prev_dir_ads
        )
        static_job.append_name(f"static_adsconfig_{i}")
        adsorption_jobs.append(static_job)

        ads_outputs["static_energy"].append(static_job.output.output.energy)
        ads_outputs["dirs"].append(ads_job.output.dir_name)

    ads_flow = Flow(adsorption_jobs, ads_outputs)
    return Response(replace=ads_flow)


@job(output_schema=AdsorptionDocument)
def adsorption_calculations(
    adslab_structures: list[Structure],
    adslabs_data: dict[str, list],
    molecule_dft_energy: float,
    slab_dft_energy: float,
) -> AdsorptionDocument:
    """Calculate the adsorption energies and return an AdsorptionDocument instance.

    Parameters
    ----------
    adslab_structures : list[Structure]
        The list of all possible configurations of slab structures with adsorbates.
    adslabs_data : dict[str, list]
        Dictionary containing static energies and directories of the adsorption slabs.
    molecule_dft_energy : float
        The static energy of the molecule.
    slab_dft_energy : float
        The static energy of the slab.

    Returns
    -------
    AdsorptionDocument
        An AdsorptionDocument instance containing all adsorption data.
    """
    adsorption_energies = []
    configuration_numbers = []
    job_dirs = []

    for idx in range(len(adslab_structures)):
        ads_energy = (
            adslabs_data["static_energy"][idx] - molecule_dft_energy - slab_dft_energy
        )
        adsorption_energies.append(ads_energy)
        configuration_numbers.append(idx)
        job_dirs.append(adslabs_data["dirs"][idx])

    # Sort the data by adsorption energy
    sorted_indices = sorted(
        range(len(adsorption_energies)), key=lambda k: adsorption_energies[k]
    )

    # Apply the sorted indices to all lists
    sorted_structures = [adslab_structures[i] for i in sorted_indices]
    sorted_configuration_numbers = [configuration_numbers[i] for i in sorted_indices]
    sorted_adsorption_energies = [adsorption_energies[i] for i in sorted_indices]
    sorted_job_dirs = [job_dirs[i] for i in sorted_indices]

    # Create and return the AdsorptionDocument instance
    return AdsorptionDocument(
        structures=sorted_structures,
        configuration_numbers=sorted_configuration_numbers,
        adsorption_energies=sorted_adsorption_energies,
        job_dirs=sorted_job_dirs,
    )
