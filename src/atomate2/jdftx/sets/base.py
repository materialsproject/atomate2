"""Module defining base JDFTx input set and generator."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from importlib.resources import files as get_mod_path
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from monty.serialization import loadfn
from pymatgen.core.units import ang_to_bohr, eV_to_Ha
from pymatgen.io.core import InputGenerator
from pymatgen.io.jdftx.inputs import JDFTXInfile
from pymatgen.io.jdftx.sets import JdftxInputSet
from pymatgen.io.vasp import Kpoints

from atomate2 import SETTINGS

if TYPE_CHECKING:
    from pymatgen.core import Structure

_BASE_JDFTX_SET = loadfn(get_mod_path("pymatgen.io.jdftx") / "BaseJdftxSet.yaml")
_GENERATION_CONFIG = loadfn(
    get_mod_path("atomate2.jdftx.sets") / "GenerationConfig.yaml"
)
_PSEUDO_CONFIG = loadfn(get_mod_path("atomate2.jdftx.sets") / "PseudosConfig.yaml")


@dataclass
class JdftxInputGenerator(InputGenerator):
    """A class to generate JDFTx input sets.

    Args:
        user_settings (dict): User JDFTx settings. This allows the user to
            override the default JDFTx settings loaded in the default_settings
            argument.
        coulomb_truncation (bool) = False:
            Whether to use coulomb truncation and calculate the coulomb
            truncation center. Only works for molecules and slabs.
        auto_kpoint_density (int) = 1000:
            Reciprocal k-point density for automatic k-point calculation. If
            k-points are specified in user_settings, they will not be
            overridden.
        potential (None, float) = None:
            Potential vs SHE for GC-DFT calculation.
        calc_type (str) = "bulk":
            Type of calculation used for setting input parameters. Options are:
            ["bulk", "surface", "molecule"].
        pseudopotentials (str) = "GBRV"
        config_dict (dict): The config dictionary used to set input parameters
            used in the calculation of JDFTx tags.
        default_settings: Default JDFTx settings.
    """

    # copy _BASE_JDFTX_SET to ensure each class instance has its own copy
    # otherwise in-place changes can affect other instances
    user_settings: dict = field(default_factory=dict)
    coulomb_truncation: bool = False
    auto_kpoint_density: int = 1000
    potential: None | float = None
    calc_type: str = "bulk"
    pseudopotentials: str = "GBRV"
    config_dict: dict = field(default_factory=lambda: _GENERATION_CONFIG)
    default_settings: dict = field(default_factory=lambda: _BASE_JDFTX_SET)

    def __post_init__(self) -> None:
        """Post init formatting of arguments."""
        calc_type_options = ["bulk", "surface", "molecule"]
        if self.calc_type not in calc_type_options:
            raise ValueError(
                f"calc type f{self.calc_type} not in list of supported calc "
                "types: {calc_type_options}."
            )
        self.settings = self.default_settings.copy()
        self.settings.update(self.user_settings)
        # set default coords-type to Cartesian
        if "coords-type" not in self.settings:
            self.settings["coords-type"] = "Cartesian"
        self._apply_settings(self.settings)

    def _apply_settings(
        self, settings: dict[str, Any]
    ) -> None:  # settings as attributes
        for key, value in settings.items():
            setattr(self, key, value)

    def get_input_set(
        self,
        structure: Structure = None,
    ) -> JdftxInputSet:
        """Get a JDFTx input set for a structure.

        Parameters
        ----------
        structure
            A Pymatgen Structure.

        Returns
        -------
        JdftxInputSet
            A JDFTx input set.
        """
        self.settings.update(self.user_settings)
        self.set_kgrid(structure=structure)
        self.set_coulomb_interaction(structure=structure)
        self.set_nbands(structure=structure)
        self.set_mu()
        self.set_pseudos()
        self.set_magnetic_moments(structure=structure)
        self._apply_settings(self.settings)

        jdftxinput = JDFTXInfile.from_dict(self.settings)

        return JdftxInputSet(jdftxinput=jdftxinput, structure=structure)

    def set_kgrid(self, structure: Structure) -> None:
        """Get k-point grid.

        Parameters
        ----------
        structure
            A pymatgen structure.

        Returns
        -------
        Kpoints
            A tuple of integers specifying the k-point grid.
        """
        # never override k grid definition in user settings
        if "kpoint-folding" in self.user_settings:
            return
        # calculate k-grid with k-point density
        kpoints = Kpoints.automatic_density(
            structure=structure, kppa=self.auto_kpoint_density
        )
        kpoints = kpoints.kpts[0]
        if self.calc_type == "surface":
            kpoints = (kpoints[0], kpoints[1], 1)
        elif self.calc_type == "molecule":
            kpoints = (1, 1, 1)
        kpoint_update = {
            "kpoint-folding": {
                "n0": kpoints[0],
                "n1": kpoints[1],
                "n2": kpoints[2],
            }
        }
        self.settings.update(kpoint_update)
        return

    def set_coulomb_interaction(
        self,
        structure: Structure,
    ) -> JDFTXInfile:
        """
        Set coulomb-interaction and coulomb-truncation for JDFTXInfile.

        Description

        Parameters
        ----------
        structure
            A pymatgen structure

        Returns
        -------
        jdftxinputs
            A pymatgen.io.jdftx.inputs.JDFTXInfile object

        """
        if "coulomb-interaction" in self.settings:
            return
        if self.calc_type == "bulk":
            self.settings["coulomb-interaction"] = {
                "truncationType": "Periodic",
            }
            return
        if self.calc_type == "surface":
            self.settings["coulomb-interaction"] = {
                "truncationType": "Slab",
                "dir": "001",
            }
        elif self.calc_type == "molecule":
            self.settings["coulomb-interaction"] = {
                "truncationType": "Isolated",
            }
        com = center_of_mass(structure=structure)
        if self.settings["coords-type"] == "Cartesian":
            com = com @ structure.lattice.matrix * ang_to_bohr
        elif self.settings["coords-type"] == "Lattice":
            com = com * ang_to_bohr
        self.settings["coulomb-truncation-embed"] = {
            "c0": com[0],
            "c1": com[1],
            "c2": com[2],
        }
        return

    def set_nbands(self, structure: Structure) -> None:
        """Set number of bands in DFT calculation."""
        nelec = 0
        for atom in structure.species:
            nelec += _PSEUDO_CONFIG[self.pseudopotentials][str(atom)]
        nbands_add = int(nelec / 2) + 10
        nbands_mult = int(nelec / 2) * self.config_dict["bands_multiplier"]
        self.settings["elec-n-bands"] = max(nbands_add, nbands_mult)

    def set_pseudos(
        self,
    ) -> None:
        """Set ion-species tag corresponding to pseudopotentials."""
        if SETTINGS.JDFTX_PSEUDOS_DIR is not None:
            pseudos_str = str(
                Path(SETTINGS.JDFTX_PSEUDOS_DIR) / Path(self.pseudopotentials)
            )
        else:
            pseudos_str = self.pseudopotentials

        add_tags = [
            pseudos_str + "/$ID" + suffix
            for suffix in _PSEUDO_CONFIG[self.pseudopotentials]["suffixes"]
        ]
        # do not override pseudopotentials in settings
        if "ion-species" in self.settings:
            return
        self.settings["ion-species"] = add_tags
        return

    def set_mu(self) -> None:
        """Set absolute electron chemical potential (fermi level) for GC-DFT."""
        # never override mu in settings
        if "target-mu" in self.settings or self.potential is None:
            return
        solvent_model = self.settings["pcm-variant"]
        ashep = self.config_dict["ASHEP"][solvent_model]
        # calculate absolute potential in Hartree
        mu = -(ashep - self.potential) / eV_to_Ha
        self.settings["target-mu"] = {"mu": mu}
        return

    def set_magnetic_moments(self, structure: Structure) -> None:
        """Set the magnetic moments for each atom in the structure.

        If the user specified magnetic moments as JDFTx tags, they will
        not be prioritized. The user can also set the magnetic moments in
        the site_params dictionary attribute of the structure. If neither above
        options are set, the code will initialize all metal atoms with +5
        magnetic moments.

        Parameters
        ----------
        structure
            A pymatgen structure

        Returns
        -------
        None
        """
        # check if user set JFDTx magnetic tags and return if true
        if (
            "initial-magnetic-moments" in self.settings
            or "elec-initial-magnetization" in self.settings
        ):
            return
        # if magmoms set on structure, build JDFTx tag
        if "magmom" in structure.site_properties:
            if len(structure.species) != len(structure.site_properties["magmom"]):
                raise ValueError(
                    f"length of magmom, {structure.site_properties['magmom']} "
                    "does not match number of species in structure, "
                    f"{len(structure.species)}."
                )
            magmoms = defaultdict(list)
            for magmom, species in zip(
                structure.site_properties["magmom"], structure.species, strict=False
            ):
                magmoms[species].append(magmom)
            tag_str = ""
            for element, magmom_list in magmoms.items():
                tag_str += f"{element} " + " ".join(list(map(str, magmom_list))) + " "
        # set magmoms to +5 for all metals in structure.
        else:
            magmoms = defaultdict(list)
            for species in structure.species:
                if species.is_metal:
                    magmoms[str(species)].append(5)
                else:
                    magmoms[str(species)].append(0)
            tag_str = ""
            for element, magmom_list in magmoms.items():
                tag_str += f"{element} " + " ".join(list(map(str, magmom_list))) + " "
        self.settings["initial-magnetic-moments"] = tag_str
        return


def center_of_mass(structure: Structure) -> np.ndarray:
    """
    Calculate center of mass.

    Parameters
    ----------
    structure: Structure
        A pymatgen structure

    Returns
    -------
    np.ndarray
        A numpy array containing the center of mass in fractional coordinates.
    """
    weights = [site.species.weight for site in structure]
    return np.average(structure.frac_coords, weights=weights, axis=0)
