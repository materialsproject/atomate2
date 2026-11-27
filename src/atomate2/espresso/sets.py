"""Input set generators for Quantum ESPRESSO."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from ase.io.espresso import write_espresso_in, write_espresso_ph
from monty.serialization import loadfn
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.core import InputGenerator, InputSet

from atomate2 import SETTINGS

if TYPE_CHECKING:
    from pymatgen.core import Structure


@cache
def _load_pseudo_family(pseudo_dir: str | None, pseudo_family: str) -> dict:
    """Load the JSON file of a pseudopotential family."""
    if pseudo_dir is None:
        raise ValueError("Set ESPRESSO_PSEUDO_DIR in the atomate2 settings.")
    return loadfn(Path(pseudo_dir) / f"{pseudo_family}.json")


@dataclass
class EspressoInputGenerator(InputGenerator):
    """
    Generate the pw.x input of a Quantum ESPRESSO calculation.

    The pseudopotentials and the plane-wave cutoffs come from a JSON file in
    ``SETTINGS.ESPRESSO_PSEUDO_DIR`` named after the pseudopotential family. It has
    the format of the SSSP libraries: for each element, the name of the
    pseudopotential file ("filename") and the recommended cutoffs in Ry
    ("cutoff_wfc" and "cutoff_rho"). The largest cutoffs of the elements in the
    structure are used.

    A relaxation ("relax" or "vc-relax") stops when the energy change is below
    1e-6 Ry, the forces are below 1e-4 Ry/Bohr, and, for "vc-relax", the pressure
    is below 0.1 kbar. Sites with a nonzero "magmom" property start a
    spin-polarized calculation. pw.x does not write the wavefunctions
    (disk_io="nowf"), since only the dielectric job needs them.

    Parameters
    ----------
    calculation : str
        The pw.x calculation type, e.g. "scf" or "vc-relax".
    pseudo_family : str
        The name of the pseudopotential family.
    kspacing : float
        The largest spacing between k-points in 1/Å, including the factor of
        2π, as for KSPACING in VASP. The k-point grid is centered on Γ.
    conv_thr_per_atom : float
        The convergence threshold of the self-consistency per atom in Ry.
    user_input_data : dict
        pw.x settings that update the defaults, grouped by namelist, e.g.
        ``{"system": {"degauss": 0.02}}``.
    """

    calculation: str = "scf"
    pseudo_family: str = "SSSP_1.3.0_PBEsol_efficiency"
    kspacing: float = 0.15
    conv_thr_per_atom: float = 1e-10
    user_input_data: dict = field(default_factory=dict)

    def get_input_data(self, structure: Structure) -> dict:
        """
        Get the pw.x namelists of a structure.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure.

        Returns
        -------
        dict
            The pw.x settings grouped by namelist.
        """
        family = _load_pseudo_family(SETTINGS.ESPRESSO_PSEUDO_DIR, self.pseudo_family)
        elements = [element.symbol for element in structure.composition]
        input_data = {
            "control": {
                "calculation": self.calculation,
                "pseudo_dir": SETTINGS.ESPRESSO_PSEUDO_DIR,
                "outdir": "./",
                "disk_io": "nowf",
                "tprnfor": True,
                "tstress": True,
            },
            "system": {
                "ecutwfc": max(family[element]["cutoff_wfc"] for element in elements),
                "ecutrho": max(family[element]["cutoff_rho"] for element in elements),
                "occupations": "smearing",
                "smearing": "cold",
                "degauss": 0.01,
            },
            "electrons": {"conv_thr": self.conv_thr_per_atom * len(structure)},
        }
        if self.calculation in {"relax", "vc-relax"}:
            input_data["control"] |= {"forc_conv_thr": 1e-4, "etot_conv_thr": 1e-6}
            input_data["cell"] = {"press_conv_thr": 0.1}
        for namelist, values in self.user_input_data.items():
            input_data.setdefault(namelist, {}).update(values)
        return input_data

    def get_input_set(self, structure: Structure) -> InputSet:
        """
        Get the input set of a structure.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure.

        Returns
        -------
        InputSet
            An input set with the pw.x input file "pw.in".
        """
        family = _load_pseudo_family(SETTINGS.ESPRESSO_PSEUDO_DIR, self.pseudo_family)
        kpts = [
            max(1, math.ceil(length / self.kspacing))
            for length in structure.lattice.reciprocal_lattice.abc
        ]
        pw_input = io.StringIO()
        write_espresso_in(
            pw_input,
            AseAtomsAdaptor.get_atoms(structure),
            input_data=self.get_input_data(structure),
            pseudopotentials={
                element.symbol: family[element.symbol]["filename"]
                for element in structure.composition
            },
            kpts=kpts,
        )
        return InputSet(inputs={"pw.in": pw_input.getvalue()})


@dataclass
class EspressoDielectricInputGenerator(EspressoInputGenerator):
    """
    Generate the inputs of a dielectric calculation with pw.x and ph.x.

    ph.x computes the dielectric tensor and the Born effective charges at Γ.
    This works only for insulators, so pw.x always runs with fixed occupations.
    The "magmom" site property is removed, so pw.x is not spin polarized unless
    ``user_input_data`` asks for it. pw.x writes the wavefunctions for ph.x.

    Parameters
    ----------
    tr2_ph : float
        The convergence threshold of ph.x.
    """

    tr2_ph: float = 1e-16

    def get_input_data(self, structure: Structure) -> dict:
        """
        Get the pw.x namelists of a structure, with fixed occupations.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure.

        Returns
        -------
        dict
            The pw.x settings grouped by namelist.
        """
        input_data = super().get_input_data(structure)
        input_data["control"]["disk_io"] = "low"
        input_data["system"]["occupations"] = "fixed"
        input_data["system"].pop("smearing", None)
        input_data["system"].pop("degauss", None)
        return input_data

    def get_input_set(self, structure: Structure) -> InputSet:
        """
        Get the input set of a structure.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure.

        Returns
        -------
        InputSet
            An input set with the pw.x input file "pw.in" and the ph.x input
            file "ph.in".
        """
        if "magmom" in structure.site_properties:
            structure = structure.copy()
            structure.remove_site_property("magmom")
        input_set = super().get_input_set(structure)
        ph_input = io.StringIO()
        write_espresso_ph(
            ph_input,
            input_data={
                "inputph": {
                    "outdir": "./",
                    "tr2_ph": self.tr2_ph,
                    "epsil": True,
                    "trans": False,
                }
            },
            qpts=(0, 0, 0),
        )
        input_set["ph.in"] = ph_input.getvalue()
        return input_set
