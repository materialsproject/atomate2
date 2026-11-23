"""Jobs for inner-sphere reorganization energies."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from emmet.core.molecules.thermo import get_free_energy
from emmet.core.qc_tasks import TaskDoc
from jobflow import job
from pymatgen.core import Molecule

from atomate2.qchem.schemas.reorganization import InnerReorganizationDocument

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)

STATE_KEYS = ("1@1", "1@2", "2@2", "2@1")


def _final_molecule(source: Molecule | TaskDoc | dict) -> Molecule:
    """Optimized geometry of a task document (or the molecule itself)."""
    if isinstance(source, Molecule):
        return source
    output = source.output if isinstance(source, TaskDoc) else source["output"]
    get = output.get if isinstance(output, dict) else lambda key: getattr(output, key)
    # an optimization reports optimized_molecule; a frequency (or FFopt) task is run
    # at the optimized geometry, which is its initial_molecule
    molecule = get("optimized_molecule") or get("initial_molecule")
    return molecule if isinstance(molecule, Molecule) else Molecule.from_dict(molecule)


def _in_state(molecule: Molecule, charge: int, spin_multiplicity: int) -> Molecule:
    """Copy of a geometry in another charge and spin state."""
    return Molecule(
        species=molecule.species,
        coords=molecule.cart_coords,
        charge=charge,
        spin_multiplicity=spin_multiplicity,
    )


@job
def prepare_vertical_molecules(
    source_1: Molecule | TaskDoc,
    source_2: Molecule | TaskDoc,
    charge_1: int,
    spin_multiplicity_1: int,
    charge_2: int,
    spin_multiplicity_2: int,
) -> dict[str, Molecule]:
    """
    Build the four state/geometry combinations of the four-point scheme.

    Parameters
    ----------
    source_1, source_2 : .Molecule or .TaskDoc
        Optimized geometries of states 1 and 2, or the task documents of their
        optimizations.
    charge_1, spin_multiplicity_1, charge_2, spin_multiplicity_2 : int
        Charges and spin multiplicities of the two states.

    Returns
    -------
    dict of str to .Molecule
        Molecules keyed "i@j" (state i at geometry j), plus "1" and "2" for the two
        optimized geometries in their own states.
    """
    geometry_1 = _final_molecule(source_1)
    geometry_2 = _final_molecule(source_2)
    states = {1: (charge_1, spin_multiplicity_1), 2: (charge_2, spin_multiplicity_2)}
    geometries = {1: geometry_1, 2: geometry_2}
    molecules = {
        f"{state}@{geometry}": _in_state(geometries[geometry], *states[state])
        for state in (1, 2)
        for geometry in (1, 2)
    }
    molecules["1"], molecules["2"] = molecules["1@1"], molecules["2@2"]
    return molecules


def _free_energy(source: Any) -> float | None:
    """Free energy (eV) of a frequency task document, if it has thermochemistry."""
    if source is None or isinstance(source, Molecule):
        return None
    output = source.output if isinstance(source, TaskDoc) else source["output"]
    get = output.get if isinstance(output, dict) else lambda key: getattr(output, key)
    energy, enthalpy, entropy = (
        get("final_energy"),
        get("enthalpy"),
        get("entropy"),
    )
    if None in (energy, enthalpy, entropy):
        return None
    return get_free_energy(energy, enthalpy, entropy)


def _mulliken_spins(task_doc: TaskDoc) -> np.ndarray | None:
    """Per-atom Mulliken spins of a task document (zeros if closed shell)."""
    mulliken = task_doc.output.mulliken
    if mulliken is None or len(mulliken) == 0:
        return None
    populations = np.asarray(mulliken[-1], dtype=float)
    if populations.ndim == 1:  # restricted: charges only
        return np.zeros(len(populations))
    return populations[:, 1]


def _max_spin_change(single_points: dict[str, TaskDoc]) -> dict[str, float] | None:
    """Largest per-atom spin change of each state between the two geometries."""
    changes = {}
    for state in ("1", "2"):
        spins = [_mulliken_spins(single_points[f"{state}@{g}"]) for g in ("1", "2")]
        if any(s is None for s in spins):
            return None
        changes[state] = float(np.abs(spins[0] - spins[1]).max())
    return changes


@job(output_schema=InnerReorganizationDocument)
def analyze_inner_reorganization(
    molecules: dict[str, Molecule],
    single_points: dict[str, TaskDoc],
    single_point_uuids: dict[str, str] | None = None,
    opt_1: TaskDoc | None = None,
    opt_2: TaskDoc | None = None,
    unconstrained_points: dict[str, TaskDoc] | None = None,
    cdft_settings: dict | None = None,
) -> InnerReorganizationDocument:
    """
    Compute the inner-sphere reorganization energies from the four single points.

    Parameters
    ----------
    molecules : dict of str to .Molecule
        Output of :obj:`prepare_vertical_molecules`.
    single_points : dict of str to .TaskDoc
        Single-point task documents keyed "i@j" (state i at geometry j).
    single_point_uuids : dict of str to str or None
        UUIDs of the single-point jobs.
    opt_1, opt_2 : .TaskDoc or None
        Task documents of the optimizations; used for the adiabatic free energy if
        they contain thermochemistry (e.g. frequency flattening optimizations).
    unconstrained_points : dict of str to .TaskDoc or None
        Unconstrained single points, when ``single_points`` are constrained DFT
        (diabatic) ones.
    cdft_settings : dict or None
        Fragment and constraints of the diabatic single points.

    Returns
    -------
    InnerReorganizationDocument
        The reorganization energies.
    """
    energies = {key: single_points[key].output.final_energy for key in STATE_KEYS}
    calc = single_points["1@1"].calcs_reversed[0]
    unconstrained_energies = (
        {key: unconstrained_points[key].output.final_energy for key in STATE_KEYS}
        if unconstrained_points
        else None
    )
    # the electronic-state check uses the unconstrained (ground-state) points
    reference_points = unconstrained_points or (
        None if cdft_settings else single_points
    )
    return InnerReorganizationDocument.from_energies(
        molecule_1=molecules["1"],
        molecule_2=molecules["2"],
        energies=energies,
        free_energy_1=_free_energy(opt_1),
        free_energy_2=_free_energy(opt_2),
        unconstrained_energies=unconstrained_energies,
        max_spin_change=_max_spin_change(reference_points)
        if reference_points
        else None,
        diabatic=cdft_settings is not None,
        cdft_settings=cdft_settings,
        solvation_lot_info=getattr(calc, "solvation_lot_info", None),
        single_point_uuids=single_point_uuids,
        single_point_dirs={key: str(single_points[key].dir_name) for key in STATE_KEYS},
    )
