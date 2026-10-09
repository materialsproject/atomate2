import json
from pathlib import Path

import numpy as np
import pytest
from ase.calculators.emt import EMT
from monty.json import MontyEncoder
from phonopy import Phonopy
from pydantic import ValidationError
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure
from scipy.constants import R

from atomate2.common.schemas.phonons import (
    PhononBSDOSDoc,
    PhononComputationalSettings,
    PhononJobDirs,
    PhononUUIDs,
    ThermalDisplacementData,
    _get_thermal_displacement_data,
    _set_nac_params,
)


def test_thermal_displacement_data():
    doc = ThermalDisplacementData(freq_min_thermal_displacements=0.0)
    validated = ThermalDisplacementData.model_validate_json(
        json.dumps(doc, cls=MontyEncoder)
    )
    assert isinstance(validated, ThermalDisplacementData)


def test_phonon_bs_dos_doc():
    kwargs = {
        "total_dft_energy": None,
        "supercell_matrix": np.eye(3),
        "primitive_matrix": np.eye(3),
        "code": "test",
        "phonopy_settings": PhononComputationalSettings(
            npoints_band=1, kpath_scheme="test", kpoint_density_dos=1
        ),
        "thermal_displacement_data": None,
        "jobdirs": None,
        "uuids": None,
    }
    doc = PhononBSDOSDoc(**kwargs)
    # check validation raises no errors
    validated = PhononBSDOSDoc.model_validate_json(json.dumps(doc, cls=MontyEncoder))
    assert isinstance(validated, PhononBSDOSDoc)

    # test invalid supercell_matrix type fails
    with pytest.raises(ValidationError):
        doc = PhononBSDOSDoc(**kwargs | {"supercell_matrix": (1, 1, 1)})

    # test optional material_id
    doc = PhononBSDOSDoc(**kwargs | {"material_id": 1234})
    assert doc.material_id == 1234

    # test extra="allow" option
    doc = PhononBSDOSDoc(**kwargs | {"extra_field": "test"})
    assert doc.extra_field == "test"


def test_from_forces_born_thermal_properties(clean_dir):
    """Cu3Au with EMT reaches the classical limit of 4 x 3R per formula unit."""
    structure = Structure(
        Lattice.cubic(3.74),
        ["Au", "Cu", "Cu", "Cu"],
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    )
    supercell_matrix = 3 * np.eye(3)
    phonon = Phonopy(get_phonopy_structure(structure), supercell_matrix)
    phonon.generate_displacements(distance=0.01)
    forces = []
    for cell in phonon.supercells_with_displacements:
        atoms = AseAtomsAdaptor.get_atoms(get_pmg_structure(cell))
        atoms.calc = EMT()
        forces.append(atoms.get_forces().tolist())
    doc = PhononBSDOSDoc.from_forces_born(
        structure=structure,
        supercell_matrix=supercell_matrix,
        displacement=0.01,
        sym_reduce=True,
        symprec=1e-5,
        use_symmetrized_structure=None,
        kpath_scheme="seekpath",
        code="forcefields",
        displacement_data={"forces": forces, "dirs": [], "uuids": []},
        total_dft_energy=None,
        store_force_constants=False,
        tmax=3000,
        static_run_job_dir=None,
        static_run_uuid=None,
        born_run_job_dir=None,
        born_run_uuid=None,
        optimization_run_job_dir=None,
        optimization_run_uuid=None,
    )
    assert doc.entropies[0] == 0
    assert doc.heat_capacities[0] == 0
    assert doc.heat_capacities[-1] == pytest.approx(12 * R, rel=1e-3)
    assert doc.internal_energies[-1] == pytest.approx(12 * R * 3000, rel=1e-3)


def test_get_thermal_displacement_data(clean_dir):
    """Cu with EMT on a 7x7x7 mesh, which contains Gamma.

    With freq_min=0 and the acoustic modes at Gamma included, U is about 8e6 A^2.
    """
    structure = Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(3.61), ["Cu"], [[0, 0, 0]]
    ).get_primitive_structure()
    phonon = Phonopy(get_phonopy_structure(structure), 6 * np.eye(3), "P")
    phonon.generate_displacements(distance=0.01)
    forces = []
    for cell in phonon.supercells_with_displacements:
        atoms = AseAtomsAdaptor.get_atoms(get_pmg_structure(cell))
        atoms.calc = EMT()
        forces.append(atoms.get_forces())
    phonon.forces = forces
    phonon.produce_force_constants()

    with pytest.warns(UserWarning, match="leave out the acoustic modes at Gamma"):
        data = _get_thermal_displacement_data(
            phonon, kpoint_density_thermal_displacements=343
        )

    assert data["temperatures_thermal_displacements"] == [0, 100, 200, 300, 400, 500]
    u_300 = np.array(data["thermal_displacement_matrix"])[3, 0]
    assert np.trace(u_300) / 3 == pytest.approx(0.005751, rel=1e-3)
    assert data["freq_min_thermal_displacements"] == 0.0
    assert all(Path(f"tdispmat_{t}K.cif").is_file() for t in range(0, 501, 100))


# schemas where all fields have default values
@pytest.mark.parametrize("model_cls", [PhononJobDirs, PhononUUIDs])
def test_model_validate(model_cls):
    validated = model_cls.model_validate_json(json.dumps(model_cls(), cls=MontyEncoder))
    assert isinstance(validated, model_cls)


def get_nacl_phonon():
    structure = Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.6), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    ).get_primitive_structure()
    return Phonopy(get_phonopy_structure(structure), supercell_matrix=np.eye(3))


@pytest.mark.parametrize(
    ("code", "charge", "has_nac"),
    [
        ("vasp", 1, True),
        ("forcefields", 1, True),
        ("ase", 1, True),
        ("torchsim", 1, True),
        ("aims", 1, False),
        ("vasp", 0, False),
    ],
)
def test_set_nac_params(code, charge, has_nac):
    phonon = get_nacl_phonon()
    born = charge * np.array([1, -1])[:, None, None] * np.eye(3)
    borns, epsilon = _set_nac_params(
        phonon, born.tolist(), (2 * np.eye(3)).tolist(), 1e-4, code
    )

    assert borns == pytest.approx(born)
    assert epsilon == pytest.approx(2 * np.eye(3))
    assert (phonon.nac_params is not None) == has_nac


def test_set_nac_params_without_born():
    phonon = get_nacl_phonon()
    assert _set_nac_params(phonon, None, None, 1e-4, "vasp") == (None, None)
    assert phonon.nac_params is None
    with pytest.raises(ValueError, match="Number of Born charges"):
        _set_nac_params(phonon, [np.eye(3).tolist()], np.eye(3).tolist(), 1e-4, "vasp")
