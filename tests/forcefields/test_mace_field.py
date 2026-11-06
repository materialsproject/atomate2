"""Test the Born charges and the dielectric tensor of the MACE-Field model."""

import hashlib
import urllib.request
from importlib.metadata import version as get_imported_version

import numpy as np
import pytest
from emmet.core.utils import get_hash_blocked
from jobflow import run_locally
from pymatgen.core import Lattice, Structure

from atomate2.forcefields.flows.phonons import PhononMaker
from atomate2.forcefields.jobs import ForceFieldDielectricMaker

from .conftest import mlff_is_installed

MODEL_URL = (
    "https://github.com/mdi-group/mace-field/releases/download/1.0.2/"
    "MACEField-MH-0-omat-dielectric.model"
)
MODEL_MD5 = "1dac204204e368d94b2b8a597c9fdf8d"


@pytest.fixture(scope="module")
def mace_field_model(tmp_path_factory):
    path = tmp_path_factory.mktemp("mace_field") / MODEL_URL.rsplit("/", 1)[-1]
    urllib.request.urlretrieve(MODEL_URL, path)
    assert get_hash_blocked(str(path), hasher=hashlib.md5()) == MODEL_MD5
    return path


@pytest.fixture
def nacl_structure():
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    ).get_primitive_structure()


@pytest.mark.skipif(
    not mlff_is_installed("MACE_FIELD"), reason="MACE-Field is not installed"
)
def test_mace_field_dielectric_maker(mace_field_model, nacl_structure):
    job = ForceFieldDielectricMaker(
        calculator_kwargs={"model": str(mace_field_model)}
    ).make(nacl_structure)
    output = run_locally(job, ensure_success=True)[job.uuid][1].output

    born = np.array(output.born)
    # the acoustic sum rule holds by construction
    assert born.sum(axis=0) == pytest.approx(np.zeros((3, 3)), abs=1e-6)
    # cubic, so the tensors are isotropic
    species = [str(site.specie) for site in output.structure]
    assert born[species.index("Na")] == pytest.approx(1.0898 * np.eye(3), abs=1e-3)
    epsilon = np.array(output.epsilon_static)
    assert epsilon == pytest.approx(2.5845 * np.eye(3), abs=1e-3)
    assert output.forcefield_version == get_imported_version("mace-torch")


@pytest.mark.skipif(
    not mlff_is_installed("MACE_FIELD"), reason="MACE-Field is not installed"
)
def test_mace_field_phonon_nac(mace_field_model, nacl_structure):
    maker = PhononMaker.from_force_field_name(
        "MACE_MP_0",
        calculator_kwargs={"model": "medium-omat-0"},
        relax_initial_structure=False,
        born_maker=ForceFieldDielectricMaker(
            calculator_kwargs={"model": str(mace_field_model)}
        ),
        min_length=10,
        create_thermal_displacements=False,
        store_force_constants=False,
    )
    flow = maker.make(nacl_structure)
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    doc = responses[flow[-1].uuid][1].output

    # 6.486 THz without the MACE-Field Born charges
    assert np.max(doc.phonon_bandstructure.bands) == pytest.approx(7.505, abs=1e-2)
