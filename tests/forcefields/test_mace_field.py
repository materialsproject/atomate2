"""Test the Born charges and the dielectric tensor of the MACE-Field model.

The MACE-Field fork of mace installs as mace-torch, so these tests only run in
the mace-field CI group.
"""

import hashlib
import urllib.request

import numpy as np
import pytest
from emmet.core.utils import get_hash_blocked
from jobflow import run_locally
from pymatgen.core import Lattice, Structure

from atomate2.forcefields.jobs import ForceFieldDielectricMaker

from .conftest import mlff_is_installed

pytestmark = pytest.mark.skipif(
    not mlff_is_installed("MACE_Field"), reason="MACE-Field is not installed"
)

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


def test_mace_field_dielectric_maker(mace_field_model, clean_dir):
    structure = Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    ).get_primitive_structure()
    job = ForceFieldDielectricMaker(
        calculator_kwargs={"model": str(mace_field_model)}
    ).make(structure)
    output = run_locally(job, ensure_success=True)[job.uuid][1].output

    born = np.array(output.born)
    # the acoustic sum rule holds by construction
    assert born.sum(axis=0) == pytest.approx(np.zeros((3, 3)), abs=1e-6)
    # cubic, so the tensors are isotropic. The Materials Project's Harmonic Phonon
    # Database has 1.09 and 2.56 from DFPT.
    species = [str(site.specie) for site in output.structure]
    assert born[species.index("Na")] == pytest.approx(1.0898 * np.eye(3), abs=1e-3)
    epsilon = np.array(output.epsilon_static)
    assert epsilon == pytest.approx(2.5845 * np.eye(3), abs=1e-3)
