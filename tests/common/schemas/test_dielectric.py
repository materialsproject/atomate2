import numpy as np
import pytest
from pymatgen.core import Structure

from atomate2.common.schemas.dielectric import DielectricDocument


@pytest.fixture
def gan_structure(test_dir):
    # P6_3mc is non-centrosymmetric, so GaN is piezoelectric
    return Structure.from_file(test_dir / "structures" / "GaN.cif")


def test_dielectric_document(gan_structure):
    piezo = np.zeros((3, 6))
    piezo[2, :3] = [-0.3, -0.3, 0.6]
    born = np.zeros((len(gan_structure), 3, 3)).tolist()

    doc = DielectricDocument.from_tensors(
        gan_structure,
        epsilon_electronic=5.0 * np.eye(3),
        born=born,
        epsilon_ionic=4.0 * np.eye(3),
        piezo_electronic=piezo.tolist(),
        piezo_ionic=piezo.tolist(),
    )
    np.testing.assert_allclose(doc.epsilon_total, 9.0 * np.eye(3))
    assert doc.refractive_index == pytest.approx(np.sqrt(5.0))
    assert doc.piezoelectric.e_ij_max > 0

    # electronic tensor and Born charges only, as from MACE-Field
    doc = DielectricDocument.from_tensors(
        gan_structure, epsilon_electronic=5.0 * np.eye(3), born=born
    )
    assert doc.epsilon_ionic is None
    assert doc.epsilon_total is None
    assert doc.piezoelectric is None
