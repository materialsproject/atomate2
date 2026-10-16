import numpy as np
from emmet.core.polar import PiezoelectricDoc
from jobflow import run_locally
from pymatgen.core import Structure

from atomate2.common.jobs.dielectric import dielectric_analysis
from atomate2.common.schemas.dielectric import DielectricDocument


def test_dielectric_analysis(test_dir, memory_jobstore):
    # P6_3mc is non-centrosymmetric, so GaN has a piezoelectric document
    structure = Structure.from_file(test_dir / "structures" / "GaN.cif")
    piezo = np.zeros((3, 6))
    piezo[2, :3] = [-0.3, -0.3, 0.6]

    job = dielectric_analysis(
        structure,
        epsilon_electronic=(5.0 * np.eye(3)).tolist(),
        born=np.zeros((len(structure), 3, 3)).tolist(),
        epsilon_ionic=(4.0 * np.eye(3)).tolist(),
        piezo_electronic=piezo.tolist(),
        piezo_ionic=piezo.tolist(),
    )
    run_locally(job, store=memory_jobstore, ensure_success=True)

    # the emmet piezoelectric document must survive the round trip through the store
    doc = job.output.resolve(memory_jobstore)
    assert isinstance(doc, DielectricDocument)
    assert isinstance(doc.piezoelectric, PiezoelectricDoc)
    assert doc.piezoelectric.e_ij_max > 0
    np.testing.assert_allclose(doc.epsilon_total, 9.0 * np.eye(3))
