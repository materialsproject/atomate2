import numpy as np
import pytest
from jobflow import run_locally
from pymatgen.core import Lattice, Structure

from atomate2.common.schemas.dielectric import DielectricDocument
from atomate2.forcefields.flows.dielectric import RelaxDielectricMaker


def test_relax_dielectric(fake_dielectric_calculator, memory_jobstore, clean_dir):
    structure = Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.6), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    ).get_primitive_structure()

    flow = RelaxDielectricMaker(relax_maker=None).make(structure)
    run_locally(flow, store=memory_jobstore, create_folders=True, ensure_success=True)

    # the fake calculator gives Born charges of +2 and -2 and a susceptibility of 3
    output = flow.output.resolve(memory_jobstore)
    assert isinstance(output, DielectricDocument)
    np.testing.assert_allclose(output.epsilon_electronic, 4 * np.eye(3))
    assert output.refractive_index == pytest.approx(2)
    np.testing.assert_allclose(
        output.born, np.array([2, -2])[:, None, None] * np.eye(3)
    )
    assert output.epsilon_ionic is None
    assert output.epsilon_total is None
    assert output.piezoelectric is None
