import numpy as np
import pytest
from jobflow import run_locally
from pymatgen.core import Lattice, Structure

pytest.importorskip("pheasy")

from atomate2.espresso.flows.pheasy import PhononMaker


def test_pheasy_phonon_maker(mock_espresso, clean_dir):
    mock_espresso(
        {
            "espresso relax": "pheasy_flow/relax",
            "espresso static": "pheasy_flow/static",
            "espresso dielectric": "pheasy_flow/dielectric",
            "espresso static 1/3": "pheasy_flow/displacement_1",
            "espresso static 2/3": "pheasy_flow/displacement_2",
            "espresso static 3/3": "pheasy_flow/displacement_3",
        }
    )
    sic = Structure(
        Lattice.from_parameters(3.095, 3.095, 3.095, 60, 60, 60),
        ["Si", "C"],
        [[0, 0, 0], [0.25, 0.25, 0.25]],
    )
    maker = PhononMaker(min_length=5)
    for job_maker in (
        maker.bulk_relax_maker,
        maker.static_energy_maker,
        maker.born_maker,
        maker.phonon_displacement_maker,
    ):
        job_maker.input_set_generator.pseudo_family = "test_family"
        job_maker.input_set_generator.kspacing = 0.3
    # a coarser grid for the 16-atom supercells keeps the reference runs short
    maker.phonon_displacement_maker.input_set_generator.kspacing = 0.6

    flow = maker.make(sic)
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    doc = responses[flow.output.uuid][1].output

    assert doc.code == "espresso"
    frequencies = np.array(doc.phonon_bandstructure.frequencies)
    assert frequencies.min() > -1e-5
    assert not doc.has_imaginary_modes
    assert doc.epsilon_static[0][0] == pytest.approx(7.008, abs=0.001)
    assert doc.born[0][0][0] == pytest.approx(2.69, abs=0.01)
    assert doc.born[1][0][0] == pytest.approx(-2.69, abs=0.01)
    # with the non-analytical term correction, the LO mode at Γ is above the TO
    # mode
    assert frequencies.max() == pytest.approx(27.90, abs=0.05)
