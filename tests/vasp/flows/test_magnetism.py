from __future__ import annotations

import pytest
from jobflow import run_locally
from pymatgen.analysis.magnetism.analyzer import Ordering
from pymatgen.core import Structure

from atomate2.common.flows.magnetism import MagneticOrderingsMaker
from atomate2.common.schemas.magnetism import MagneticOrderingsDocument


def test_magnetic_orderings(mock_vasp, clean_dir, test_dir):
    structure = Structure.from_file(
        test_dir
        / "vasp"
        / "MgMn2O4_magnetic"
        / "relax_1_3_(fm)"
        / "inputs"
        / "POSCAR.gz"
    )

    ref_paths = {
        "relax 1/3 (fm)": "MgMn2O4_magnetic/relax_1_3_(fm)",
        "relax 2/3 (afm)": "MgMn2O4_magnetic/relax_2_3_(afm)",
        "relax 3/3 (afm)": "MgMn2O4_magnetic/relax_3_3_(afm)",
        "static 1/3 (fm)": "MgMn2O4_magnetic/static_1_3_(fm)",
        "static 2/3 (afm)": "MgMn2O4_magnetic/static_2_3_(afm)",
        "static 3/3 (afm)": "MgMn2O4_magnetic/static_3_3_(afm)",
    }

    fake_run_vasp_kwargs = {
        "relax 1/3 (fm)": {"incar_settings": ["NSW", "ISMEAR"]},
        "relax 2/3 (afm)": {"incar_settings": ["NSW", "ISMEAR"]},
        "relax 3/3 (afm)": {"incar_settings": ["NSW", "ISMEAR"]},
        "static 1/3 (fm)": {"incar_settings": ["NSW", "ISMEAR"]},
        "static 2/3 (afm)": {"incar_settings": ["NSW", "ISMEAR"]},
        "static 3/3 (afm)": {"incar_settings": ["NSW", "ISMEAR"]},
    }

    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    flow = MagneticOrderingsMaker().make(structure)

    responses = run_locally(flow, create_folders=True, ensure_success=True)

    final_output = responses[flow.jobs[-1].uuid][1].output
    assert isinstance(final_output, MagneticOrderingsDocument)
    assert len(final_output.outputs) == 3
    assert (
        final_output.ground_state_uuid
        == min(final_output.outputs, key=lambda doc: doc.energy_per_atom).uuid
    )
    assert final_output.ground_state_ordering == Ordering.AFM
    assert final_output.ground_state_energy == pytest.approx(-104.29910777)
    assert final_output.ground_state_energy_per_atom == pytest.approx(-7.44993626929)
