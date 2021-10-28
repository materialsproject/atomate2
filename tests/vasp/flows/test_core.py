import pytest


def test_double_relax(mock_vasp, clean_dir, si_structure):
    from jobflow import run_locally

    from atomate2.vasp.flows.core import DoubleRelaxMaker
    from atomate2.vasp.schemas.task import TaskDocument

    # mapping from job name to directory containing test files
    ref_paths = {
        "relax 1": "Si_double_relax/relax_1",
        "relax 2": "Si_double_relax/relax_2",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "relax 1": {"incar_settings": ["NSW", "ISMEAR"]},
        "relax 2": {"incar_settings": ["NSW", "ISMEAR"]},
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # !!! Insert code to generate flow/job below, i.e.:
    flow = DoubleRelaxMaker().make(si_structure)

    # Run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # !!! Insert additional validation on the outputs of the jobs; e.g.,
    output1 = responses[flow.jobs[0].uuid][1].output
    output2 = responses[flow.jobs[1].uuid][1].output

    print(output2.output.energy)

    assert isinstance(output1, TaskDocument)
    assert output1.output.energy == pytest.approx(-10.85083141)
    assert output2.output.energy == pytest.approx(-10.84177648)


def test_band_structure(mock_vasp, clean_dir, si_structure):
    from jobflow import run_locally
    from pymatgen.electronic_structure.bandstructure import (
        BandStructure,
        BandStructureSymmLine,
    )

    from atomate2.vasp.flows.core import BandStructureMaker
    from atomate2.vasp.schemas.calculation import VaspObject

    # mapping from job name to directory containing test files
    ref_paths = {
        "non-scf line": "Si_band_structure/non-scf_line",
        "non-scf uniform": "Si_band_structure/non-scf_uniform",
        "static": "Si_band_structure/static",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "non-scf line": {"incar_settings": ["NSW", "ISMEAR"]},
        "non-scf uniform": {"incar_settings": ["NSW", "ISMEAR"]},
        "static": {"incar_settings": ["NSW", "ISMEAR"]},
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # Generate the flow
    flow = BandStructureMaker().make(si_structure)

    # Run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # Additional validation on the outputs of the jobs; e.g.,
    static_output = responses[flow.jobs[0].uuid][1].output
    uniform_output = responses[flow.jobs[1].uuid][1].output
    line_output = responses[flow.jobs[2].uuid][1].output

    assert static_output.output.energy == pytest.approx(-10.85037078)
    assert static_output.included_objects is None
    assert set(uniform_output.vasp_objects) == set(
        [VaspObject.BANDSTRUCTURE, VaspObject.DOS]
    )

    assert isinstance(
        line_output.vasp_objects[VaspObject.BANDSTRUCTURE], BandStructureSymmLine
    )
    assert isinstance(
        uniform_output.vasp_objects[VaspObject.BANDSTRUCTURE], BandStructure
    )
