import pytest
from emmet.core.tasks import TaskDoc
from emmet.core.vasp.calculation import VaspObject
from jobflow import run_locally
from pymatgen.electronic_structure.bandstructure import (
    BandStructure,
    BandStructureSymmLine,
)

from atomate2.vasp.flows.core import (
    BandStructureMaker,
    DoubleRelaxMaker,
    HSEBandStructureMaker,
    HSELineModeBandStructureMaker,
    HSEOpticsMaker,
    HSEUniformBandStructureMaker,
    LineModeBandStructureMaker,
    MVLGWBandStructureMaker,
    OpticsMaker,
    UniformBandStructureMaker,
)
from atomate2.vasp.jobs.core import RelaxMaker
from atomate2.vasp.sets.core import RelaxSetGenerator


def test_double_relax(mock_vasp, clean_dir, si_structure):
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

    # generate flow
    flow = DoubleRelaxMaker().make(si_structure)

    # Run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # validate output
    output1 = responses[flow.jobs[0].uuid][1].output
    output2 = responses[flow.jobs[1].uuid][1].output

    assert isinstance(output1, TaskDoc)
    assert output1.output.energy == pytest.approx(-10.85043620)
    assert output2.output.energy == pytest.approx(-10.84177648)

    # Now try with two identical but non-default makers
    ref_paths = {
        "relax 1": "Si_double_relax_swaps/swap1/relax_1",
        "relax 2": "Si_double_relax_swaps/swap1/relax_2",
    }
    fake_run_vasp_kwargs = {
        "relax 1": {"incar_settings": ["LREAL"]},
        "relax 2": {"incar_settings": ["LREAL"]},
    }
    mock_vasp(ref_paths, fake_run_vasp_kwargs)
    my_custom_maker = RelaxMaker(
        input_set_generator=RelaxSetGenerator(user_incar_settings={"LREAL": "Auto"})
    )
    flow = DoubleRelaxMaker(
        relax_maker1=my_custom_maker, relax_maker2=my_custom_maker
    ).make(si_structure)
    run_locally(flow, create_folders=True, ensure_success=True)

    # Try the same as above but with the .from_relax_maker() class method
    flow = DoubleRelaxMaker.from_relax_maker(my_custom_maker).make(si_structure)
    run_locally(flow, create_folders=True, ensure_success=True)

    # Try DoubleRelaxMaker with a non-default second maker only
    ref_paths = {
        "relax 1": "Si_double_relax_swaps/swap2/relax_1",
        "relax 2": "Si_double_relax_swaps/swap2/relax_2",
    }
    fake_run_vasp_kwargs = {
        "relax 1": {"incar_settings": ["LREAL"]},
        "relax 2": {"incar_settings": ["LREAL"]},
    }
    mock_vasp(ref_paths, fake_run_vasp_kwargs)
    flow = DoubleRelaxMaker(relax_maker2=my_custom_maker).make(si_structure)
    run_locally(flow, create_folders=True, ensure_success=True)


def test_band_structure(mock_vasp, clean_dir, si_structure):
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
    assert set(uniform_output.vasp_objects) == {
        VaspObject.BANDSTRUCTURE,
        VaspObject.DOS,
    }

    assert isinstance(
        line_output.vasp_objects[VaspObject.BANDSTRUCTURE], BandStructureSymmLine
    )
    assert isinstance(
        uniform_output.vasp_objects[VaspObject.BANDSTRUCTURE], BandStructure
    )


def test_uniform_band_structure(mock_vasp, clean_dir, si_structure):
    # mapping from job name to directory containing test files
    ref_paths = {
        "non-scf uniform": "Si_band_structure/non-scf_uniform",
        "static": "Si_band_structure/static",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "non-scf uniform": {"incar_settings": ["NSW", "ISMEAR"]},
        "static": {"incar_settings": ["NSW", "ISMEAR"]},
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # Generate the flow
    flow = UniformBandStructureMaker().make(si_structure)

    # Run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # Additional validation on the outputs of the jobs; e.g.,
    static_output = responses[flow.jobs[0].uuid][1].output
    uniform_output = responses[flow.jobs[1].uuid][1].output

    assert static_output.output.energy == pytest.approx(-10.85037078)
    assert static_output.included_objects is None
    assert set(uniform_output.vasp_objects) == {
        VaspObject.BANDSTRUCTURE,
        VaspObject.DOS,
    }
    assert isinstance(
        uniform_output.vasp_objects[VaspObject.BANDSTRUCTURE], BandStructure
    )


def test_line_mode_band_structure(mock_vasp, clean_dir, si_structure):
    # mapping from job name to directory containing test files
    ref_paths = {
        "non-scf line": "Si_band_structure/non-scf_line",
        "static": "Si_band_structure/static",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "non-scf line": {"incar_settings": ["NSW", "ISMEAR"]},
        "static": {"incar_settings": ["NSW", "ISMEAR"]},
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # Generate the flow
    flow = LineModeBandStructureMaker().make(si_structure)

    # Run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # Additional validation on the outputs of the jobs; e.g.,
    static_output = responses[flow.jobs[0].uuid][1].output
    line_output = responses[flow.jobs[1].uuid][1].output

    assert static_output.output.energy == pytest.approx(-10.85037078)
    assert static_output.included_objects is None
    assert isinstance(
        line_output.vasp_objects[VaspObject.BANDSTRUCTURE], BandStructureSymmLine
    )


def test_hse_band_structure(mock_vasp, clean_dir, si_structure):
    # mapping from job name to directory containing test files
    ref_paths = {
        "hse band structure line": "Si_hse_band_structure/hse_band_structure_line",
        "hse band structure uniform": "Si_hse_band_structure/"
        "hse_band_structure_uniform",
        "hse static": "Si_hse_band_structure/hse_static",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "hse band structure line": {"incar_settings": ["NSW", "ISMEAR"]},
        "hse band structure uniform": {"incar_settings": ["NSW", "ISMEAR"]},
        "hse static": {"incar_settings": ["NSW", "ISMEAR"]},
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # generate flow
    flow = HSEBandStructureMaker().make(si_structure)
    flow.jobs[0].maker.input_set_generator.user_incar_settings["KSPACING"] = 0.4

    # Run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # validation on the outputs
    static_output = responses[flow.jobs[0].uuid][1].output
    uniform_output = responses[flow.jobs[1].uuid][1].output
    line_output = responses[flow.jobs[2].uuid][1].output

    assert static_output.output.energy == pytest.approx(-12.52887403)
    assert static_output.included_objects is None
    assert set(uniform_output.vasp_objects) == {
        VaspObject.BANDSTRUCTURE,
        VaspObject.DOS,
    }

    assert isinstance(
        line_output.vasp_objects[VaspObject.BANDSTRUCTURE], BandStructureSymmLine
    )
    assert isinstance(
        uniform_output.vasp_objects[VaspObject.BANDSTRUCTURE], BandStructure
    )


def test_hse_uniform_band_structure(mock_vasp, clean_dir, si_structure):
    # mapping from job name to directory containing test files
    ref_paths = {
        "hse band structure uniform": "Si_hse_band_structure/"
        "hse_band_structure_uniform",
        "hse static": "Si_hse_band_structure/hse_static",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "hse band structure uniform": {"incar_settings": ["NSW", "ISMEAR"]},
        "hse static": {"incar_settings": ["NSW", "ISMEAR"]},
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # generate flow
    flow = HSEUniformBandStructureMaker().make(si_structure)
    flow.jobs[0].maker.input_set_generator.user_incar_settings["KSPACING"] = 0.4

    # Run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # validation on the outputs
    static_output = responses[flow.jobs[0].uuid][1].output
    uniform_output = responses[flow.jobs[1].uuid][1].output

    assert static_output.output.energy == pytest.approx(-12.52887403)
    assert static_output.included_objects is None
    assert set(uniform_output.vasp_objects) == {
        VaspObject.BANDSTRUCTURE,
        VaspObject.DOS,
    }
    assert isinstance(
        uniform_output.vasp_objects[VaspObject.BANDSTRUCTURE], BandStructure
    )


def test_hse_line_mode_band_structure(mock_vasp, clean_dir, si_structure):
    # mapping from job name to directory containing test files
    ref_paths = {
        "hse band structure line": "Si_hse_band_structure/hse_band_structure_line",
        "hse static": "Si_hse_band_structure/hse_static",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "hse band structure line": {"incar_settings": ["NSW", "ISMEAR"]},
        "hse static": {"incar_settings": ["NSW", "ISMEAR"]},
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # generate flow
    flow = HSELineModeBandStructureMaker().make(si_structure)
    flow.jobs[0].maker.input_set_generator.user_incar_settings["KSPACING"] = 0.4

    # Run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # validation on the outputs
    static_output = responses[flow.jobs[0].uuid][1].output
    line_output = responses[flow.jobs[1].uuid][1].output

    assert static_output.output.energy == pytest.approx(-12.52887403)
    assert static_output.included_objects is None
    assert isinstance(
        line_output.vasp_objects[VaspObject.BANDSTRUCTURE], BandStructureSymmLine
    )


def test_optics(mock_vasp, clean_dir, si_structure):
    # mapping from job name to directory containing test files
    ref_paths = {"optics": "Si_optics/optics", "static": "Si_optics/static"}

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "optics": {"incar_settings": ["NSW", "ISMEAR"]},
        "static": {"incar_settings": ["NSW", "ISMEAR"]},
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # generate flow
    flow = OpticsMaker().make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # validation on the outputs
    output1 = responses[flow.jobs[0].uuid][1].output
    output2 = responses[flow.jobs[1].uuid][1].output
    assert isinstance(output1, TaskDoc)
    assert isinstance(output2, TaskDoc)
    assert output1.output.energy == pytest.approx(-10.85037078)
    assert output2.calcs_reversed[0].output.frequency_dependent_dielectric.real[0] == [
        13.6062,
        13.6063,
        13.6062,
        0.0,
        0.0,
        0.0,
    ]


def test_hse_optics(mock_vasp, clean_dir, si_structure):
    # mapping from job name to directory containing test files
    ref_paths = {
        "hse optics": "Si_hse_optics/hse_optics",
        "hse static": "Si_hse_optics/hse_static",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "hse_optics": {"incar_settings": ["NSW", "ISMEAR"]},
        "hse_static": {"incar_settings": ["NSW", "ISMEAR"]},
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # generate flow
    flow = HSEOpticsMaker().make(si_structure)
    flow.update_maker_kwargs(
        {"_set": {"input_set_generator->user_incar_settings->KSPACING": 0.5}},
        dict_mod=True,
    )

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # validation on the outputs
    output1 = responses[flow.jobs[0].uuid][1].output
    output2 = responses[flow.jobs[1].uuid][1].output
    assert isinstance(output1, TaskDoc)
    assert isinstance(output2, TaskDoc)
    assert output1.output.energy == pytest.approx(-12.41767353)
    assert output2.calcs_reversed[0].output.frequency_dependent_dielectric.real[0] == [
        13.8738,
        13.8738,
        13.8738,
        0.0,
        0.0,
        0.0,
    ]


def test_mvl_gw(mock_vasp, clean_dir, si_structure):
    from emmet.core.tasks import TaskDoc
    from jobflow import run_locally

    from atomate2.vasp.powerups import (
        update_user_incar_settings,
        update_user_kpoints_settings,
        update_user_potcar_functional,
    )

    # mapping from job name to directory containing test files
    ref_paths = {
        "MVL G0W0": "Si_G0W0/MVL_G0W0",
        "MVL nscf": "Si_G0W0/MVL_nscf",
        "MVL static": "Si_G0W0/MVL_static",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "MVL G0W0": {"incar_settings": ["NSW", "ISMEAR"]},
        "MVL nscf": {"incar_settings": ["NSW", "ISMEAR"]},
        "MVL static": {"incar_settings": ["NSW", "ISMEAR"]},
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    flow = MVLGWBandStructureMaker().make(si_structure)
    flow = update_user_kpoints_settings(flow, kpoints_updates={"reciprocal_density": 5})
    flow = update_user_potcar_functional(flow, potcar_functional="PBE_54")
    flow = update_user_incar_settings(flow, incar_updates={"ISPIN": 1})

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # validate the outputs
    output1 = responses[flow.jobs[0].uuid][1].output
    output2 = responses[flow.jobs[1].uuid][1].output
    output3 = responses[flow.jobs[2].uuid][1].output
    assert isinstance(output1, TaskDoc)
    assert isinstance(output2, TaskDoc)
    assert isinstance(output3, TaskDoc)
    assert output1.output.energy == pytest.approx(-10.22237938)
    assert output1.output.energy == pytest.approx(0.7161)
    assert output2.output.energy == pytest.approx(-10.2223794)
    assert output3.output.bandgap == pytest.approx(1.3488000000000007)
