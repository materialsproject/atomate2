from copy import deepcopy
from itertools import product
from pathlib import Path

import pytest
from jobflow import run_locally
from numpy.testing import assert_allclose
from pymatgen.core.structure import Structure

from atomate2.common.schemas.qha import PhononQHADoc
from atomate2.forcefields.flows.phonons import PhononMaker
from atomate2.forcefields.flows.qha import CHGNetQhaMaker, ForceFieldQhaMaker
from atomate2.forcefields.jobs import ForceFieldRelaxMaker


def test_qha_dir(clean_dir, si_structure: Structure, tmp_path: Path):
    # TODO brittle due to inability to adjust dtypes in CHGNetRelaxMaker

    flow = CHGNetQhaMaker(
        number_of_frames=5,
        ignore_imaginary_modes=True,
        min_length=10,
        phonon_maker=PhononMaker(
            store_force_constants=False,
            bulk_relax_maker=None,
            generate_frequencies_eigenvectors_kwargs={
                "tol_imaginary_modes": 5e-1,
                "tmin": 0,
                "tmax": 1000,
                "tstep": 10,
            },
        ),
    ).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # # validate the outputs
    ph_bs_dos_doc = responses[flow[-1].uuid][1].output
    assert isinstance(ph_bs_dos_doc, PhononQHADoc)


def test_qha_dir_change_defaults(clean_dir, si_structure: Structure, tmp_path: Path):
    # TODO brittle due to inability to adjust dtypes in CHGNetRelaxMaker

    flow = CHGNetQhaMaker(
        number_of_frames=4,
        ignore_imaginary_modes=True,
        linear_strain=(-0.03, 0.03),
        min_length=10,
        phonon_maker=PhononMaker(
            store_force_constants=False,
            bulk_relax_maker=None,
            generate_frequencies_eigenvectors_kwargs={
                "tol_imaginary_modes": 5e-1,
                "tmin": 0,
                "tmax": 1000,
                "tstep": 10,
            },
        ),
    ).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # # validate the outputs
    ph_bs_dos_doc = responses[flow[-1].uuid][1].output
    assert isinstance(ph_bs_dos_doc, PhononQHADoc)
    assert len(ph_bs_dos_doc.volumes) == 5
    assert ph_bs_dos_doc.volumes[0] == pytest.approx(ph_bs_dos_doc.volumes[2] * 0.97**3)
    assert ph_bs_dos_doc.volumes[4] == pytest.approx(ph_bs_dos_doc.volumes[2] * 1.03**3)


def test_qha_dir_manual_supercell(clean_dir, si_structure: Structure, tmp_path: Path):
    # TODO brittle due to inability to adjust dtypes in CHGNetRelaxMaker
    matrix = [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    flow = CHGNetQhaMaker(
        number_of_frames=4,
        ignore_imaginary_modes=True,
        min_length=10,
        phonon_maker=PhononMaker(
            store_force_constants=False,
            bulk_relax_maker=None,
            generate_frequencies_eigenvectors_kwargs={
                "tol_imaginary_modes": 5e-1,
                "tmin": 0,
                "tmax": 1000,
                "tstep": 10,
            },
        ),
    ).make(si_structure, supercell_matrix=matrix)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # # validate the outputs
    ph_bs_dos_doc = responses[flow[-1].uuid][1].output
    assert isinstance(ph_bs_dos_doc, PhononQHADoc)
    assert_allclose(ph_bs_dos_doc.supercell_matrix, matrix)


@pytest.mark.parametrize(
    "mlff,relax_initial_structure,run_eos_flow",
    list(product(("CHGNet", "M3GNet"), (True, False), (True, False))),
)
def test_instantiation(mlff: str, relax_initial_structure: bool, run_eos_flow: bool):
    no_maker = ["phonon_maker.bulk_relax_maker"]
    has_maker = [
        "phonon_maker.static_energy_maker",
        "phonon_maker.phonon_displacement_maker",
    ]

    for k, v in {
        "initial_relax_maker": relax_initial_structure,
        "eos_relax_maker": run_eos_flow,
    }.items():
        if v:
            has_maker.append(k)
        else:
            no_maker.append(k)
    if relax_initial_structure:
        has_maker.append("initial_relax_maker")
    else:
        no_maker.append("initial_relax_maker")

    tests = {
        **dict.fromkeys(has_maker, True),
        **dict.fromkeys(no_maker, False),
    }

    maker = ForceFieldQhaMaker.from_force_field_name(
        mlff, relax_initial_structure=relax_initial_structure, run_eos_flow=run_eos_flow
    )

    for attr, test_has_attr in tests.items():
        sub_maker = deepcopy(maker)
        for sub_attr in attr.split("."):
            sub_maker = getattr(sub_maker, sub_attr)

        if test_has_attr:
            assert isinstance(sub_maker, ForceFieldRelaxMaker)
            assert mlff in sub_maker.force_field_name
        else:
            assert sub_maker is None


def test_ext_load_qha_initialization():
    calculator_meta = {
        "@module": "mace.calculators",
        "@callable": "mace_mp",
    }
    maker = ForceFieldQhaMaker.from_force_field_name(
        calculator_meta, relax_initial_structure=True, run_eos_flow=True
    )

    ase_calculator_name = "mace_mp"
    assert maker.initial_relax_maker.ase_calculator_name == ase_calculator_name
    assert maker.eos_relax_maker.ase_calculator_name == ase_calculator_name
    assert maker.phonon_maker.ase_calculator_name == ase_calculator_name
