from importlib.metadata import version as get_imported_version
from pathlib import Path

import pytest
from jobflow import run_locally
from pymatgen.core import Structure
from pytest import approx, importorskip

from atomate2.forcefields.jobs import (
    CHGNetRelaxMaker,
    CHGNetStaticMaker,
    GAPRelaxMaker,
    GAPStaticMaker,
    M3GNetRelaxMaker,
    M3GNetStaticMaker,
    MACERelaxMaker,
    MACEStaticMaker,
    NequipRelaxMaker,
    NequipStaticMaker,
)
from atomate2.forcefields.schemas import ForceFieldTaskDocument


def test_chgnet_static_maker(si_structure):
    task_doc_kwargs = {"ionic_step_data": ("structure", "energy")}

    # generate job
    job = CHGNetStaticMaker(task_document_kwargs=task_doc_kwargs).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validate job outputs
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, ForceFieldTaskDocument)
    assert output1.output.energy == approx(-10.6275062, rel=1e-4)
    assert output1.output.ionic_steps[-1].magmoms is None
    assert output1.output.n_steps == 1

    assert output1.forcefield_version == get_imported_version("chgnet")


@pytest.mark.parametrize(
    "input_structure, fix_symmetry, relax_cell, symprec",
    [
        ("si_structure", False, False, None),
        ("si_structure", False, True, None),
        ("ba_ti_o3_structure", True, True, 1e-2),
        ("ba_ti_o3_structure", False, True, 1e-2),
        ("ba_ti_o3_structure", True, True, 1e-1),
    ],
    indirect=["input_structure"],
)
def test_chgnet_relax_maker(
    request,
    input_structure,
    test_dir: Path,
    relax_cell: bool,
    fix_symmetry: bool,
    symprec: float,
):
    if input_structure == request.getfixturevalue("ba_ti_o3_structure"):
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        # translate one atom to break symmetry but stay below symprec threshold
        input_structure.translate_sites(1, [symprec / 10.0, 0, 0])
        job = CHGNetRelaxMaker(
            relax_kwargs={"fmax": 0.01},
            fix_symmetry=fix_symmetry,
            symprec=symprec,
        ).make(input_structure)
        # get space group number of input structure
        initial_space_group = SpacegroupAnalyzer(
            input_structure, symprec=symprec
        ).get_space_group_number()
        responses = run_locally(job, ensure_success=True)
        output1 = responses[job.uuid][1].output
        assert output1.is_force_converged
        final_space_group = SpacegroupAnalyzer(
            output1.output.structure, symprec=symprec
        ).get_space_group_number()
        if fix_symmetry:
            assert initial_space_group == final_space_group
        else:
            assert initial_space_group != final_space_group
    else:
        # translate one atom to ensure a small number of relaxation steps are taken
        input_structure.translate_sites(0, [0, 0, 0.1])
        max_step = 25
        # generate job
        job = CHGNetRelaxMaker(steps=max_step, relax_cell=relax_cell).make(
            input_structure
        )

        # run the flow or job and ensure that it finished running successfully
        responses = run_locally(job, ensure_success=True)

        # validate job outputs
        output1 = responses[job.uuid][1].output
        assert isinstance(output1, ForceFieldTaskDocument)
        if relax_cell:
            assert not output1.is_force_converged
            assert output1.output.n_steps == max_step + 2
            assert output1.output.energy == approx(-10.62461, abs=1e-2)
            assert output1.output.ionic_steps[-1].magmoms[0] == approx(
                0.00251674, rel=1e-1
            )
        else:
            assert output1.is_force_converged
            assert output1.output.n_steps == 13
            assert output1.output.energy == approx(-10.6274, rel=1e-2)
            assert output1.output.ionic_steps[-1].magmoms[0] == approx(
                0.00303572, rel=1e-2
            )

        # check the force_field_task_doc attributes
        assert Path(responses[job.uuid][1].output.dir_name).exists()


def test_m3gnet_static_maker(si_structure):
    task_doc_kwargs = {"ionic_step_data": ("structure", "energy")}

    # generate job
    job = M3GNetStaticMaker(task_document_kwargs=task_doc_kwargs).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validate job outputs
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, ForceFieldTaskDocument)
    assert output1.output.energy == approx(-10.8, abs=0.2)
    assert output1.output.n_steps == 1

    assert output1.forcefield_version == get_imported_version("matgl")


def test_m3gnet_relax_maker(si_structure):
    # translate one atom to ensure a small number of relaxation steps are taken
    si_structure.translate_sites(0, [0, 0, 0.1])

    # generate job
    max_step = 25
    job = M3GNetRelaxMaker(steps=max_step).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validate job outputs
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, ForceFieldTaskDocument)
    assert output1.is_force_converged
    assert output1.output.energy == approx(-10.8, abs=0.2)
    assert output1.output.n_steps == 24


mace_paths = pytest.mark.parametrize(
    "model",
    [
        # None, # TODO uncomment once https://github.com/ACEsuit/mace/pull/230 is merged
        # to test loading MACE checkpoint on the fly from figshare
        f"{Path(__file__).parent.parent}/test_data/forcefields/mace/MACE.model",
    ],
)


@mace_paths
def test_mace_static_maker(si_structure: Structure, test_dir: Path, model):
    task_doc_kwargs = {"ionic_step_data": ("structure", "energy")}

    # generate job
    # NOTE the test model is not trained on Si, so the energy is not accurate
    job = MACEStaticMaker(
        calculator_kwargs={"model": model}, task_document_kwargs=task_doc_kwargs
    ).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validation the outputs of the job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, ForceFieldTaskDocument)
    assert output1.output.energy == approx(-0.068231, rel=1e-4)
    assert output1.output.n_steps == 1
    assert output1.forcefield_version == get_imported_version("mace-torch")


@pytest.mark.parametrize(
    "input_structure, fix_symmetry, relax_cell, symprec",
    [
        ("si_structure", False, False, None),
        ("si_structure", False, True, None),
        ("ba_ti_o3_structure", True, True, 1e-2),
        ("ba_ti_o3_structure", False, True, 1e-2),
        ("ba_ti_o3_structure", True, True, 1e-1),
    ],
    indirect=["input_structure"],
)
@mace_paths
def test_mace_relax_maker(
    request,
    input_structure,
    test_dir: Path,
    model,
    relax_cell: bool,
    fix_symmetry: bool,
    symprec: float,
):
    if input_structure == request.getfixturevalue("ba_ti_o3_structure"):
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        # translate one atom to break symmetry but stay below symprec threshold
        input_structure.translate_sites(1, [symprec / 10.0, 0, 0])
        job = MACERelaxMaker(
            relax_kwargs={"fmax": 0.01},
            fix_symmetry=fix_symmetry,
            symprec=symprec,
        ).make(input_structure)
        # get space group number of input structure
        initial_space_group = SpacegroupAnalyzer(
            input_structure, symprec=symprec
        ).get_space_group_number()
        responses = run_locally(job, ensure_success=True)
        output1 = responses[job.uuid][1].output
        assert output1.is_force_converged
        final_space_group = SpacegroupAnalyzer(
            output1.output.structure, symprec=symprec
        ).get_space_group_number()
        if fix_symmetry:
            assert initial_space_group == final_space_group
        else:
            assert initial_space_group != final_space_group
    else:
        # translate one atom to ensure a small number of relaxation steps are taken
        input_structure.translate_sites(0, [0, 0, 0.1])
        # generate job
        # NOTE the test model is not trained on Si, so the energy is not accurate
        job = MACERelaxMaker(
            calculator_kwargs={"model": model},
            steps=25,
            optimizer_kwargs={"optimizer": "BFGSLineSearch"},
            relax_cell=relax_cell,
        ).make(input_structure)

        # run the flow or job and ensure that it finished running successfully
        responses = run_locally(job, ensure_success=True)

        # validating the outputs of the job
        output1 = responses[job.uuid][1].output
        assert output1.is_force_converged
        assert isinstance(output1, ForceFieldTaskDocument)
        if relax_cell:
            assert output1.output.energy == approx(-0.0526856, rel=1e-1)
            assert output1.output.n_steps >= 4
        else:
            assert output1.output.energy == approx(-0.051912, rel=1e-4)
            assert output1.output.n_steps == 4


def test_gap_static_maker(si_structure: Structure, test_dir):
    importorskip("quippy")

    task_doc_kwargs = {"ionic_step_data": ("structure", "energy")}

    # generate job
    # Test files have been provided by Yuanbin Liu (University of Oxford)
    job = GAPStaticMaker(
        calculator_kwargs={
            "args_str": "IP GAP",
            "param_filename": str(test_dir / "forcefields" / "gap" / "gap_file.xml"),
        },
        task_document_kwargs=task_doc_kwargs,
    ).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validation the outputs of the job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, ForceFieldTaskDocument)
    assert output1.output.energy == approx(-10.8523, rel=1e-4)
    assert output1.output.n_steps == 1
    assert output1.forcefield_version == get_imported_version("quippy-ase")


def test_nequip_static_maker(sr_ti_o3_structure: Structure, test_dir: Path):
    importorskip("nequip")
    task_doc_kwargs = {"ionic_step_data": ("structure", "energy")}

    # generate job
    # NOTE the test model is not trained on Si, so the energy is not accurate
    job = NequipStaticMaker(
        task_document_kwargs=task_doc_kwargs,
        calculator_kwargs={
            "model_path": test_dir / "forcefields" / "nequip" / "nequip_ff_sr_ti_o3.pth"
        },
    ).make(sr_ti_o3_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validation the outputs of the job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, ForceFieldTaskDocument)
    assert output1.output.energy == approx(-44.40017, rel=1e-4)
    assert output1.output.n_steps == 1
    assert output1.forcefield_version == get_imported_version("nequip")


@pytest.mark.parametrize(
    ("relax_cell", "fix_symmetry"),
    [(True, False), (False, False), (True, True), (False, True)],
)
def test_nequip_relax_maker(
    sr_ti_o3_structure: Structure,
    test_dir: Path,
    relax_cell: bool,
    fix_symmetry: bool,
):
    importorskip("nequip")
    # translate one atom to ensure a small number of relaxation steps are taken
    sr_ti_o3_structure.translate_sites(0, [0, 0, 0.2])
    # generate job
    job = NequipRelaxMaker(
        steps=25,
        optimizer_kwargs={"optimizer": "BFGSLineSearch"},
        relax_cell=relax_cell,
        calculator_kwargs={
            "model_path": test_dir / "forcefields" / "nequip" / "nequip_ff_sr_ti_o3.pth"
        },
    ).make(sr_ti_o3_structure)
    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validation the outputs of the job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, ForceFieldTaskDocument)
    if relax_cell:
        assert output1.output.energy == approx(-44.407, rel=1e-3)
        assert output1.output.n_steps == 5
    else:
        assert output1.output.energy == approx(-44.40015, rel=1e-4)
        assert output1.output.n_steps == 5


@pytest.mark.parametrize("relax_cell", [True, False])
def test_gap_relax_maker(si_structure: Structure, test_dir: Path, relax_cell: bool):
    importorskip("quippy")

    # translate one atom to ensure a small number of relaxation steps are taken
    si_structure.translate_sites(0, [0, 0, 0.1])

    # generate job
    # Test files have been provided by Yuanbin Liu (University of Oxford)
    max_step = 25
    job = GAPRelaxMaker(
        calculator_kwargs={
            "args_str": "IP GAP",
            "param_filename": str(test_dir / "forcefields" / "gap" / "gap_file.xml"),
        },
        steps=max_step,
        relax_cell=relax_cell,
    ).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validating the outputs of the job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, ForceFieldTaskDocument)
    if relax_cell:
        assert not output1.is_force_converged
        assert output1.output.energy == approx(-13.08492, rel=1e-2)
        assert output1.output.n_steps == max_step + 2
    else:
        assert output1.is_force_converged
        assert output1.output.energy == approx(-10.8523, rel=1e-4)
        assert output1.output.n_steps == 17
