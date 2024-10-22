from importlib.metadata import version as get_imported_version
from pathlib import Path

import pytest
from jobflow import run_locally
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pytest import approx, importorskip

from atomate2.forcefields.jobs import (
    CHGNetRelaxMaker,
    CHGNetStaticMaker,
    ForceFieldRelaxMaker,
    ForceFieldStaticMaker,
    GAPRelaxMaker,
    GAPStaticMaker,
    M3GNetRelaxMaker,
    M3GNetStaticMaker,
    MACERelaxMaker,
    MACEStaticMaker,
    NEPRelaxMaker,
    NEPStaticMaker,
    NequipRelaxMaker,
    NequipStaticMaker,
)
from atomate2.forcefields.schemas import ForceFieldTaskDocument


def test_maker_initialization():
    # test that makers can be initialized from str or value enum

    from atomate2.forcefields import MLFF

    for mlff in MLFF.__members__:
        assert ForceFieldRelaxMaker(
            force_field_name=MLFF(mlff)
        ) == ForceFieldRelaxMaker(force_field_name=mlff)
        assert ForceFieldRelaxMaker(
            force_field_name=str(MLFF(mlff))
        ) == ForceFieldRelaxMaker(force_field_name=mlff)


def test_chgnet_static_maker(si_structure):
    # generate job
    job = ForceFieldStaticMaker(
        force_field_name="CHGNet",
        ionic_step_data=("structure", "energy"),
    ).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validate job outputs
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, ForceFieldTaskDocument)
    assert output1.output.energy == approx(-10.6275062, rel=1e-4)
    assert output1.output.ionic_steps[-1].magmoms is None
    assert output1.output.n_steps == 1

    assert output1.forcefield_version == get_imported_version("chgnet")

    with pytest.warns(FutureWarning):
        CHGNetStaticMaker()


@pytest.mark.parametrize(
    "fix_symmetry, symprec", [(True, 1e-2), (False, 1e-2), (True, 1e-1)]
)
def test_chgnet_relax_maker_fix_symmetry(
    ba_ti_o3_structure: Structure,
    fix_symmetry: bool,
    symprec: float,
):
    # translate one atom to break symmetry but stay below symprec threshold
    ba_ti_o3_structure.translate_sites(1, [symprec / 10.0, 0, 0])
    job = ForceFieldRelaxMaker(
        force_field_name="CHGNet",
        relax_kwargs={"fmax": 0.01},
        fix_symmetry=fix_symmetry,
        symprec=symprec,
    ).make(ba_ti_o3_structure)
    # get space group number of input structure
    initial_space_group = SpacegroupAnalyzer(
        ba_ti_o3_structure, symprec=symprec
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


@pytest.mark.parametrize("relax_cell", [True, False])
def test_chgnet_relax_maker(si_structure: Structure, relax_cell: bool):
    # translate one atom to ensure a small number of relaxation steps are taken
    si_structure.translate_sites(0, [0, 0, 0.1])

    max_step = 25
    # generate job
    job = ForceFieldRelaxMaker(
        force_field_name="CHGNet",
        steps=max_step,
        relax_cell=relax_cell,
    ).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validate job outputs
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, ForceFieldTaskDocument)
    if relax_cell:
        assert not output1.is_force_converged
        assert output1.output.n_steps == max_step + 2
        assert output1.output.energy == approx(-10.62461, abs=1e-2)
        assert output1.output.ionic_steps[-1].magmoms[0] == approx(0.00251674, rel=1e-1)
    else:
        assert output1.is_force_converged
        assert output1.output.n_steps == 13
        assert output1.output.energy == approx(-10.6274, rel=1e-2)
        assert output1.output.ionic_steps[-1].magmoms[0] == approx(0.00303572, rel=1e-2)

    # check the force_field_task_doc attributes
    assert Path(responses[job.uuid][1].output.dir_name).exists()

    with pytest.warns(FutureWarning):
        CHGNetRelaxMaker()


@pytest.mark.skip(reason="M3GNet requires DGL which is PyTorch 2.4 incompatible")
def test_m3gnet_static_maker(si_structure):
    # generate job
    job = ForceFieldStaticMaker(
        force_field_name="M3GNet",
        ionic_step_data=("structure", "energy"),
    ).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validate job outputs
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, ForceFieldTaskDocument)
    assert output1.output.energy == approx(-10.8, abs=0.2)
    assert output1.output.n_steps == 1

    assert output1.forcefield_version == get_imported_version("matgl")

    with pytest.warns(FutureWarning):
        M3GNetStaticMaker()


@pytest.mark.skip(reason="M3GNet requires DGL which is PyTorch 2.4 incompatible")
def test_m3gnet_relax_maker(si_structure):
    # translate one atom to ensure a small number of relaxation steps are taken
    si_structure.translate_sites(0, [0, 0, 0.1])

    # generate job
    max_step = 25
    job = ForceFieldRelaxMaker(
        force_field_name="M3GNet",
        steps=max_step,
    ).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validate job outputs
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, ForceFieldTaskDocument)
    assert output1.is_force_converged
    assert output1.output.energy == approx(-10.8, abs=0.2)
    assert output1.output.n_steps == 24

    with pytest.warns(FutureWarning):
        M3GNetRelaxMaker()


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
    # generate job
    # NOTE the test model is not trained on Si, so the energy is not accurate
    job = ForceFieldStaticMaker(
        force_field_name="MACE",
        ionic_step_data=("structure", "energy"),
        calculator_kwargs={"model": model},
    ).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validation the outputs of the job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, ForceFieldTaskDocument)
    assert output1.output.energy == approx(-0.068231, rel=1e-4)
    assert output1.output.n_steps == 1
    assert output1.forcefield_version == get_imported_version("mace-torch")

    with pytest.warns(FutureWarning):
        MACEStaticMaker()


@pytest.mark.parametrize(
    "fix_symmetry, symprec", [(True, 1e-2), (False, 1e-2), (True, 1e-1)]
)
def test_mace_relax_maker_fix_symmetry(
    ba_ti_o3_structure: Structure,
    fix_symmetry: bool,
    symprec: float,
):
    # translate one atom to break symmetry but stay below symprec threshold
    ba_ti_o3_structure.translate_sites(1, [symprec / 10.0, 0, 0])
    job = ForceFieldRelaxMaker(
        force_field_name="MACE",
        relax_kwargs={"fmax": 0.02},
        fix_symmetry=fix_symmetry,
        symprec=symprec,
    ).make(ba_ti_o3_structure)
    # get space group number of input structure
    initial_space_group = SpacegroupAnalyzer(
        ba_ti_o3_structure, symprec=symprec
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

    with pytest.warns(FutureWarning):
        MACERelaxMaker()


@pytest.mark.parametrize(
    "fix_symmetry, symprec", [(True, 1e-2), (False, 1e-2), (True, 1e-1)]
)
@pytest.mark.parametrize("relax_cell", [True, False])
@mace_paths
def test_mace_relax_maker(
    si_structure: Structure,
    model,
    relax_cell: bool,
    fix_symmetry: bool,
    symprec: float,
):
    from ase.spacegroup.symmetrize import check_symmetry, is_subgroup

    _, init_spg_num = si_structure.get_space_group_info()
    assert init_spg_num == 227

    # translate one atom to ensure a small number of relaxation steps are taken
    si_structure.translate_sites(0, [0, 0, 0.1])

    _, init_spg_num = si_structure.get_space_group_info()
    assert init_spg_num == 12

    # generate job
    # NOTE the test model is not trained on Si, so the energy is not accurate
    job = ForceFieldRelaxMaker(
        force_field_name="MACE",
        calculator_kwargs={"model": model},
        steps=25,
        optimizer_kwargs={"optimizer": "BFGSLineSearch"},
        relax_cell=relax_cell,
        fix_symmetry=fix_symmetry,
        symprec=symprec,
        relax_kwargs={"fmax": 0.04},
    ).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validating the outputs of the job
    output1 = responses[job.uuid][1].output
    assert output1.is_force_converged
    assert isinstance(output1, ForceFieldTaskDocument)

    si_atoms = si_structure.to_ase_atoms()
    symmetry_ops_init = check_symmetry(si_atoms, symprec=1.0e-3)
    si_atoms_final = output1.output.structure.to_ase_atoms()
    symmetry_ops_final = check_symmetry(si_atoms_final, symprec=1.0e-3)

    # get space group number of input structure
    _, final_spg_num = output1.output.structure.get_space_group_info()
    if relax_cell:
        assert final_spg_num == 12
    else:
        assert final_spg_num == 74

    if fix_symmetry:  # if symmetry is fixed, the symmetry should be the same or higher
        assert is_subgroup(symmetry_ops_init, symmetry_ops_final)
    else:  # if symmetry is not fixed, it can both increase or decrease
        assert not is_subgroup(symmetry_ops_init, symmetry_ops_final)

    if relax_cell:
        assert output1.output.energy == approx(-0.071117445, rel=1e-1)
        assert output1.output.n_steps >= 4
    else:
        assert output1.output.energy == approx(-0.06772976, rel=1e-4)
        assert output1.output.n_steps == 7


def test_gap_static_maker(si_structure: Structure, test_dir):
    importorskip("quippy")

    # generate job
    # Test files have been provided by @YuanbinLiu (University of Oxford)
    job = ForceFieldStaticMaker(
        force_field_name="GAP",
        ionic_step_data=("structure", "energy"),
        calculator_kwargs={
            "args_str": "IP GAP",
            "param_filename": str(test_dir / "forcefields" / "gap" / "gap_file.xml"),
        },
    ).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validation the outputs of the job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, ForceFieldTaskDocument)
    assert output1.output.energy == approx(-10.8523, rel=1e-4)
    assert output1.output.n_steps == 1
    assert output1.forcefield_version == get_imported_version("quippy-ase")

    with pytest.warns(FutureWarning):
        GAPStaticMaker()


@pytest.mark.parametrize("relax_cell", [True, False])
def test_gap_relax_maker(si_structure: Structure, test_dir: Path, relax_cell: bool):
    importorskip("quippy")

    # translate one atom to ensure a small number of relaxation steps are taken
    si_structure.translate_sites(0, [0, 0, 0.1])

    # generate job
    # Test files have been provided by @YuanbinLiu (University of Oxford)
    max_step = 25
    job = ForceFieldRelaxMaker(
        force_field_name="GAP",
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

    with pytest.warns(FutureWarning):
        GAPRelaxMaker()


def test_nep_static_maker(al2_au_structure: Structure, test_dir: Path):
    # NOTE: The test NEP model is specifically trained on 16 elemental metals
    # thus a new Al2Au structure is added.
    # The NEP model used for the tests is licensed under a
    # [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)
    # and downloaded from https://doi.org/10.5281/zenodo.10081677
    # Also cite the original work if you use this specific model : https://arxiv.org/abs/2311.04732
    job = ForceFieldStaticMaker(
        force_field_name="NEP",
        ionic_step_data=("structure", "energy"),
        calculator_kwargs={
            "model_filename": test_dir / "forcefields" / "nep" / "nep.txt"
        },
    ).make(al2_au_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validation the outputs of the job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, ForceFieldTaskDocument)
    assert output1.output.energy == approx(-47.65972, rel=1e-4)
    assert output1.output.n_steps == 1

    with pytest.warns(FutureWarning):
        NEPStaticMaker()


@pytest.mark.parametrize(
    ("relax_cell", "fix_symmetry"),
    [(True, False), (False, True)],
)
def test_nep_relax_maker(
    al2_au_structure: Structure,
    test_dir: Path,
    relax_cell: bool,
    fix_symmetry: bool,
):
    # NOTE: The test NEP model is specifically trained on 16 elemental metals
    # thus a new Al2Au structure is added.
    # The NEP model used for the tests is licensed under a
    # [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)
    # and downloaded from https://doi.org/10.5281/zenodo.10081677
    # Also cite the original work if you use this specific model : https://arxiv.org/abs/2311.04732

    # generate job
    job = ForceFieldRelaxMaker(
        force_field_name="NEP",
        steps=25,
        optimizer_kwargs={"optimizer": "BFGSLineSearch"},
        relax_cell=relax_cell,
        fix_symmetry=fix_symmetry,
        calculator_kwargs={
            "model_filename": test_dir / "forcefields" / "nep" / "nep.txt"
        },
    ).make(al2_au_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, ensure_success=True)

    # validate the outputs of the job
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, ForceFieldTaskDocument)
    if relax_cell:
        assert output1.output.energy == approx(-47.6727, rel=1e-3)
        assert output1.output.n_steps == 3
    else:
        assert output1.output.energy == approx(-47.659721, rel=1e-4)
        assert output1.output.n_steps == 2

    # fix_symmetry makes no difference for this structure relaxer combo
    # just testing that passing fix_symmetry doesn't break
    final_spg_num = output1.output.structure.get_space_group_info()[1]
    assert final_spg_num == 225

    with pytest.warns(FutureWarning):
        NEPRelaxMaker()


def test_nequip_static_maker(sr_ti_o3_structure: Structure, test_dir: Path):
    importorskip("nequip")

    # generate job
    # NOTE the test model is not trained on Si, so the energy is not accurate
    job = ForceFieldStaticMaker(
        force_field_name="Nequip",
        ionic_step_data=("structure", "energy"),
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

    with pytest.warns(FutureWarning):
        NequipStaticMaker()


@pytest.mark.parametrize(
    ("relax_cell", "fix_symmetry"),
    [(True, False), (False, True)],
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
    job = ForceFieldRelaxMaker(
        force_field_name="Nequip",
        steps=25,
        optimizer_kwargs={"optimizer": "BFGSLineSearch"},
        relax_cell=relax_cell,
        fix_symmetry=fix_symmetry,
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

    # fix_symmetry makes no difference for this structure relaxer combo
    # just testing that passing fix_symmetry doesn't break
    final_spg_num = output1.output.structure.get_space_group_info()[1]
    assert final_spg_num == 99

    with pytest.warns(FutureWarning):
        NequipRelaxMaker()
