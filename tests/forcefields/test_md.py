"""Tests for forcefield MD flows."""

import sys
from contextlib import nullcontext
from importlib.metadata import version as get_imported_version
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from ase import units
from ase.io import Trajectory as AseTrajectory
from ase.md.verlet import VelocityVerlet
from emmet.core.trajectory import AtomTrajectory
from jobflow import run_locally
from monty.serialization import loadfn
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.core.trajectory import Trajectory as PmgTrajectory

from atomate2.forcefields import MLFF
from atomate2.forcefields.md import ForceFieldMDMaker
from atomate2.forcefields.schemas import ForceFieldTaskDocument


def test_maker_initialization():
    # test that makers can be initialized from str or value enum

    from atomate2.forcefields import MLFF

    for mlff in MLFF.__members__:
        context_mgr = nullcontext()
        if mlff == "MACE":
            context_mgr = pytest.warns(UserWarning, match="default MP-trained MACE")

        with context_mgr:
            assert ForceFieldMDMaker(force_field_name=MLFF(mlff)) == ForceFieldMDMaker(
                force_field_name=mlff
            )
            assert ForceFieldMDMaker(
                force_field_name=str(MLFF(mlff))
            ) == ForceFieldMDMaker(force_field_name=mlff)


_mlffs_for_test = set(MLFF).difference(
    map(MLFF, ("Forcefield", "MatterSim", "Allegro", "OCP", "M3GNet", "MACE"))
)
_md_test_params = sorted(product(_mlffs_for_test, [True, False]), key=str)


@pytest.mark.parametrize("ff_name, use_emmet_models", _md_test_params)
def test_ml_ff_md_maker(
    ff_name,
    use_emmet_models,
    si_structure,
    sr_ti_o3_structure,
    al2_au_structure,
    test_dir,
    clean_dir,
    get_deepmd_pretrained_model_path,
):
    if ff_name == MLFF.GAP and sys.version_info >= (3, 12):
        pytest.skip(
            "GAP model not compatible with Python 3.12, waiting on https://github.com/libAtoms/QUIP/issues/645"
        )

    n_steps = 5

    ref_energies_per_atom = {
        MLFF.CHGNet: -5.380889892578125,
        MLFF.M3GNet: -5.387282371520996,
        MLFF.MACE_MP_0: -5.311369895935059,
        MLFF.MACE_MPA_0: -5.40242338180542,
        MLFF.MACE_MP_0B3: -5.403963088989258,
        MLFF.GAP: -5.391255755606209,
        MLFF.NEP: -3.966232215741286,
        MLFF.Nequip: -8.84670181274414,
        MLFF.SevenNet: -5.394115447998047,
        MLFF.DeepMD: -744.6197365326168,
        MLFF.MATPES_PBE: -5.230762481689453,
        MLFF.MATPES_R2SCAN: -8.561729431152344,
    }

    # ASE can slightly change tolerances on structure positions
    matcher = StructureMatcher()

    calculator_kwargs = {}
    unit_cell_structure = si_structure.copy()
    if ff_name == MLFF.GAP:
        calculator_kwargs = {
            "args_str": "IP GAP",
            "param_filename": str(test_dir / "forcefields" / "gap" / "gap_file.xml"),
        }
    elif ff_name == MLFF.NEP:
        # NOTE: The test NEP model is specifically trained on 16 elemental metals
        # thus a new Al2Au structure is added.
        # The NEP model used for the tests is licensed under a
        # [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)
        # and downloaded from https://doi.org/10.5281/zenodo.10081677
        # Also cite the original work if you use this specific model : https://arxiv.org/abs/2311.04732
        calculator_kwargs = {
            "model_filename": test_dir / "forcefields" / "nep" / "nep.txt"
        }
        unit_cell_structure = al2_au_structure.copy()
    elif ff_name == MLFF.Nequip:
        calculator_kwargs = {
            "model_path": test_dir / "forcefields" / "nequip" / "nequip_ff_sr_ti_o3.pth"
        }
        unit_cell_structure = sr_ti_o3_structure.copy()
    elif ff_name == MLFF.DeepMD:
        calculator_kwargs = {"model": get_deepmd_pretrained_model_path}
        unit_cell_structure = sr_ti_o3_structure.copy()

    structure = unit_cell_structure.to_conventional() * (2, 2, 2)

    job = ForceFieldMDMaker(
        force_field_name=ff_name,
        n_steps=n_steps,
        traj_file="md_traj.json.gz",
        traj_file_fmt="pmg",
        store_trajectory="partial",
        # check that `structure` alias to `mol_or_struct` works:
        ionic_step_data=("energy", "forces", "stress", "structure"),
        calculator_kwargs=calculator_kwargs,
        use_emmet_models=use_emmet_models,
    ).make(structure)
    response = run_locally(job, ensure_success=True)
    task_doc = response[next(iter(response))][1].output

    # Check that energies are reasonably close to reference values
    assert task_doc.output.energy / len(structure) == pytest.approx(
        ref_energies_per_atom[ff_name], abs=0.1
    )

    # Check that we have the right number of MD steps
    # ASE logs the initial structure energy, and doesn't count this as an MD step
    assert matcher.fit(task_doc.output.ionic_steps[0].structure, structure)
    assert len(task_doc.output.ionic_steps) == n_steps + 1

    # Check that the ionic steps have the expected physical properties
    assert all(
        key in step.model_dump()
        for key in ("energy", "forces", "stress", "mol_or_struct")
        for step in task_doc.output.ionic_steps
    )

    # `structure` aliases `mol_or_struct`
    assert all(
        step.structure == step.mol_or_struct for step in task_doc.output.ionic_steps
    )

    # Check that the trajectory has expected physical properties
    assert task_doc.included_objects == ["trajectory"]
    assert len(task_doc.objects["trajectory"]) == n_steps + 1
    assert task_doc.objects == task_doc.forcefield_objects  # test legacy alias

    if use_emmet_models:
        assert all(
            getattr(task_doc.objects["trajectory"], key, None) is not None
            for key in ("energy", "forces", "stress", "velocities", "temperature")
        )
        assert isinstance(task_doc.objects["trajectory"], AtomTrajectory)
    else:
        assert all(
            frame.get(key) is not None
            for key in ("energy", "forces", "stress", "velocities", "temperature")
            for frame in task_doc.objects["trajectory"].frame_properties
        )
        assert isinstance(task_doc.objects["trajectory"], PmgTrajectory)


@pytest.mark.parametrize(
    "traj_file,ff_name", [("trajectory.json.gz", "CHGNet"), ("atoms.traj", "CHGNet")]
)
def test_traj_file(traj_file, ff_name, si_structure, clean_dir):
    n_steps = 5

    # Check that traj file written to disk is consistent with trajectory
    # stored to the task document

    if ".json.gz" in traj_file:
        traj_file_fmt = "pmg"
        traj_file_loader = loadfn
    else:
        traj_file_fmt = "ase"
        traj_file_loader = AseTrajectory

    structure = si_structure.to_conventional() * (2, 2, 2)
    job = ForceFieldMDMaker(
        force_field_name=ff_name,
        n_steps=n_steps,
        traj_file=traj_file,
        traj_file_fmt=traj_file_fmt,
        use_emmet_models=True,
    ).make(structure)
    response = run_locally(job, ensure_success=True)
    task_doc = response[next(iter(response))][1].output

    traj_from_file = traj_file_loader(traj_file)

    assert len(traj_from_file) == n_steps + 1

    if traj_file_fmt == "pmg":
        other_traj = {
            key: [
                traj_from_file.frame_properties[idx][key]
                for idx in range(len(traj_from_file))
            ]
            for key in ("energy", "temperature", "forces", "velocities")
        }
    elif traj_file_fmt == "ase":
        other_traj = {
            k: [getattr(traj_from_file[idx], v)() for idx in range(n_steps + 1)]
            for k, v in {
                "energy": "get_potential_energy",
                "temperature": "get_temperature",
                "forces": "get_forces",
                "velocities": "get_velocities",
            }.items()
        }

    assert all(
        np.all(
            np.abs(
                np.array(other_traj[key])
                - np.array(getattr(task_doc.objects["trajectory"], key))
            )
        )
        < 1e-6
        for key in ("energy", "temperature", "forces", "velocities")
    )


def test_nve_and_dynamics_obj(si_structure: Structure, test_dir: Path):
    # This test serves two purposes:
    # 1. Test the NVE calculator
    # 2. Test specifying the `dynamics` kwarg of the `MDMaker` as a str
    #    vs. as an ase.md.md.MolecularDynamics object

    output = {}
    for key in ("from_str", "from_dyn"):
        if key == "from_str":
            dyn = "velocityverlet"
        elif key == "from_dyn":
            dyn = VelocityVerlet

        job = ForceFieldMDMaker(
            force_field_name="CHGNet",
            ensemble="nve",
            dynamics=dyn,
            n_steps=50,
            traj_file=None,
            ionic_step_data=("energy", "forces", "stress", "structure"),
        ).make(si_structure)

        response = run_locally(job, ensure_success=True)
        output[key] = response[job.uuid][1].output

    # check that energy and volume are constants
    ref_toten = -10.7
    assert output["from_str"].output.energy == pytest.approx(ref_toten, abs=0.1)
    assert output["from_str"].output.structure.volume == pytest.approx(
        output["from_str"].input.structure.volume
    )

    assert all(
        step.energy == pytest.approx(ref_toten, abs=0.1)
        for step in output["from_str"].output.ionic_steps
    )
    # ensure that output is consistent if molecular dynamics object is specified
    # as str or as MolecularDynamics object
    for attr in ("energy", "forces", "stress", "structure"):
        vals = {
            key: getattr(output[key].output, attr) for key in ("from_str", "from_dyn")
        }

        if isinstance(vals["from_str"], float):
            assert vals["from_str"] == pytest.approx(vals["from_dyn"])
        elif isinstance(vals["from_str"], Structure):
            assert vals["from_str"] == vals["from_dyn"]
        else:
            assert all(
                vals["from_str"][i][j] == pytest.approx(vals["from_dyn"][i][j])
                for i in range(len(vals["from_str"]))
                for j in range(len(vals["from_str"][i]))
            )


@pytest.mark.parametrize("ff_name", ["CHGNet"])
def test_temp_schedule(ff_name, si_structure, clean_dir):
    n_steps = 50
    temp_schedule = [300, 3000]

    structure = si_structure.to_conventional() * (2, 2, 2)

    job = ForceFieldMDMaker(
        force_field_name=ff_name,
        n_steps=n_steps,
        traj_file=None,
        dynamics="nose-hoover",
        temperature=temp_schedule,
        ase_md_kwargs={"ttime": 50.0 * units.fs, "pfactor": None},
        use_emmet_models=True,
    ).make(structure)
    response = run_locally(job, ensure_success=True)
    task_doc = response[next(iter(response))][1].output

    temp_history = task_doc.objects["trajectory"].temperature

    assert temp_history[-1] > temp_schedule[0]


@pytest.mark.parametrize("ff_name", ["MACE-MP-0"])
def test_press_schedule(ff_name, si_structure, clean_dir):
    n_steps = 20
    press_schedule = [0, 10]  # kBar

    structure = si_structure.to_conventional() * (3, 3, 3)

    job = ForceFieldMDMaker(
        force_field_name=ff_name,
        ensemble="npt",
        n_steps=n_steps,
        traj_file="md_traj.json.gz",
        traj_file_fmt="pmg",
        dynamics="nose-hoover",
        pressure=press_schedule,
        ase_md_kwargs={
            "ttime": 50.0 * units.fs,
            "pfactor": (75.0 * units.fs) ** 2 * units.GPa,
        },
    ).make(structure)
    run_locally(job, ensure_success=True)

    traj_from_file = loadfn("md_traj.json.gz")

    stress_history = [
        sum(traj_from_file.frame_properties[idx]["stress"][:3]) / 3.0
        for idx in range(len(traj_from_file))
    ]

    assert stress_history[-1] < stress_history[0]


def test_ext_load_md_maker(si_structure: Structure):
    calculator_meta = {
        "@module": "mace.calculators",
        "@callable": "mace_mp",
    }

    unit_cell_structure = si_structure.copy()
    structure = unit_cell_structure.to_conventional() * (2, 2, 2)

    job = ForceFieldMDMaker(
        force_field_name=calculator_meta,
        n_steps=5,
        traj_file="md_traj.json.gz",
        traj_file_fmt="pmg",
        store_trajectory="partial",
        ionic_step_data=("energy", "forces", "stress", "mol_or_struct"),
    ).make(structure)
    response = run_locally(job, ensure_success=True)
    task_doc = response[next(iter(response))][1].output
    assert isinstance(task_doc, ForceFieldTaskDocument)

    assert task_doc.forcefield_name == "mace_mp"
    assert task_doc.forcefield_version == get_imported_version("mace_torch")
