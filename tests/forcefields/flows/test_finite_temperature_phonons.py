"""Tests for the force field finite-temperature phonon workflow.

The tests run the workflow with ASE's EMT potential, so they need no reference
data and no machine-learned force field. pheasy is only installed in the
numpy-limited forcefield CI job, so the tests are skipped in the other
forcefield jobs.
"""

import pytest

pytest.importorskip("pheasy")

import numpy as np
from ase import units
from jobflow import run_locally
from pymatgen.core import Lattice, Structure

from atomate2.ase.md import MDEnsemble
from atomate2.forcefields.flows.finite_temperature_phonons import (
    ForceFieldFiniteTemperaturePhononMaker,
    _get_force_field_md_maker,
)
from atomate2.forcefields.jobs import (
    ForceFieldDielectricMaker,
    ForceFieldRelaxMaker,
    ForceFieldStaticMaker,
)
from atomate2.forcefields.md import ForceFieldMDMaker
from atomate2.vasp.jobs.core import DielectricMaker

EMT = {"@module": "ase.calculators.emt", "@callable": "EMT"}


@pytest.fixture
def cu3au():
    """L1_2 Cu3Au with Au first, so that sorting by electronegativity reorders it."""
    return Structure(
        Lattice.cubic(3.75),
        ["Au", "Cu", "Cu", "Cu"],
        [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
    )


def run_emt_flow(structure, **kwargs):
    """Run the force field workflow with EMT and return its output document."""
    # the Langevin thermostat of ASE draws from numpy's global generator
    np.random.seed(103)  # noqa: NPY002
    maker = ForceFieldFiniteTemperaturePhononMaker.from_force_field_name(
        EMT, min_length=7.0, md_time_step=2.0, **kwargs
    )
    flow = maker.make(structure)
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    return responses[flow.output.uuid][1].output


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"md_runs": 0}, "md_runs and n_snapshots must be at least 1"),
        ({"alpha_min": -2}, "alpha_min must be below -2"),
        ({"md_time_step": 0}, "must be positive"),
        ({"equilibration_time": 8.0}, "fewer than the 50 snapshots"),
        ({"equilibration_time": -1.0}, "fewer than the 50 snapshots"),
        (
            {"md_time": 0.01, "equilibration_time": 0, "n_snapshots": 5, "md_runs": 20},
            "larger than the 10",
        ),
        ({"md_maker": None}, "md_maker and phonon_displacement_maker must be set"),
        ({"code": "aims"}, "code must be one of"),
        ({"md_code": None}, "md_code must be one of"),
        ({"code": "vasp", "socket": True}, "socket is not supported by VASP"),
        (
            {"npt_maker": ForceFieldMDMaker(), "npt_equilibration_time": 8.0},
            "npt_equilibration_time must be",
        ),
        (
            {"fixed_cell_relax_maker": ForceFieldRelaxMaker()},
            "only used with npt_maker",
        ),
    ],
)
def test_maker_checks(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ForceFieldFiniteTemperaturePhononMaker(**kwargs)


def test_force_field_md_maker(cu3au):
    maker = ForceFieldFiniteTemperaturePhononMaker.from_force_field_name(
        EMT,
        md_time=1.0,
        md_time_step=2.0,
        equilibration_time=0.5,
        md_runs=3,
        random_seed=5,
    )
    assert maker.get_md_steps() == [167, 167, 166]
    md_maker = maker.get_md_maker(167)
    assert md_maker.n_steps == 167
    assert md_maker.time_step == 2.0
    assert md_maker.temperature == 300
    assert md_maker.mb_velocity_seed == 5
    assert md_maker.store_trajectory == "no"
    # Langevin by default, with the default friction of AseMDMaker
    assert md_maker.dynamics == "langevin"
    assert md_maker.ase_md_kwargs == {}
    # the maker of the flow is not changed
    assert maker.md_maker.n_steps == 1000
    assert maker.md_maker.dynamics is None
    # the force field relaxation keeps the symmetry
    assert maker.bulk_relax_maker.fix_symmetry
    assert ForceFieldFiniteTemperaturePhononMaker().bulk_relax_maker.fix_symmetry

    # the MD jobs have a number only when there are several
    flow = maker.make(cu3au, supercell_matrix=np.eye(3).tolist())
    md_names = [job.name for job in flow.jobs if "MD" in job.name]
    assert md_names == ["ASE MD 1/3", "ASE MD 2/3", "ASE MD 3/3"]
    maker = ForceFieldFiniteTemperaturePhononMaker(md_time=2.0)
    flow = maker.make(cu3au, supercell_matrix=np.eye(3).tolist())
    assert [job.name for job in flow.jobs if "MD" in job.name] == ["ASE MD"]

    nose_hoover = ForceFieldFiniteTemperaturePhononMaker(
        thermostat="nose-hoover", md_time_step=2.0
    ).get_md_maker(10)
    assert nose_hoover.dynamics == "nose-hoover-chain"
    assert nose_hoover.ase_md_kwargs["tchain"] == 1
    # a thermostat period of 40 steps, 80 fs
    period = np.pi * np.sqrt(2) * nose_hoover.ase_md_kwargs["tdamp"] / units.fs
    assert period == pytest.approx(80.0)

    maker = ForceFieldFiniteTemperaturePhononMaker(
        md_maker=ForceFieldMDMaker(ase_md_kwargs={"tdamp": 10})
    )
    with pytest.raises(ValueError, match="must not set dynamics or ase_md_kwargs"):
        maker.get_md_maker(10)
    with pytest.raises(TypeError, match="must be a ForceFieldMDMaker"):
        _get_force_field_md_maker(
            ForceFieldFiniteTemperaturePhononMaker(md_maker=ForceFieldStaticMaker()), 10
        )

    with pytest.raises(ValueError, match="diagonal supercell matrix"):
        maker.make(cu3au, supercell_matrix=[[1, 1, 0], [0, 1, 0], [0, 0, 1]])


def test_force_field_npt_maker(cu3au):
    maker = ForceFieldFiniteTemperaturePhononMaker.from_force_field_name(
        EMT, run_npt=True, md_time_step=2.0, pressure=10.0
    )
    npt_maker = maker.get_npt_maker(100)
    assert npt_maker.ensemble == MDEnsemble.npt
    assert npt_maker.dynamics == "nose-hoover-chain"
    assert npt_maker.n_steps == 100
    assert npt_maker.pressure == 10.0
    # a barostat time constant of 1000 steps
    assert npt_maker.ase_md_kwargs["pdamp"] / units.fs == pytest.approx(2000)
    assert maker.get_md_maker(100).ensemble == MDEnsemble.nvt
    # the atoms are relaxed in the cell from the NPT MD
    assert not maker.fixed_cell_relax_maker.relax_cell
    assert maker.fixed_cell_relax_maker.fix_symmetry

    flow = maker.make(cu3au, supercell_matrix=np.eye(3).tolist())
    names = [job.name for job in flow.jobs]
    assert "ASE MD NPT" in names
    assert "get_npt_structure" in names

    # no NPT MD by default
    maker = ForceFieldFiniteTemperaturePhononMaker.from_force_field_name(EMT)
    assert maker.npt_maker is None
    assert maker.fixed_cell_relax_maker is None
    flow = maker.make(cu3au, supercell_matrix=np.eye(3).tolist())
    assert not any("NPT" in job.name for job in flow.jobs)


def test_force_field_with_vasp_born_charges(cu3au):
    """A VASP born_maker after a force field relaxation gets no prev_dir."""
    maker = ForceFieldFiniteTemperaturePhononMaker.from_force_field_name(EMT)
    maker.born_maker = DielectricMaker()
    flow = maker.make(cu3au, supercell_matrix=np.eye(3).tolist())
    born_job = next(job for job in flow.jobs if job.name == "dielectric")
    assert born_job.function_kwargs.get("prev_dir") is None


def test_finite_temperature_phonon_maker_emt(clean_dir, cu3au):
    """Run the whole workflow with EMT on Cu3Au at 300 K."""
    # a 2x2x2 supercell of the cubic cell, 32 atoms, and 2 ps of MD at 2 fs in
    # two MD jobs
    doc = run_emt_flow(
        cu3au, md_time=2.0, md_runs=2, equilibration_time=0.5, n_snapshots=12
    )

    assert [str(site.specie) for site in doc.structure] == ["Cu", "Cu", "Cu", "Au"]
    assert doc.temperature == 300
    assert doc.thermostat == "langevin"
    assert doc.md_code == doc.code == "forcefields"
    assert doc.md_force_field_name == doc.force_field_name == "ase.calculators.emt.EMT"
    assert doc.force_field_kwargs == {}
    assert doc.md_time == pytest.approx(2.0)
    assert doc.n_snapshots == 12
    # 250 steps of 2 fs are left out
    assert doc.snapshot_times[0] == pytest.approx(0.502)
    assert doc.snapshot_times[-1] == pytest.approx(2.0)
    assert len(doc.md_uuids) == len(doc.md_job_dirs) == 2
    assert len(doc.uuids.displacements_uuids) == 13
    assert doc.uuids.optimization_run_uuid is not None
    assert doc.uuids.born_run_uuid is None
    assert doc.supercell_matrix == ((2, 0, 0), (0, 2, 0), (0, 0, 2))
    assert doc.trajectory_health.verdict == "stable"
    assert doc.max_residual_force < 1e-8
    assert np.array(doc.force_constants.force_constants).shape == (32, 32, 3, 3)
    # Cu3Au is stable at 300 K
    assert not doc.has_imaginary_modes
    assert doc.n_imaginary_modes == 0
    # the snapshot amplitude agrees with the one of the fitted force constants
    assert doc.rms_displacement == pytest.approx(
        doc.rms_displacement_from_force_constants, rel=0.25
    )
    # values of this run, to catch changes of the results
    assert np.max(doc.phonon_bandstructure.bands) == pytest.approx(6.815, rel=0.05)
    # the acoustic modes at Gamma
    assert doc.lowest_frequency == pytest.approx(0, abs=1e-3)
    assert doc.rms_displacement == pytest.approx(0.1235, rel=0.05)
    assert doc.force_rmse == pytest.approx(0.1045, rel=0.15)


def test_finite_temperature_phonon_maker_emt_npt(clean_dir, cu3au):
    """Run the workflow with an NPT MD first, with EMT on Cu3Au at 300 K."""
    doc = run_emt_flow(
        cu3au,
        run_npt=True,
        npt_time=2.0,
        npt_equilibration_time=0.5,
        md_time=1.0,
        equilibration_time=0.2,
        n_snapshots=10,
    )

    assert doc.pressure == 0.0
    assert doc.npt_time == 2.0
    assert doc.npt_equilibration_time == 0.5
    assert doc.npt_trajectory_health.verdict == "stable"
    assert doc.npt_uuid is not None
    assert doc.npt_job_dir is not None
    assert doc.fixed_cell_relax_uuid is not None
    # the atoms are relaxed in the cell from the NPT MD
    assert doc.max_residual_force < 1e-3
    # the NVT MD and the fit use the cubic cell from the NPT MD, which expanded
    assert doc.structure.lattice.abc == pytest.approx([doc.structure.lattice.a] * 3)
    assert doc.structure.lattice.angles == pytest.approx((90, 90, 90))
    assert doc.structure.volume > doc.npt_input_structure.volume
    assert doc.structure.lattice.a == pytest.approx(3.7328, rel=2e-3)
    assert doc.trajectory_health.verdict == "stable"
    assert not doc.has_imaginary_modes
    assert np.max(doc.phonon_bandstructure.bands) == pytest.approx(6.554, rel=0.05)


def test_finite_temperature_phonon_maker_emt_born_charges(
    clean_dir, cu3au, fake_dielectric_calculator
):
    """Born charges from a force field dielectric job enter the NAC."""
    np.random.seed(103)  # noqa: NPY002
    maker = ForceFieldFiniteTemperaturePhononMaker.from_force_field_name(
        EMT,
        min_length=7.0,
        md_time_step=2.0,
        md_time=1.0,
        equilibration_time=0.2,
        n_snapshots=10,
    )
    maker.born_maker = ForceFieldDielectricMaker()
    flow = maker.make(cu3au)
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    doc = responses[flow.output.uuid][1].output

    # the fake calculator gives +2 to the first species of the sorted cell, Cu
    born = np.array(doc.born)
    assert born.shape == (4, 3, 3)
    assert np.all(np.diagonal(born[:3], axis1=1, axis2=2) > 0)
    assert np.all(np.diagonal(born[3:], axis1=1, axis2=2) < 0)
    assert np.array(doc.epsilon_static) == pytest.approx(4 * np.eye(3))
    assert doc.uuids.born_run_uuid is not None
    assert doc.phonon_bandstructure.has_nac
