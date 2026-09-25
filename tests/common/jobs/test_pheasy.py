"""Tests for the anharmonic force constant fit of the pheasy workflow.

Forces come from ASE's EMT potential, so the jobs run end to end without any
DFT reference data.
"""

import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from jobflow import run_locally
from phonopy.file_IO import parse_FORCE_CONSTANTS
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

import atomate2.common.jobs.pheasy as pheasy_jobs
from atomate2.common.jobs.pheasy import (
    _check_lasso_alpha,
    _get_num_anharmonic_supercells,
    _run_harmonic_fit,
    generate_frequencies_eigenvectors,
    generate_phonon_displacements,
)
from atomate2.common.jobs.phonons import (
    _generate_phonon_object,
    _get_num_irreducible_fcs,
)

# fcs_cutoff_radius in Bohr. 8 Bohr (4.2 A) covers the first two neighbour
# shells of fcc Cu (2.55 and 3.61 A) and stays inside the 10.8 A supercell.
FCS_CUTOFF_RADIUS = [-1, 8, 8]

COMMON_KWARGS = {
    "supercell_matrix": [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
    "displacement": 0.01,
    "sym_reduce": True,
    "symprec": 1e-3,
    "use_symmetrized_structure": None,
    "kpath_scheme": "seekpath",
    "code": "vasp",
}

FIT_KWARGS = {
    "total_dft_energy": None,
    "static_run_job_dir": None,
    "static_run_uuid": None,
    "born_run_job_dir": None,
    "born_run_uuid": None,
    "optimization_run_job_dir": None,
    "optimization_run_uuid": None,
}


def _emt_displacement_data(structures: list[Structure]) -> dict:
    forces = []
    for structure in structures:
        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.calc = EMT()
        forces.append(atoms.get_forces().tolist())
    return {
        "forces": forces,
        "displaced_structures": structures,
        "dirs": [None] * len(structures),
        "uuids": [None] * len(structures),
    }


def _max_abs_fc(filename: Path, key: str) -> float:
    with h5py.File(filename) as f:
        return float(np.abs(f[key][:]).max())


def _cu_structure() -> Structure:
    return AseAtomsAdaptor.get_structure(bulk("Cu", "fcc", a=3.61, cubic=True))


@pytest.mark.parametrize("num_displaced_supercells", [0, 3])
def test_harmonic_and_anharmonic_split(tmp_dir, num_displaced_supercells):
    """The fit job must split the forces where the displacement job did.

    Before this was fixed, the fit job used one harmonic supercell fewer than
    the displacement job made, and it ignored num_displaced_supercells.
    """
    # a distorted Cu cell needs more than three finite displacements, so the
    # random-displacement path is used for the harmonic force constants
    structure = Structure(
        Lattice.orthorhombic(3.6, 3.7, 3.8),
        ["Cu"] * 4,
        [[0, 0, 0], [0.5, 0.5, 0.02], [0.5, 0.03, 0.5], [0.01, 0.5, 0.5]],
    )
    kwargs = {**COMMON_KWARGS, "supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]]}
    assert len(_generate_phonon_object(structure, **kwargs).displacements) > 3
    anhar_kwargs = {
        "cal_anhar_fcs": True,
        "fcs_cutoff_radius": [-1, 6, 6],
        "num_displaced_supercells": num_displaced_supercells,
    }

    job = generate_phonon_displacements(
        structure=structure,
        displacement_anhar=0.03,
        num_disp_anhar=20,
        **anhar_kwargs,
        **kwargs,
    )
    responses = run_locally(job, create_folders=True, ensure_success=True)
    displacements = responses[job.uuid][1].output
    # the harmonic set, 20 anharmonic supercells, then the undisplaced supercell
    num_har = len(displacements) - 20 - 1
    assert num_har == (num_displaced_supercells or 14)

    job = generate_frequencies_eigenvectors(
        structure=structure,
        displacement_data=_emt_displacement_data(displacements),
        **anhar_kwargs,
        **FIT_KWARGS,
        **kwargs,
    )
    run_locally(job, create_folders=True, ensure_success=True)

    (fit_dir,) = (path.parent for path in Path.cwd().glob("job_*/disp_matrix.npy"))
    harmonic = np.load(fit_dir / "disp_matrix.npy")
    anharmonic = np.load(fit_dir / "disp_matrix_anhar.npy")
    assert harmonic.shape[0] == num_har
    assert anharmonic.shape[0] == 20

    # the anharmonic set is drawn with its own seed, so its directions differ
    # from those of the random harmonic set
    assert not np.allclose(harmonic[0] / 0.01, anharmonic[0] / 0.03)


def test_get_num_anharmonic_supercells(monkeypatch):
    phonon = _generate_phonon_object(_cu_structure(), **COMMON_KWARGS)
    kwargs = {
        "supercell": phonon.supercell,
        "anhar_max_order": 3,
        "fcs_cutoff_radius": [-1, 12, 10],
        "anhar_fit_methods": ["one-shot"],
    }

    # a value set by the user is used as it is
    assert _get_num_anharmonic_supercells(num_disp_anhar=7, **kwargs) == 7

    # sized from the free force constants, above the floor of 20
    assert _get_num_irreducible_fcs(phonon.supercell, 3, [-1, 12]) == [25, 324]
    assert _get_num_anharmonic_supercells(num_disp_anhar=0, **kwargs) == 108

    # the first cutoff is not used, since pheasy fits fc2 without a cutoff
    fc2_cutoff = {**kwargs, "fcs_cutoff_radius": [5, 12, 10]}
    assert _get_num_anharmonic_supercells(num_disp_anhar=0, **fc2_cutoff) == 108

    # fourth order, with the three-body limit this workflow passes to pheasy
    order_4 = {**kwargs, "anhar_max_order": 4, "anhar_fit_methods": ["cocktail"]}
    assert _get_num_anharmonic_supercells(num_disp_anhar=0, **order_4) == 224

    # a binary compound, L1_2 Cu3Au, needs one cutoff per pair of elements
    cu3au = Structure(
        Lattice.cubic(3.75),
        ["Au", "Cu", "Cu", "Cu"],
        [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
    )
    binary = {
        **kwargs,
        "supercell": _generate_phonon_object(cu3au, **COMMON_KWARGS).supercell,
    }
    assert _get_num_anharmonic_supercells(num_disp_anhar=0, **binary) == 337

    # above the ceiling the job stops
    monkeypatch.setattr(pheasy_jobs, "_MAX_NUM_DISP_ANHAR", 30)
    with pytest.raises(ValueError, match="more than the limit of 30"):
        _get_num_anharmonic_supercells(num_disp_anhar=0, **kwargs)


def test_run_harmonic_fit(monkeypatch):
    """The fit flags of the pheasy workflow and the finite-temperature fit."""
    calls = []

    def fake_run(args, check):
        calls.append((args, check))

    monkeypatch.setattr(pheasy_jobs.subprocess, "run", fake_run)
    matrix = np.diag([2, 3, 4])

    _run_harmonic_fit(matrix, 1e-3, 5)
    assert [args[:4] for args, _ in calls] == [["pheasy", "--dim", "2", "3"]] * 4
    assert all(check for _, check in calls)
    assert [args[9] for args, _ in calls] == ["-s", "-c", "-d", "-f"]
    fit = " ".join(calls[-1][0])
    assert "-l LASSO --std --seed 103 --rasr BHH --ndata 5" in fit
    assert "--alpha_min" not in fit
    assert " -o " not in fit

    calls.clear()
    _run_harmonic_fit(matrix, 1e-3, 3, use_lasso=False)
    fit = " ".join(calls[-1][0])
    assert "-f --full_ifc --rasr BHH --ndata 3" in fit
    for flag in ("-l", "--std", "--seed"):
        assert flag not in calls[-1][0]

    calls.clear()
    _run_harmonic_fit(
        matrix, 1e-3, 5, rotational_sum_rule=None, alpha_min=-8, log_file="x.log"
    )
    fit = " ".join(calls[-1][0])
    assert "--alpha_min -8 --seed 103 --ndata 5" in fit
    assert fit.endswith("-o x.log")
    assert "--rasr" not in fit


def test_check_lasso_alpha(tmp_dir):
    log_file = Path("pheasy_anharmonic_fit.log")

    log_file.write_text("- alpha_min: 1e-12\n- alpha_opt: 2.947052e-09\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _check_lasso_alpha(log_file, alpha_min=-12) == pytest.approx(
            2.947052e-09
        )

    log_file.write_text("- alpha_min: 1e-12\n- alpha_opt: 1.000000e-12\n")
    with pytest.warns(UserWarning, match="on the lower bound.*anhar_alpha_min"):
        _check_lasso_alpha(log_file, alpha_min=-12)
    with pytest.warns(UserWarning, match="Lower alpha_min and refit"):
        _check_lasso_alpha(log_file, alpha_min=-12, alpha_min_name="alpha_min")

    log_file.write_text("- alpha_max: 1e-2\n- alpha_opt: 1.000000e-02\n")
    with pytest.warns(UserWarning, match="on the upper bound"):
        _check_lasso_alpha(log_file, alpha_min=-12)

    log_file.write_text("Fitting force constants via the ordinary least-square.\n")
    with pytest.raises(RuntimeError, match="No LASSO alpha"):
        _check_lasso_alpha(log_file, alpha_min=-12)

    log_file.unlink()
    with pytest.raises(FileNotFoundError, match="exited without writing"):
        _check_lasso_alpha(log_file, alpha_min=-12)


def test_anharmonic_fit_cocktail_and_one_shot(tmp_dir):
    structure = _cu_structure()
    anhar_kwargs = {
        "cal_anhar_fcs": True,
        "fcs_cutoff_radius": FCS_CUTOFF_RADIUS,
        "anhar_fit_methods": ["cocktail", "one-shot"],
    }

    job = generate_phonon_displacements(
        structure=structure,
        num_displaced_supercells=0,
        displacement_anhar=0.03,
        num_disp_anhar=0,
        **anhar_kwargs,
        **COMMON_KWARGS,
    )
    responses = run_locally(job, create_folders=True, ensure_success=True)
    displacements = responses[job.uuid][1].output
    # one finite displacement for fcc Cu, the anharmonic set at its floor of
    # 20, then the undisplaced supercell
    assert len(displacements) == 1 + 20 + 1

    job = generate_frequencies_eigenvectors(
        structure=structure,
        displacement_data=_emt_displacement_data(displacements),
        **anhar_kwargs,
        **FIT_KWARGS,
        **COMMON_KWARGS,
    )
    # a LASSO penalty on either bound of the search fails the test
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="The LASSO penalty")
        run_locally(job, create_folders=True, ensure_success=True)

    (fit_dir,) = (path.parent for path in Path.cwd().glob("job_*/fc3.hdf5"))
    one_shot_dir = fit_dir / "one_shot"

    # both fits used LASSO
    for folder in (fit_dir, one_shot_dir):
        log = (folder / "pheasy_anharmonic_fit.log").read_text()
        assert "coordinate descent LASSO" in log

    # the one-shot fc2 agrees with the harmonic fc2, which it does not overwrite
    harmonic_fc2 = np.abs(parse_FORCE_CONSTANTS(fit_dir / "FORCE_CONSTANTS")).max()
    assert _max_abs_fc(one_shot_dir / "fc2.hdf5", "fc2") == pytest.approx(
        harmonic_fc2, rel=0.05
    )

    # third-order force constants in eV/A^3. With 20 supercells, the one-shot
    # value changed by about 10 percent with the LASSO seed in local runs,
    # while the cocktail value did not, so only the cocktail value is pinned.
    cocktail_fc3 = _max_abs_fc(fit_dir / "fc3.hdf5", "fc3")
    assert cocktail_fc3 == pytest.approx(4.551, rel=0.02)
    assert _max_abs_fc(one_shot_dir / "fc3.hdf5", "fc3") == pytest.approx(
        cocktail_fc3, rel=0.15
    )


def test_anharmonic_fit_fourth_order(tmp_dir):
    """The fourth-order fit runs and writes fc4."""
    structure = _cu_structure()
    anhar_kwargs = {
        "cal_anhar_fcs": True,
        "fcs_cutoff_radius": FCS_CUTOFF_RADIUS,
        "anhar_max_order": 4,
        "anhar_fit_methods": ["cocktail"],
    }

    job = generate_phonon_displacements(
        structure=structure,
        num_displaced_supercells=0,
        # the workflow default. At 0.03 A, cross-validation chose the lowest
        # penalty, 1e-12.
        displacement_anhar=0.08,
        num_disp_anhar=20,
        **anhar_kwargs,
        **COMMON_KWARGS,
    )
    responses = run_locally(job, create_folders=True, ensure_success=True)
    displacements = responses[job.uuid][1].output

    job = generate_frequencies_eigenvectors(
        structure=structure,
        displacement_data=_emt_displacement_data(displacements),
        **anhar_kwargs,
        **FIT_KWARGS,
        **COMMON_KWARGS,
    )
    # a LASSO penalty on either bound of the search fails the test
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="The LASSO penalty")
        run_locally(job, create_folders=True, ensure_success=True)

    (fit_dir,) = (path.parent for path in Path.cwd().glob("job_*/fc4.hdf5"))
    assert (fit_dir / "FORCE_CONSTANTS_4TH").exists()
    assert (
        "coordinate descent LASSO"
        in (fit_dir / "pheasy_anharmonic_fit.log").read_text()
    )

    # force constants in eV/A^3 and eV/A^4
    assert _max_abs_fc(fit_dir / "fc3.hdf5", "fc3") == pytest.approx(4.601, rel=0.02)
    assert _max_abs_fc(fit_dir / "fc4.hdf5", "fc4") == pytest.approx(48.92, rel=0.05)
