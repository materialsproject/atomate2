# flake8: noqa
import numpy as np
import pytest

from ase.io import read, ParseError
from ase.stress import full_3x3_to_voigt_6_stress
from numpy.linalg import norm
from pathlib import Path

parent = Path(__file__).parents[1]


def test_parse_socketio(tmpdir):
    traj = read(parent / "test_data/aims_out/socket.out", ":", format="aims-output")
    assert len(traj) == 6
    p0 = [[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.24, 0.9279, 0.0]]
    p1 = [
        [-0.00044436, 0.00021651, 0.00068957],
        [0.96112981, -0.00029923, 0.00096836],
        [-0.24091781, 0.93010946, 0.00061317],
    ]
    p_end = [
        [-0.00156048, -0.00072446, 0.00045281],
        [0.98615072, -0.00962614, -0.00053732],
        [-0.25646779, 0.95117586, 0.00820183],
    ]
    assert np.allclose(traj[0].get_positions(), p0)
    assert np.allclose(traj[1].get_positions(), p1)
    assert np.allclose(traj[-1].get_positions(), p_end)

    f0 = [
        [-0.481289284665163e00, -0.615051370384412e00, 0.811297123282653e-27],
        [0.762033585727896e00, -0.942008578636939e-01, -0.973556547939183e-27],
        [-0.280744301062733e00, 0.709252228248106e00, -0.649037698626122e-27],
    ]
    f1 = [
        [-0.346210275412861e00, -0.520615919604426e00, -0.966369462150621e-04],
        [0.587866333819113e00, -0.830442530429637e-01, 0.171037714240380e-03],
        [-0.241656058406252e00, 0.603660172647390e00, -0.744007680253175e-04],
    ]
    f_end = [
        [0.492882061544499e00, 0.499117230159087e00, 0.347959116743205e-02],
        [-0.724281788245024e00, 0.800633239635954e-01, 0.130633777464187e-02],
        [0.231399726700525e00, -0.579180554122683e00, -0.478592894207392e-02],
    ]
    assert np.allclose(traj[0].get_forces(), f0)
    assert np.allclose(traj[1].get_forces(), f1)
    assert np.allclose(traj[-1].get_forces(), f_end)


def test_parse_md(tmpdir):
    traj = read(parent / "test_data/aims_out/md.out", ":", format="aims-output")
    assert len(traj) == 5
    p0 = [[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.24, 0.9279, 0.0]]
    p1 = [
        [0.00247722, -0.00200215, 0.00000000],
        [0.93156204, -0.00330135, 0.00000000],
        [-0.25248383, 0.96298223, 0.00000000],
    ]
    p_end = [
        [-0.00044308, -0.00190646, 0.00000000],
        [0.98936333, -0.01341746, -0.00000000],
        [-0.26393022, 0.97157934, 0.00000000],
    ]
    assert np.allclose(traj[0].get_positions(), p0)
    assert np.allclose(traj[1].get_positions(), p1)
    assert np.allclose(traj[-1].get_positions(), p_end)

    f0 = [
        [-0.481289284665163e00, -0.615051370384412e00, 0.811297123282653e-27],
        [0.762033585727896e00, -0.942008578636939e-01, -0.973556547939183e-27],
        [-0.280744301062733e00, 0.709252228248106e00, -0.649037698626122e-27],
    ]
    f1 = [
        [-0.284519402890037e01, 0.121286349030924e01, 0.691733365155783e-17],
        [0.257911758656866e01, -0.471469245294899e-01, -0.730166238143266e-18],
        [0.266076442331705e00, -0.116571656577975e01, -0.618716741081841e-17],
    ]
    f_end = [
        [0.266848800470869e00, 0.137113486710510e01, 0.717235335588107e-17],
        [-0.812308728045051e00, 0.142880785873554e00, 0.874856850626564e-18],
        [0.545459927574182e00, -0.151401565297866e01, -0.804721020748119e-17],
    ]
    assert np.allclose(traj[0].get_forces(), f0)
    assert np.allclose(traj[1].get_forces(), f1)
    assert np.allclose(traj[-1].get_forces(), f_end)


def test_parse_relax(tmpdir):
    traj = read(parent / "test_data/aims_out/relax.out", ":", format="aims-output")
    assert len(traj) == 8
    p0 = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
    assert all([np.allclose(at.get_scaled_positions(), p0) for at in traj])
    assert all([np.allclose(at.get_forces(), np.zeros((2, 3))) for at in traj])

    s0 = full_3x3_to_voigt_6_stress(
        [
            [0.07748922, 0.0, 0.0],
            [0.0, 0.07748923, 0.0],
            [0.0, 0.0, 0.07748923],
        ]
    )
    s_end = full_3x3_to_voigt_6_stress(
        [
            [-0.00007093, 0.00001488, 0.00001488],
            [0.00001488, -0.00007093, 0.00001488],
            [0.00001488, 0.00001488, -0.00007093],
        ]
    )
    assert np.allclose(traj[0].get_stress(), s0)
    assert np.allclose(traj[-1].get_stress(), s_end)

    cell_0 = [
        [0.000, 2.903, 2.903],
        [2.903, 0.000, 2.903],
        [2.903, 2.903, 0.000],
    ]
    cell_end = [
        [0.00006390, 2.70950938, 2.70950938],
        [2.70950941, 0.00006393, 2.70950939],
        [2.70950941, 2.70950939, 0.00006393],
    ]
    assert np.allclose(traj[0].get_cell(), cell_0)
    assert np.allclose(traj[-1].get_cell(), cell_end)


def test_parse_singlepoint(tmpdir):
    atoms = read(parent / "test_data/aims_out/singlepoint.out", format="aims-output")
    p0 = [[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.24, 0.9279, 0.0]]
    assert np.allclose(atoms.get_positions(), p0)

    f0 = [
        [-0.478299830661005e01, -0.616960341437854e01, 0.162259424656531e-27],
        [0.692109878170016e01, -0.419925410034428e00, 0.000000000000000e00],
        [-0.213810047509011e01, 0.658952882441297e01, 0.243389136984796e-27],
    ]
    assert np.allclose(atoms.get_forces(), f0)


def test_parse_dfpt_dielectric(tmpdir):
    outfile = parent / "test_data/aims_out/DFPT_dielectric.out"
    atoms = read(outfile, format="aims-output")

    diel = atoms.calc.results["dielectric_tensor"]

    diel_0 = [
        [7.18759265e00, -1.0000000e-15, 1.9000000e-14],
        [-1.000000e-15, 7.18759284e00, 2.59000000e-13],
        [2.0000000e-14, 2.58000000e-13, 7.1875928e00],
    ]

    assert np.allclose(diel, diel_0)

def test_parse_polarization(tmpdir):
    outfile = parent / "test_data/aims_out/polarization.out"
    atoms = read(outfile, format="aims-output")

    polar = atoms.calc.results["polarization"]

    polar_0 = [-51.045557E-03, -51.045557E-03, -51.458008E-03]

    assert np.allclose(polar, polar_0)

def test_preamble_failed(tmpdir):
    outfile = parent / "test_data/aims_out/preamble_fail.out"
    with pytest.raises(ParseError, match='No SCF steps'):
        read(outfile, format="aims-output")

def test_numerical_stress(tmpdir):
    outfile = parent / "test_data/aims_out/numerical_stress.out"

    atoms = read(outfile, format="aims-output")
    stress = atoms.get_stress()
    stress_actual = [
        0.00244726, 0.00267442, 0.00258710, 0.00000005, -0.00000026, -0.00000007
    ]

    assert np.allclose(stress, stress_actual)
