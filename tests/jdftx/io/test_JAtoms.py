from pytest import approx
import pytest
from pymatgen.util.typing import PathLike
from pymatgen.core.units import Ha_to_eV
from atomate2.jdftx.io.JAtoms import JEiter, JEiters, JAtoms
from pathlib import Path
from pymatgen.util.typing import PathLike
from pymatgen.core.units import Ha_to_eV, bohr_to_ang
import numpy as np


ex_files_dir = Path(__file__).parents[0] / "example_files"
ex_slice_fname1 = ex_files_dir / "ex_text_slice_forJAtoms_latmin"
ex_slice1 = []
with open(ex_slice_fname1, "r") as f:
    for line in f:
        ex_slice1.append(line)
ex_slice1_known = {
    "mu0": 0.713855355*Ha_to_eV,
    "mu-1": +0.703866408*Ha_to_eV,
    "E0": -246.455370884127575*Ha_to_eV,
    "E-1": -246.531007900240667*Ha_to_eV,
    "nEminSteps": 18,
    "EconvReason": "|Delta F|<1.000000e-07 for 2 iters",
    "cell_00": 6.16844*bohr_to_ang,
    "strain_00": 10.0,
    "stress_00": -1.69853e-06,
    "nAtoms": 8,
    "posn0": np.array(0.000011000000000,2.394209000000000,1.474913000000000)*bohr_to_ang,
    "force0": np.array(0.000003219385226,0.000024941936105,-0.000004667309539)*Ha_to_eV/bohr_to_ang
}

@pytest.mark.parametrize("eslice,eknowns",
                         [(ex_slice1, ex_slice1_known)
                           ])
def test_JAtoms(eslice: list[str], eknowns: dict):
    jat = JAtoms.from_text_slice(eslice)
    assert jat.elecMinData[0].mu == approx(eknowns["mu0"])



