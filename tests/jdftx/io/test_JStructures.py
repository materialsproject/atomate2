from pytest import approx
import pytest
from pymatgen.core.units import Ha_to_eV
from atomate2.jdftx.io.JStructure import JStructures
from pathlib import Path
from pymatgen.util.typing import PathLike
from pymatgen.core.units import Ha_to_eV, bohr_to_ang
import numpy as np

ex_files_dir = Path(__file__).parents[0] / "example_files"
ex_outslice_fname1 = ex_files_dir / "ex_out_slice_latmin"
ex_outslice1 = []
with open(ex_outslice_fname1, "r") as f:
    for line in f:
        ex_outslice1.append(line)
ex_outslice1_known = {
    "mu0_0": 0.713855355*Ha_to_eV,
    "mu0_-1": 0.703866408*Ha_to_eV,
    "nEminSteps0": 18,
    "etype0": "F",
    "E0": -246.531007900240667*Ha_to_eV,
    "mu-1_0": 0.704400512*Ha_to_eV,
    "mu-1_-1": 0.704399109*Ha_to_eV,
    "nEminSteps-1": 4,
    "etype-1": "F",
    "E-1": -246.531042396724303*Ha_to_eV,
    "nGeomSteps": 7,
}

@pytest.mark.parametrize("ex_slice, ex_slice_known", [(ex_outslice1, ex_outslice1_known)])
def test_JStructures(ex_slice: list[str], ex_slice_known: dict[str, float]):
    jstruct = JStructures.from_out_slice(ex_slice, iter_type="lattice")
    assert jstruct[0].elecMinData[0].mu == approx(ex_slice_known["mu0_0"])
    assert jstruct[0].elecMinData[-1].mu == approx(ex_slice_known["mu0_-1"])
    assert jstruct[-1].elecMinData[0].mu == approx(ex_slice_known["mu-1_0"])
    assert jstruct[-1].elecMinData[-1].mu == approx(ex_slice_known["mu-1_-1"])
    assert len(jstruct[0].elecMinData) == ex_slice_known["nEminSteps0"]
    assert len(jstruct[-1].elecMinData) == ex_slice_known["nEminSteps-1"]
    assert jstruct[0].etype == ex_slice_known["etype0"]
    assert jstruct[0].E == approx(ex_slice_known["E0"])
    assert jstruct[-1].etype == ex_slice_known["etype-1"]
    assert jstruct[-1].E == approx(ex_slice_known["E-1"])
    assert len(jstruct) == ex_slice_known["nGeomSteps"]