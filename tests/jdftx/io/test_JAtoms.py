from pytest import approx
import pytest
from pymatgen.util.typing import PathLike
from pymatgen.core.units import Ha_to_eV

ex_fillings_line1 = "FillingsUpdate:  mu: +0.714406772  nElectrons: 64.000000  magneticMoment: [ Abs: 0.00578  Tot: -0.00141 ]"
ex_fillings_line1_known = {
    "mu": 0.714406772*Ha_to_eV,
    "nElectrons": 64.0,
    "abs_magneticMoment": 0.00578,
    "tot_magneticMoment": -0.00141
}


def test_JEiter()