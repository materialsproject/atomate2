from jdftx.io.JEiter import JEiter
from pytest import approx
import pytest
from pymatgen.util.typing import PathLike
from pymatgen.core.units import Ha_to_eV
from jdftx.io.JEiters import JEiters

ex_fillings_line1 = "FillingsUpdate:  mu: +0.714406772  nElectrons: 64.000000  magneticMoment: [ Abs: 0.00578  Tot: -0.00141 ]"
ex_fillings_line1_known = {
    "mu": 0.714406772*Ha_to_eV,
    "nElectrons": 64.0,
    "abs_magneticMoment": 0.00578,
    "tot_magneticMoment": -0.00141
}

ex_fillings_line2 = "FillingsUpdate:  mu: +0.814406772  nElectrons: 60.000000  magneticMoment: [ Abs: 0.0578  Tot: -0.0141 ]"
ex_fillings_line2_known = {
    "mu": 0.814406772*Ha_to_eV,
    "nElectrons": 60.0,
    "abs_magneticMoment": 0.0578,
    "tot_magneticMoment": -0.0141
}

ex_subspace_line1 = "SubspaceRotationAdjust: set factor to 0.229"
ex_subspace_line1_known = {
    "subspace": 0.229
}

ex_subspace_line2 = "SubspaceRotationAdjust: set factor to 0.329"
ex_subspace_line2_known = {
    "subspace": 0.329
}

ex_iter_line1 = "ElecMinimize: Iter:   6  F: -246.531038317370076  |grad|_K:  6.157e-08  alpha:  5.534e-01  linmin: -4.478e-06  t[s]:    248.68"
ex_iter_line1_known = {
    "iter": 6,
    "E": -246.531038317370076*Ha_to_eV,
    "grad_K": 6.157e-08,
    "alpha": 5.534e-01,
    "linmin": -4.478e-06,
    "t_s": 248.68
}

ex_iter_line2 = "ElecMinimize: Iter:   7  F: -240.531038317370076  |grad|_K:  6.157e-07  alpha:  5.534e-02  linmin: -5.478e-06  t[s]:    48.68"
ex_iter_line2_known = {
    "iter": 7,
    "E": -240.531038317370076*Ha_to_eV,
    "grad_K": 6.157e-07,
    "alpha": 5.534e-02,
    "linmin": -5.478e-06,
    "t_s": 48.68
}


ex_lines1 = [
    ex_fillings_line1, ex_subspace_line1, ex_iter_line1
]
ex_lines2 = [
    ex_fillings_line2, ex_subspace_line2, ex_iter_line2
]
ex_known1 = {
    "iter": ex_iter_line1_known,
    "fill": ex_fillings_line1_known,
    "subspace": ex_subspace_line1_known,
}

ex_known2 = {
    "iter": ex_iter_line2_known,
    "fill": ex_fillings_line2_known,
    "subspace": ex_subspace_line2_known,
}


@pytest.mark.parametrize("exfill_line,exfill_known,exiter_line,exiter_known,exsubspace_line,exsubspace_known",
                         [(ex_fillings_line1, ex_fillings_line1_known,
                           ex_iter_line1, ex_iter_line1_known,
                           ex_subspace_line1, ex_subspace_line1_known)
                           ])
def test_JEiter(exfill_line: str, exfill_known: dict[str, float],
                exiter_line: str, exiter_known: dict[str, float],
                exsubspace_line: str, exsubspace_known: dict[str, float],
                etype: str = "F", eitertype = "ElecMinimize"):
    ex_lines_collect = [exiter_line, exfill_line, exsubspace_line]
    jei = JEiter._from_lines_collect(ex_lines_collect, eitertype, etype)
    assert exfill_known["mu"] == approx(jei.mu)
    assert exfill_known["nElectrons"] == approx(jei.nElectrons)
    assert exfill_known["abs_magneticMoment"] == approx(jei.abs_magneticMoment)
    assert exfill_known["tot_magneticMoment"] == approx(jei.tot_magneticMoment)
    #
    assert exiter_known["iter"] == jei.iter
    assert exiter_known["E"] == approx(jei.E)
    assert exiter_known["grad_K"] == approx(jei.grad_K)
    assert exiter_known["alpha"] == approx(jei.alpha)
    assert exiter_known["linmin"] == approx(jei.linmin)
    assert exiter_known["t_s"] == approx(jei.t_s)
    #
    assert exsubspace_known["subspace"] == approx(jei.subspaceRotationAdjust)

@pytest.mark.parametrize("ex_lines,ex_knowns",
                         [([ex_lines1, ex_lines2], [ex_known1, ex_known2])
                           ])
def test_JEiters(ex_lines: list[list[str]], ex_knowns: list[dict],
                 etype: str = "F", eitertype = "ElecMinimize"):
    text_slice = []
    for exl in ex_lines:
        text_slice += exl
    jeis = JEiters.from_text_slice(text_slice, iter_type=eitertype, etype=etype)
    for i in range(len(ex_lines)):
        assert ex_knowns[i]["fill"]["mu"] == approx(jeis[i].mu)
        assert ex_knowns[i]["fill"]["nElectrons"] == approx(jeis[i].nElectrons)
        assert ex_knowns[i]["fill"]["abs_magneticMoment"] == approx(jeis[i].abs_magneticMoment)
        assert ex_knowns[i]["fill"]["tot_magneticMoment"] == approx(jeis[i].tot_magneticMoment)
        #
        assert ex_knowns[i]["iter"]["iter"] == jeis[i].iter
        assert ex_knowns[i]["iter"]["E"] == approx(jeis[i].E)
        assert ex_knowns[i]["iter"]["grad_K"] == approx(jeis[i].grad_K)
        assert ex_knowns[i]["iter"]["alpha"] == approx(jeis[i].alpha)
        assert ex_knowns[i]["iter"]["linmin"] == approx(jeis[i].linmin)
        assert ex_knowns[i]["iter"]["t_s"] == approx(jeis[i].t_s)
        #
        assert ex_knowns[i]["subspace"]["subspace"] == approx(jeis[i].subspaceRotationAdjust)

