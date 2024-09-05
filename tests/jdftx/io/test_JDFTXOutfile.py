from pathlib import Path
from atomate2.jdftx.io.JDFTXOutfile import JDFTXOutfile
from pytest import approx
import pytest
from pymatgen.util.typing import PathLike
from pymatgen.core.units import Ha_to_eV

ex_files_dir = Path(__file__).parents[0] / "example_files"
example_sp_known = {
    "Nspin": 1,
    "spintype": None,
    "broadening_type": "MP1",
    "broadening": 0.00367493*Ha_to_eV,
    "truncation_type": "slab",
    "pwcut": 30*Ha_to_eV,
    "fftgrid": (54, 54, 224),
    "kgrid": (6, 6, 1),
    "Emin": -3.836283*Ha_to_eV,
    "HOMO": -0.212435*Ha_to_eV,
    "EFermi": -0.209509*Ha_to_eV,
    "LUMO": -0.209424*Ha_to_eV,
    "Emax": 0.113409*Ha_to_eV,
    "Egap": 0.003011*Ha_to_eV,
    "is_metal": True,
    "fluid": None,
    "total_electrons": 288.0,
    "Nbands": 288,
    "Nat": 16,
    "F": -1940.762261217305650*Ha_to_eV,
    "TS": -0.0001776512106456*Ha_to_eV,
    "Etot": -1940.7624388685162558*Ha_to_eV,
    "KE": 593.1822417205943339*Ha_to_eV,
    "Exc": -185.5577583222759870*Ha_to_eV,
    "Epulay": 0.0000125227478554*Ha_to_eV,
    "Enl": 174.1667582919756114*Ha_to_eV,
    "Eloc": 29663.3545152997867262*Ha_to_eV,
    "EH": -15284.4385436602351547*Ha_to_eV,
    "Eewald": -16901.4696647211094387*Ha_to_eV,
}

@pytest.mark.parametrize("filename,known", [(ex_files_dir / Path("example_sp.out"),
                                             example_sp_known)]
                                             )
def test_JDFTXOutfile_fromfile(
    filename: PathLike,
    known: dict
    ):
    # filename = ex_files_dir / Path("jdftx.out")
    jout = JDFTXOutfile.from_file(filename)
    assert jout.Nspin == known["Nspin"]
    assert jout.spintype is known["spintype"]
    assert jout.broadening_type == known["broadening_type"]
    assert jout.broadening == approx(known["broadening"])
    assert jout.truncation_type == known["truncation_type"]
    assert jout.pwcut == approx(known["pwcut"])
    # Don't bully me, I'm testing this way incase we flip-flop between lists and tuples
    for i in range(3):
        assert jout.fftgrid[i] == known["fftgrid"][i]
    for i in range(3):
        assert jout.kgrid[i] == known["kgrid"][i]
    assert jout.Emin == approx(known["Emin"])
    assert jout.HOMO == approx(known["HOMO"])
    assert jout.EFermi == approx(known["EFermi"])
    assert jout.LUMO == approx(known["LUMO"])
    assert jout.Emax == approx(known["Emax"])
    assert jout.Egap == approx(known["Egap"])
    # TODO: filling tests
    # assert jout.HOMO_filling == approx(None)
    # assert jout.LUMO_filling == approx(None)
    assert jout.is_metal == known["is_metal"]
    assert jout.fluid == known["fluid"]
    #
    assert jout.total_electrons == approx(known["total_electrons"])
    assert jout.Nbands == known["Nbands"]
    #
    assert jout.Nat == known["Nat"]
    for listlike in (
        jout.atom_coords, jout.atom_coords_final, jout.atom_coords_initial,
        jout.atom_elements, jout.atom_elements_int
        ):
        assert len(listlike) == known["Nat"]
    assert jout.Ecomponents["F"] == approx(known["F"])
    assert jout.Ecomponents["TS"] == approx(known["TS"])
    assert jout.Ecomponents["Etot"] == approx(known["Etot"])
    assert jout.Ecomponents["KE"] == approx(known["KE"])
    assert jout.Ecomponents["Exc"] == approx(known["Exc"])
    assert jout.Ecomponents["Epulay"] == approx(known["Epulay"])
    assert jout.Ecomponents["Enl"] == approx(known["Enl"])
    assert jout.Ecomponents["Eloc"] == approx(known["Eloc"])
    assert jout.Ecomponents["EH"] == approx(known["EH"])
    assert jout.Ecomponents["Eewald"] == approx(known["Eewald"])


test_JDFTXOutfile_fromfile(ex_files_dir / Path("example_sp.out"), example_sp_known)
