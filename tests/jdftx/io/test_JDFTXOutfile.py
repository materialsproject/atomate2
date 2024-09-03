from pathlib import Path
from atomate2.jdftx.io.JDFTXOutfile import JDFTXOutfile
from pytest import approx

ex_files_dir = Path(__file__).parents[0]

def test_JDFTXOutfile():
    filename = ex_files_dir / Path("jdftx.out")
    jout = JDFTXOutfile.from_file(filename)
    assert jout.Ecomponents["F"] == approx(-1940.762261217305650)
