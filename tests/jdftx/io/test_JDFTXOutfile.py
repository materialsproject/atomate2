from pathlib import Path
from atomate2.jdftx.io.JDFTXOutfile import JDFTXOutfile, HA2EV
from pytest import approx

ex_files_dir = Path(__file__).parents[0] / "example_files"

def test_JDFTXOutfile():
    filename = ex_files_dir / Path("jdftx.out")
    print(filename)
    jout = JDFTXOutfile.from_file(filename)
    assert jout.Ecomponents["F"] == approx(-1940.762261217305650*HA2EV)

# test_JDFTXOutfile()
