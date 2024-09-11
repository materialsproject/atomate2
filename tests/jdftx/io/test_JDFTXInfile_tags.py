from atomate2.jdftx.io.JEiter import JEiter
from pytest import approx
# import pytest
from pymatgen.util.typing import PathLike
from pymatgen.core.units import Ha_to_eV
from atomate2.jdftx.io.JDFTXInfile_master_format import *
from atomate2.jdftx.io.JDFTXInfile import JDFTXInfile
from pathlib import Path
import os

infile = Path(os.getcwd()) / "tests" / "jdftx" / "io" / "example_files" / "example_sp.in"
testwrite = Path(os.getcwd()) / "tests" / "jdftx" / "io" / "example_files" / "example_sp_copy.in"
jif = JDFTXInfile.from_file(infile)
jif.write_file(testwrite)
jiflist = jif.get_text_list()
tag_ex = "fluid-anion"

get_tag_object(tag_ex)

