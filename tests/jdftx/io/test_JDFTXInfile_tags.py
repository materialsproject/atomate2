from atomate2.jdftx.io.JEiter import JEiter
from pytest import approx
import pytest
from pymatgen.util.typing import PathLike
from pymatgen.core.units import Ha_to_eV
from atomate2.jdftx.io.JDFTXInfile_master_format import *


tag_ex = "fluid-anion"

get_tag_object(tag_ex)

