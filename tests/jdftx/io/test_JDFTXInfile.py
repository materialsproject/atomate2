from pathlib import Path
from atomate2.jdftx.io.JDFTXOutfile import JDFTXOutfile
from atomate2.jdftx.io.JDFTXInfile import JDFTXInfile
from pytest import approx
import pytest
from pymatgen.util.typing import PathLike
from pymatgen.core.units import Ha_to_eV
import os

ex_files_dir = Path(__file__).parents[0] / "example_files"

ex_infile1_fname = ex_files_dir / "CO.in"
jif = JDFTXInfile.from_file(ex_infile1_fname)
out = jif.get_list_representation(jif)
jif.get_text_list()


@pytest.mark.parametrize("infile_fname", [ex_infile1_fname])
def test_JDFTXInfile_self_consistency(infile_fname: PathLike):
    jif = JDFTXInfile.from_file(infile_fname)
    dict_jif = jif.as_dict()
    jif2 = JDFTXInfile.from_dict(dict_jif)
    # Removing this requirement for now
    str_jif = str(jif)
    with open(ex_files_dir / "str_jif", "w") as f:
        f.write(str_jif)
    with open(ex_files_dir / "str_jif2", "w") as f:
        f.write(rf'{str_jif}')
    str_dict_jif = str(dict_jif)
    with open(ex_files_dir / "str_dict_jif", "w") as f:
        f.write(str_dict_jif)
    jif3 = JDFTXInfile.from_dict(dict(str(jif)))
    jif3 = JDFTXInfile.from_str(str(jif))
    tmp_fname = ex_files_dir / "tmp.in"
    jif.write_file(tmp_fname)
    jif4 = JDFTXInfile.from_file(tmp_fname)
    jifs = [jif, jif2, jif4]
    for i in range(len(jifs)):
        for j in range(i+1, len(jifs)):
            assert is_identical_jif(jifs[i], jifs[j])
    return None

def is_identical_jif(jif1: JDFTXInfile, jif2: JDFTXInfile):
    for key in jif1:
        if key not in jif2:
            return False
        else:
            v1 = jif1[key]
            v2 = jif2[key]
    return True


def is_identical_jif_val(v1, v2):
    if not type(v1) == type(v2):
        return False
    else:
        if isinstance(v1, float):
            return v1 == approx(v2)
        elif True in [isinstance(v1, str), isinstance(v1, int)]:
            return v1 == v2
        elif True in [isinstance(v1, list)]:
            if not len(v1) == len(v2):
                return False
            else:
                for i, v in enumerate(v1):
                    if not is_identical_jif_val(v, v2[i]):
                        return False
                return True

test_JDFTXInfile_self_consistency(ex_infile1_fname)
        
        