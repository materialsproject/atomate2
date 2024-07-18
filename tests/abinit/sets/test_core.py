import os

import pytest
from abipy.abio.inputs import AbinitInput

from atomate2.abinit.sets.base import AbinitInputSet
from atomate2.abinit.sets.core import StaticSetGenerator


def test_init_static_generator():
    StaticSetGenerator()


def test_static_generator_get_abinit_input(si_structure, abinit_test_dir):
    ssg = StaticSetGenerator()
    si_pseudo = os.path.join(abinit_test_dir, "pseudos", "14si.fhi")
    abinit_input = ssg.get_abinit_input(structure=si_structure, pseudos=si_pseudo)
    assert isinstance(abinit_input, AbinitInput)
    with pytest.raises(RuntimeError, match=r"Structure is mandatory.*"):
        ssg.get_abinit_input()

    with pytest.raises(
        RuntimeError,
        match=r"Previous outputs not allowed.*",
    ):
        ssg.get_abinit_input(structure=si_structure, prev_outputs="prev_fake_dir")


def test_static_generator_get_input_set(si_structure):
    ssg = StaticSetGenerator()
    input_set = ssg.get_input_set(structure=si_structure)
    assert isinstance(input_set, AbinitInputSet)


# def test_static_generator_restart(si_structure):
#     ssg1 = StaticSetGenerator(ecut=8.5, nband=13, smearing=None)
#     ssg2 = StaticSetGenerator(nband=8, smearing=None)
#     ssg3 = StaticSetGenerator()
#
#     with ScratchDir(".") as tmpdir:
#         input_set1 = ssg1.get_input_set(
#             structure=si_structure, smearing="marzari5:0.5 eV"
#         )
#         assert isinstance(input_set1, AbinitInputSet)
#         assert input_set1.abinit_input["occopt"] == 5
#         input_set1.write_input("first_run")
#         dirlist = os.listdir("first_run")
#         assert len(dirlist) == 6
#         assert "abinit_input_set_generator.json" in dirlist
#         assert "abinit_input.json" in dirlist
#         with open("first_run/run.abi") as file:
#             runabi1_str = file.read()
#         assert "ecut 8.5" in runabi1_str
#         assert "nband 13" in runabi1_str
#         assert "occopt 5" in runabi1_str
#         matched_tsmear_lines = [
#             line for line in runabi1_str.split("\n") if "tsmear" in line
#         ]
#         assert len(matched_tsmear_lines) == 1
#         _, tsmear, tsmear_unit = matched_tsmear_lines[0].split()
#         assert float(tsmear) == pytest.approx(0.0183746610878275)
#         assert tsmear_unit == "Ha"
#         outdenpath = os.path.join(
#             tmpdir, "first_run", OUTDIR_NAME, f"{OUTDATAFILE_PREFIX}_DEN"
#         )
#         Path(outdenpath).touch()
#
#         input_set2 = ssg2.get_input_set(
#             structure=si_structure,
#             charge=1.0,
#             restart_from=os.path.join(tmpdir, "first_run"),
#         )
#         input_set2.write_input("second_run")
#         indenpath = os.path.join(
#             tmpdir, "second_run", INDIR_NAME, f"{INDATAFILE_PREFIX}_DEN"
#         )
#         assert os.path.islink(indenpath)
#         assert os.readlink(indenpath) == outdenpath
#         with open("second_run/run.abi") as file:
#             runabi2_str = file.read()
#         assert "ecut 8.5" in runabi2_str
#         assert "nband 8" in runabi2_str
#         assert "occopt" not in runabi2_str
#         assert "tsmear" not in runabi2_str
#         assert "charge 1.0" in runabi2_str
#         assert "irdden 1" in runabi2_str
#
#         outwfkpath = os.path.join(
#             tmpdir, "first_run", OUTDIR_NAME, f"{OUTDATAFILE_PREFIX}_WFK"
#         )
#         Path(outwfkpath).touch()
#         input_set3 = ssg3.get_input_set(
#             structure=si_structure, restart_from=os.path.join(tmpdir, "first_run")
#         )
#         input_set3.write_input("second_run_bis")
#         indenpath = os.path.join(
#             tmpdir, "second_run_bis", INDIR_NAME, f"{INDATAFILE_PREFIX}_DEN"
#         )
#         inwfkpath = os.path.join(
#             tmpdir, "second_run_bis", INDIR_NAME, f"{INDATAFILE_PREFIX}_WFK"
#         )
#         assert os.path.islink(inwfkpath)
#         assert os.readlink(inwfkpath) == outwfkpath
#         with open("second_run_bis/run.abi") as file:
#             runabi3_str = file.read()
#         assert "ecut 8.5" in runabi3_str
#         assert "nband 13" in runabi3_str
#         assert "occopt 5" in runabi3_str
#         matched_tsmear_lines = [
#             line for line in runabi3_str.split("\n") if "tsmear" in line
#         ]
#         assert len(matched_tsmear_lines) == 1
#         _, tsmear, tsmear_unit = matched_tsmear_lines[0].split()
#         assert float(tsmear) == pytest.approx(0.0183746610878275)
#         assert tsmear_unit == "Ha"
#         outdenpath = os.path.join(
#             tmpdir, "first_run", OUTDIR_NAME, f"{OUTDATAFILE_PREFIX}_DEN"
#         )
