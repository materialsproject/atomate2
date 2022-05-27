import os
from pathlib import Path

import pytest
from abipy.abio.inputs import AbinitInput
from monty.tempfile import ScratchDir

from atomate2.abinit.sets.base import AbinitInputSet
from atomate2.abinit.sets.core import StaticSetGenerator
from atomate2.abinit.utils.common import (
    INDATAFILE_PREFIX,
    INDIR_NAME,
    OUTDATAFILE_PREFIX,
    OUTDIR_NAME,
)


class TestStaticSetGenerator:
    def test_init(self):
        ssg = StaticSetGenerator()
        for param in ssg.params:
            assert not ssg.param_is_explicitly_set(param)

        ssg = StaticSetGenerator(ecut=5.0)
        for param in ssg.params:
            if param == "ecut":
                assert ssg.param_is_explicitly_set(param)
            else:
                assert not ssg.param_is_explicitly_set(param)

        ssg.pawecutdg = None
        for param in ssg.params:
            if param in ["ecut", "pawecutdg"]:
                assert ssg.param_is_explicitly_set(param)
            else:
                assert not ssg.param_is_explicitly_set(param)

    def test_on_restart(self, si_structure):
        abinit_input = AbinitInput(si_structure, pseudos=StaticSetGenerator.pseudos)
        assert "ntime" not in abinit_input
        assert "ionmov" not in abinit_input
        assert "optcell" not in abinit_input
        abinit_input_mov = abinit_input.new_with_vars(ntime=10, ionmov=2, optcell=2)
        assert "ntime" in abinit_input_mov
        assert "ionmov" in abinit_input_mov
        assert "optcell" in abinit_input_mov
        ssg = StaticSetGenerator()
        ssg.on_restart(abinit_input=abinit_input_mov)
        assert not "ntime" in abinit_input_mov
        assert not "ionmov" in abinit_input_mov
        assert not "optcell" in abinit_input_mov

    def test_get_abinit_input(self, si_structure):
        ssg = StaticSetGenerator()
        abinit_input = ssg.get_abinit_input(structure=si_structure, pseudos=ssg.pseudos)
        assert isinstance(abinit_input, AbinitInput)
        with pytest.raises(
            RuntimeError, match=r"Structure is mandatory for StaticSet generation."
        ):
            ssg.get_abinit_input()

        with pytest.raises(
            RuntimeError,
            match=r"Previous outputs not allowed for StaticSetGenerator. "
            r"To restart from a previous static or otherwise scf "
            r"\(e.g. relaxation\) calculation, use restart_from argument of "
            r"get_input_set method instead.",
        ):
            ssg.get_abinit_input(structure=si_structure, prev_outputs="prev_fake_dir")

    def test_get_input_set(self, si_structure):
        ssg = StaticSetGenerator()
        input_set = ssg.get_input_set(structure=si_structure)
        assert isinstance(input_set, AbinitInputSet)

    def test_restart(self, si_structure):
        ssg1 = StaticSetGenerator(ecut=8.5, nband=13, smearing=None)
        ssg2 = StaticSetGenerator(nband=8, smearing=None)
        ssg3 = StaticSetGenerator()

        with ScratchDir(".") as tmpdir:
            input_set1 = ssg1.get_input_set(
                structure=si_structure, smearing="marzari5:0.5 eV"
            )
            assert isinstance(input_set1, AbinitInputSet)
            input_set1.write_input("first_run")
            with open("first_run/run.abi", "r") as f:
                runabi1_str = f.read()
            assert "ecut 8.5" in runabi1_str
            assert "nband 13" in runabi1_str
            assert "occopt 5" in runabi1_str
            matched_tsmear_lines = [
                line for line in runabi1_str.split("\n") if "tsmear" in line
            ]
            assert len(matched_tsmear_lines) == 1
            _, tsmear, tsmear_unit = matched_tsmear_lines[0].split()
            assert float(tsmear) == pytest.approx(0.0183746610878275)
            assert tsmear_unit == "Ha"
            outdenpath = os.path.join(
                tmpdir, "first_run", OUTDIR_NAME, f"{OUTDATAFILE_PREFIX}_DEN"
            )
            Path(outdenpath).touch()

            input_set2 = ssg2.get_input_set(
                structure=si_structure,
                charge=1.0,
                restart_from=os.path.join(tmpdir, "first_run"),
            )
            input_set2.write_input("second_run")
            indenpath = os.path.join(
                tmpdir, "second_run", INDIR_NAME, f"{INDATAFILE_PREFIX}_DEN"
            )
            assert os.path.islink(indenpath)
            assert os.readlink(indenpath) == outdenpath
            with open("second_run/run.abi", "r") as f:
                runabi2_str = f.read()
            assert "ecut 8.5" in runabi2_str
            assert "nband 8" in runabi2_str
            assert "occopt" not in runabi2_str
            assert "tsmear" not in runabi2_str
            assert "charge 1.0" in runabi2_str
            assert "irdden 1" in runabi2_str

            outwfkpath = os.path.join(
                tmpdir, "first_run", OUTDIR_NAME, f"{OUTDATAFILE_PREFIX}_WFK"
            )
            Path(outwfkpath).touch()
            input_set3 = ssg3.get_input_set(
                structure=si_structure, restart_from=os.path.join(tmpdir, "first_run")
            )
            input_set3.write_input("second_run_bis")
            indenpath = os.path.join(
                tmpdir, "second_run_bis", INDIR_NAME, f"{INDATAFILE_PREFIX}_DEN"
            )
            inwfkpath = os.path.join(
                tmpdir, "second_run_bis", INDIR_NAME, f"{INDATAFILE_PREFIX}_WFK"
            )
            assert os.path.islink(inwfkpath)
            assert os.readlink(inwfkpath) == outwfkpath
            with open("second_run_bis/run.abi", "r") as f:
                runabi3_str = f.read()
            assert "ecut 8.5" in runabi3_str
            assert "nband 13" in runabi3_str
            assert "occopt 5" in runabi3_str
            matched_tsmear_lines = [
                line for line in runabi3_str.split("\n") if "tsmear" in line
            ]
            assert len(matched_tsmear_lines) == 1
            _, tsmear, tsmear_unit = matched_tsmear_lines[0].split()
            assert float(tsmear) == pytest.approx(0.0183746610878275)
            assert tsmear_unit == "Ha"
            outdenpath = os.path.join(
                tmpdir, "first_run", OUTDIR_NAME, f"{OUTDATAFILE_PREFIX}_DEN"
            )
