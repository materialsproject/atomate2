import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, List, Optional

import pytest
from abipy.abio.input_tags import SCF
from abipy.abio.inputs import AbinitInput
from monty.os import makedirs_p
from monty.tempfile import ScratchDir

from atomate2.abinit.files import load_abinit_input
from atomate2.abinit.sets.base import AbinitInputSet, AbinitInputSetGenerator
from atomate2.abinit.utils.common import INDIR_NAME, OUTDIR_NAME, InitializationError


class TestAbinitInputSet:
    def test_init(self):
        abinit_input = "FakeAbinitInput"
        ais = AbinitInputSet(abinit_input=abinit_input)
        assert ais.inputs == {"run.abi": abinit_input}
        assert ais.input_files is None
        assert ais.link_files is True
        ais = AbinitInputSet(
            abinit_input=abinit_input,
            input_files=["/some/input/file", "/some/other/input/file"],
            link_files=False,
        )
        assert ais.inputs == {"run.abi": abinit_input}
        assert ais.input_files == ["/some/input/file", "/some/other/input/file"]
        assert ais.link_files is False

    def test_write_input(self, abinit_test_dir):
        abinit_input = load_abinit_input(
            os.path.join(abinit_test_dir, "abinit_inputs"), fname="abinit_input_Si.json"
        )
        with ScratchDir("."):
            ais = AbinitInputSet(abinit_input=abinit_input)
            ais.write_input("testdir")
            assert os.path.exists("testdir")
            dirlist = os.listdir("testdir")
            assert len(dirlist) == 5
            assert "indata" in dirlist
            assert "outdata" in dirlist
            assert "tmpdata" in dirlist
            assert os.path.isdir("testdir/indata")
            assert os.path.isdir("testdir/outdata")
            assert os.path.isdir("testdir/tmpdata")
            assert "run.abi" in dirlist
            assert "abinit_input.json" in dirlist
            with open("testdir/run.abi", "r") as f:
                abistr = f.read()
                assert "ecut" in abistr
            with open("testdir/abinit_input.json", "r") as f:
                abijsonstr = f.read()
                assert "@module" in abijsonstr
            with pytest.raises(FileExistsError):
                ais.write_input("testdir", overwrite=False)

        with ScratchDir(".") as tmpdir:
            prev_output_dir = os.path.join(tmpdir, "prev_output")
            prev_outdata = os.path.join(prev_output_dir, OUTDIR_NAME)
            makedirs_p(prev_outdata)
            out_den = Path(os.path.join(prev_outdata, "out_DEN"))
            out_den.touch()
            ais = AbinitInputSet(
                abinit_input=abinit_input, input_files=[out_den], validation=False
            )
            ais.write_input("testdir")
            in_den = os.path.join("testdir", "indata", "in_DEN")
            assert os.path.islink(in_den)
            assert os.readlink(in_den) == str(out_den)
            with pytest.raises(
                ValueError, match=r"'irdden' ird variable not set for 'in_DEN' file."
            ):
                ais = AbinitInputSet(
                    abinit_input=abinit_input, input_files=[out_den], validation=True
                )
                ais.write_input("testdir")
            abinit_input["irdden"] = 1
            ais = AbinitInputSet(
                abinit_input=abinit_input,
                input_files=[out_den],
                validation=True,
                link_files=False,
            )
            ais.write_input("testdir")
            assert os.path.exists(in_den)
            assert os.path.isfile(in_den)
            assert not os.path.islink(in_den)
            with open("testdir/run.abi", "r") as f:
                abistr = f.read()
                assert "irdden 1" in abistr
            abinit_input["irdden"] = 2
            with pytest.raises(
                ValueError,
                match=r"'irdden 2' ird variable is wrong for 'in_DEN' file. "
                r"Should be 'irdden 1'.",
            ):
                ais = AbinitInputSet(
                    abinit_input=abinit_input, input_files=[out_den], validation=True
                )
                ais.write_input("testdir")
            del abinit_input["irdden"]

        with ScratchDir(".") as tmpdir:
            prev_output_dir = os.path.join(tmpdir, "prev_output")
            prev_outdata = os.path.join(prev_output_dir, OUTDIR_NAME)
            makedirs_p(prev_outdata)
            with pytest.raises(
                ValueError,
                match=r"'fake_WRONGEXT' file in input directory "
                r"does not have a valid abinit extension.",
            ):
                wrontext_file = Path(os.path.join(prev_outdata, "fake_WRONGEXT"))
                wrontext_file.touch()
                ais = AbinitInputSet(
                    abinit_input=abinit_input,
                    input_files=[wrontext_file],
                    validation=True,
                )
                ais.write_input("testdir")

        with ScratchDir(".") as tmpdir:
            prev_output_dir = os.path.join(tmpdir, "prev_output")
            prev_outdata = os.path.join(prev_output_dir, OUTDIR_NAME)
            makedirs_p(prev_outdata)
            Path(os.path.join(prev_outdata, "out_TIM1_DEN")).touch()
            Path(os.path.join(prev_outdata, "out_TIM2_DEN")).touch()
            out_den = Path(os.path.join(prev_outdata, "out_TIM15_DEN"))
            out_den.touch()
            abinit_input["irdden"] = 1
            ais = AbinitInputSet(
                abinit_input=abinit_input,
                input_files=[{str(out_den): "in_DEN"}],
                validation=True,
            )
            ais.write_input("testdir")
            in_den = os.path.join("testdir", INDIR_NAME, "in_DEN")
            assert os.path.islink(in_den)
            assert os.readlink(in_den) == str(out_den)


@dataclass
class SomeAbinitInputSetGenerator(AbinitInputSetGenerator):
    calc_type: str = "some_calc"

    param1: int = 1
    param2: Optional[float] = None
    param3: List[int] = field(default_factory=list)

    extra_abivars: dict = field(default_factory=dict)

    restart_from_deps: tuple = (f"{SCF}:WFK|DEN",)

    # class variables
    params: ClassVar[tuple] = (
        "param1",
        "param2",
        "param3",
    )

    def get_abinit_input(
        self, structure=None, pseudos=None, prev_outputs=None, **kwargs
    ):
        return AbinitInput(structure=structure, pseudos=AbinitInputSetGenerator.pseudos)


class TestAbinitInputSetGenerator:
    def test_check_format_prev_dirs(self):
        aisg = AbinitInputSetGenerator()
        prev_outputs = aisg.check_format_prev_dirs(None)
        assert prev_outputs is None
        prev_outputs = aisg.check_format_prev_dirs("/some/path")
        assert prev_outputs == ["/some/path"]
        prev_outputs = aisg.check_format_prev_dirs(Path("/some/path"))
        assert prev_outputs == ["/some/path"]
        with pytest.raises(
            RuntimeError,
            match=r"Previous directories should be provided as a list "
            "or tuple of str or a single str.",
        ):
            aisg.check_format_prev_dirs(3.5)
        with pytest.raises(
            RuntimeError, match=r"Previous directory should be a str or a Path."
        ):
            aisg.check_format_prev_dirs(["/some/path", 3.5])
        prev_outputs = aisg.check_format_prev_dirs(
            ["/some/path", Path("/some/other/path")]
        )
        assert prev_outputs == ["/some/path", "/some/other/path"]

    def test_resolve_dep(self):
        aisg = AbinitInputSetGenerator()
        with ScratchDir(".") as tmpdir:
            prev_output_dir = os.path.join(tmpdir, "prev_output")
            prev_outdata = os.path.join(prev_output_dir, OUTDIR_NAME)
            makedirs_p(prev_outdata)
            Path(os.path.join(prev_outdata, "out_DEN")).touch()
            irdvars, restart_file = aisg.resolve_dep_exts(
                prev_output_dir, exts=("WFK", "DEN")
            )
            assert irdvars == {"irdden": 1}
            assert restart_file == [os.path.join(prev_outdata, "out_DEN")]
            Path(os.path.join(prev_outdata, "out_WFK")).touch()
            irdvars, restart_file = aisg.resolve_dep_exts(
                prev_output_dir, exts=("WFK", "DEN")
            )
            assert irdvars == {"irdwfk": 1}
            assert restart_file == [os.path.join(prev_outdata, "out_WFK")]
            irdvars, restart_file = aisg.resolve_dep_exts(
                prev_output_dir, exts=("DEN",)
            )
            assert irdvars == {"irdden": 1}
            assert restart_file == [os.path.join(prev_outdata, "out_DEN")]
        with ScratchDir(".") as tmpdir:
            prev_output_dir = os.path.join(tmpdir, "prev_output")
            prev_outdata = os.path.join(prev_output_dir, OUTDIR_NAME)
            makedirs_p(prev_outdata)
            Path(os.path.join(prev_outdata, "out_WFK")).touch()
            irdvars, restart_file = aisg.resolve_dep_exts(
                prev_output_dir, exts=("WFK", "DEN")
            )
            assert irdvars == {"irdwfk": 1}
            assert restart_file == [os.path.join(prev_outdata, "out_WFK")]
            with pytest.raises(
                InitializationError, match=r"Cannot find DDB file to restart from."
            ):
                aisg.resolve_dep_exts(prev_output_dir, exts=("DDB",))
            with pytest.raises(
                InitializationError,
                match=r"Cannot find DDB or DVDB or DKK file to restart from.",
            ):
                aisg.resolve_dep_exts(prev_output_dir, exts=("DDB", "DVDB", "DKK"))
        with ScratchDir(".") as tmpdir:
            prev_output_dir = os.path.join(tmpdir, "prev_output")
            prev_outdata = os.path.join(prev_output_dir, OUTDIR_NAME)
            makedirs_p(prev_outdata)
            Path(os.path.join(prev_outdata, "out_TIM1_DEN")).touch()
            Path(os.path.join(prev_outdata, "out_TIM2_DEN")).touch()
            Path(os.path.join(prev_outdata, "out_TIM15_DEN")).touch()
            irdvars, restart_file = aisg.resolve_dep_exts(
                prev_output_dir, exts=("DEN",)
            )
            assert irdvars == {"irdden": 1}
            assert restart_file == [
                {os.path.join(prev_outdata, "out_TIM15_DEN"): "in_DEN"}
            ]
            Path(os.path.join(prev_outdata, "out_DEN")).touch()
            irdvars, restart_file = aisg.resolve_dep_exts(
                prev_output_dir, exts=("DEN",)
            )
            assert irdvars == {"irdden": 1}
            assert restart_file == [os.path.join(prev_outdata, "out_DEN")]

    def test_resolve_deps(self):
        aisg = AbinitInputSetGenerator()
        with ScratchDir(".") as tmpdir1:
            prev_output_dir1 = os.path.join(tmpdir1, "prev_output")
            prev_outdata1 = os.path.join(prev_output_dir1, OUTDIR_NAME)
            makedirs_p(prev_outdata1)
            den = Path(os.path.join(prev_outdata1, "out_DEN"))
            den.touch()
            with ScratchDir(".") as tmpdir2:
                prev_output_dir2 = os.path.join(tmpdir2, "prev_output")
                prev_outdata2 = os.path.join(prev_output_dir2, OUTDIR_NAME)
                makedirs_p(prev_outdata2)
                wfk = Path(os.path.join(prev_outdata2, "out_WFK"))
                wfk.touch()
                prev_outputs = [prev_output_dir1, prev_output_dir2]
                irdvars, input_files = aisg.resolve_deps(
                    prev_outputs, deps=("any:WFK|DEN",), check_runlevel=False
                )
                assert irdvars == {"irdden": 1, "irdwfk": 1}
                assert str(wfk) in input_files
                assert str(den) in input_files
                assert len(input_files) == 2

    def test_restart_from_params(self):
        saisg1 = SomeAbinitInputSetGenerator(param2=3.0, param3=[1, 2, 3])
        kwargs1 = {"param2": 5.0}
        params1, extra_abivars1 = saisg1._get_parameters(
            kwargs=kwargs1, prev_generator=None
        )
        assert params1 == {"param1": 1, "param2": 5.0, "param3": [1, 2, 3]}
        assert extra_abivars1 == {}
        saisg1_copy = saisg1._get_generator(
            gen_params=params1, extra_abivars=extra_abivars1
        )
        saisg2 = SomeAbinitInputSetGenerator(extra_abivars={"extra_param1": 1})
        kwargs2 = {"param2": 10.0}
        params2, extra_abivars2 = saisg2._get_parameters(
            kwargs=kwargs2, prev_generator=saisg1_copy
        )
        assert params2 == {"param1": 1, "param2": 10.0, "param3": [1, 2, 3]}
        assert extra_abivars2 == {"extra_param1": 1}
        saisg2_copy = saisg2._get_generator(
            gen_params=params2, extra_abivars=extra_abivars2
        )
        saisg3 = SomeAbinitInputSetGenerator(
            param3=[4, 5], extra_abivars={"extra_param2": 11}
        )
        saisg3.param1 = 15
        kwargs3 = {"extra_abivars": {"extra_param1": 2}, "param1": 8}
        params3, extra_abivars3 = saisg3._get_parameters(
            kwargs=kwargs3, prev_generator=saisg2_copy
        )
        assert params3 == {"param1": 8, "param2": 10.0, "param3": [4, 5]}
        assert extra_abivars3 == {"extra_param1": 2, "extra_param2": 11}
        saisg3_copy = saisg3._get_generator(
            gen_params=params3, extra_abivars=extra_abivars3
        )
        assert not saisg1.param_is_explicitly_set("param1")
        assert saisg1.param_is_explicitly_set("param2")
        assert saisg1.param_is_explicitly_set("param3")
        assert saisg1_copy.param_is_explicitly_set("param1")
        assert saisg1_copy.param_is_explicitly_set("param2")
        assert saisg1_copy.param_is_explicitly_set("param3")
        assert not saisg2.param_is_explicitly_set("param1")
        assert not saisg2.param_is_explicitly_set("param2")
        assert not saisg2.param_is_explicitly_set("param3")
        assert saisg2_copy.param_is_explicitly_set("param1")
        assert saisg2_copy.param_is_explicitly_set("param2")
        assert saisg2_copy.param_is_explicitly_set("param3")
        assert saisg3.param_is_explicitly_set("param1")
        assert not saisg3.param_is_explicitly_set("param2")
        assert saisg3.param_is_explicitly_set("param3")
        assert saisg3_copy.param_is_explicitly_set("param1")
        assert saisg3_copy.param_is_explicitly_set("param2")
        assert saisg3_copy.param_is_explicitly_set("param3")

    def test_get_input_set(self, mocker, si_structure):
        with ScratchDir(".") as tmpdir:
            saisg = SomeAbinitInputSetGenerator(
                param1=2, extra_abivars={"ecut": 5.0, "nstep": 25}
            )
            spy = mocker.spy(saisg, "_get_generator")
            abinit_input_set = saisg.get_input_set(
                structure=si_structure,
                param2=0.5,
                extra_abivars={"nstep": 35, "tsmear": 0.04},
            )
            assert spy.call_count == 1
            returned_gen = spy.spy_return
            assert returned_gen.param1 == 2
            assert returned_gen.param2 == 0.5
            assert returned_gen.param3 == []
            assert returned_gen.extra_abivars == {
                "ecut": 5.0,
                "nstep": 35,
                "tsmear": 0.04,
            }
            abinit_input_set.write_input("output1")
            output1 = os.path.join(tmpdir, "output1")

            saisg.param3 = [1]
            out_wfk1 = Path(os.path.join(output1, OUTDIR_NAME, "out_WFK"))
            out_wfk1.touch()
            spy.reset_mock()
            abinit_input_set = saisg.get_input_set(
                structure=si_structure,
                restart_from=output1,
                param2=5.5,
                extra_abivars={"nstep": 5},
            )
            abinit_input_set.write_input("output2")
            output2 = os.path.join(tmpdir, "output2")
            in_wfk2 = os.path.join(output2, INDIR_NAME, "in_WFK")
            assert os.path.islink(in_wfk2)
            assert os.readlink(in_wfk2) == str(out_wfk1)
            assert spy.call_count == 1
            returned_gen = spy.spy_return
            assert returned_gen.param1 == 2
            assert returned_gen.param2 == 5.5
            assert returned_gen.param3 == [1]
            assert returned_gen.extra_abivars == {
                "ecut": 5.0,
                "nstep": 5,
                "tsmear": 0.04,
            }
