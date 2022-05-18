import os
from pathlib import Path

import pytest
from monty.os import makedirs_p
from monty.tempfile import ScratchDir

from atomate2.abinit.sets.base import AbinitInputSetGenerator, PrevOutput
from atomate2.abinit.utils.common import OUTDIR_NAME, InitializationError


class TestAbinitInputSetGenerator:
    def test_check_format_prev_dirs(self):
        aisg = AbinitInputSetGenerator()
        prev_outputs = aisg.check_format_prev_dirs("directory_path")
        assert prev_outputs == [
            PrevOutput(dirname="directory_path", exts=("WFK", "DEN"))
        ]
        prev_outputs = aisg.check_format_prev_dirs(["directory_path"])
        assert prev_outputs == [
            PrevOutput(dirname="directory_path", exts=("WFK", "DEN"))
        ]
        prev_outputs = aisg.check_format_prev_dirs(
            ["directory_path", "directory_path2"]
        )
        assert prev_outputs == [
            PrevOutput(dirname="directory_path", exts=("WFK", "DEN")),
            PrevOutput(dirname="directory_path2", exts=("WFK", "DEN")),
        ]
        prev_outputs = aisg.check_format_prev_dirs(["directory_path", "WFK"])
        assert prev_outputs == [
            PrevOutput(dirname="directory_path", exts=("WFK",)),
        ]
        prev_outputs = aisg.check_format_prev_dirs(
            [
                "directory_path",
                "directory_path2",
                ["directory_path3", ["1WF", "1DEN"]],
                "directory_path4",
            ]
        )
        assert prev_outputs == [
            PrevOutput(dirname="directory_path", exts=("WFK", "DEN")),
            PrevOutput(dirname="directory_path2", exts=("WFK", "DEN")),
            PrevOutput(dirname="directory_path3", exts=("1WF", "1DEN")),
            PrevOutput(dirname="directory_path4", exts=("WFK", "DEN")),
        ]
        # TODO: add the test of the raises ?

    def test_resolve_dep(self):
        aisg = AbinitInputSetGenerator()
        with ScratchDir(".") as tmpdir:
            prev_output_dir = os.path.join(tmpdir, "prev_output")
            prev_output = PrevOutput(
                dirname=prev_output_dir,
                exts=(
                    "WFK",
                    "DEN",
                ),
            )
            prev_outdata = os.path.join(prev_output_dir, OUTDIR_NAME)
            makedirs_p(prev_outdata)
            Path(os.path.join(prev_outdata, "output_DEN")).touch()
            irdvars, restart_file = aisg.resolve_dep(prev_output)
            assert irdvars == {"irdden": 1}
            assert restart_file == os.path.join(prev_outdata, "output_DEN")
            Path(os.path.join(prev_outdata, "output_WFK")).touch()
            irdvars, restart_file = aisg.resolve_dep(prev_output)
            assert irdvars == {"irdwfk": 1}
            assert restart_file == os.path.join(prev_outdata, "output_WFK")
        with ScratchDir(".") as tmpdir:
            prev_output_dir = os.path.join(tmpdir, "prev_output")
            prev_output = PrevOutput(
                dirname=prev_output_dir,
                exts=(
                    "WFK",
                    "DEN",
                ),
            )
            prev_outdata = os.path.join(prev_output_dir, OUTDIR_NAME)
            makedirs_p(prev_outdata)
            Path(os.path.join(prev_outdata, "output_WFK")).touch()
            irdvars, restart_file = aisg.resolve_dep(prev_output)
            assert irdvars == {"irdwfk": 1}
            assert restart_file == os.path.join(prev_outdata, "output_WFK")
            prev_output = PrevOutput(dirname=prev_output_dir, exts=("DDB",))
            with pytest.raises(
                InitializationError, match=r"Cannot find DDB file to restart from."
            ):
                aisg.resolve_dep(prev_output)
            prev_output = PrevOutput(
                dirname=prev_output_dir,
                exts=(
                    "DDB",
                    "DVDB",
                    "DKK",
                ),
            )
            with pytest.raises(
                InitializationError,
                match=r"Cannot find DDB or DVDB or DKK file to restart from.",
            ):
                aisg.resolve_dep(prev_output)

    def test_resolve_deps(self):
        aisg = AbinitInputSetGenerator()
        with ScratchDir(".") as tmpdir1:
            prev_output_dir1 = os.path.join(tmpdir1, "prev_output")
            prev_outdata1 = os.path.join(prev_output_dir1, OUTDIR_NAME)
            makedirs_p(prev_outdata1)
            Path(os.path.join(prev_outdata1, "output_DEN")).touch()
            with ScratchDir(".") as tmpdir2:
                prev_output_dir2 = os.path.join(tmpdir2, "prev_output")
                prev_outdata2 = os.path.join(prev_output_dir2, OUTDIR_NAME)
                makedirs_p(prev_outdata2)
                Path(os.path.join(prev_outdata2, "output_WFK")).touch()
                prev_outputs = [
                    PrevOutput(dirname=prev_output_dir1, exts=("DEN",)),
                    PrevOutput(dirname=prev_output_dir2, exts=("WFK",)),
                ]
                irdvars, input_files = aisg.resolve_deps(prev_outputs)
                assert irdvars == {"irdden": 1, "irdwfk": 1}
