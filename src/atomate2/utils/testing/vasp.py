"""Utilities for testing VASP calculations."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from types import NoneType
from typing import TYPE_CHECKING, Any, Final, Literal, get_args

from jobflow import CURRENT_JOB
from monty.io import zopen
from monty.os.path import zpath as monty_zpath
from monty.serialization import dumpfn, loadfn
from pydantic import BaseModel, model_validator
from pymatgen.io.vasp import Incar, Kpoints, Poscar, Potcar
from pymatgen.io.vasp.sets import VaspInputSet
from pymatgen.util.coord import in_coord_list_pbc

import atomate2.vasp.jobs.base
import atomate2.vasp.jobs.neb

try:
    import atomate2.vasp.jobs.defect

    pmg_defects_installed = True
except ImportError:
    pmg_defects_installed = False

import atomate2.vasp.run
from atomate2.vasp.sets.base import VaspInputGenerator

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from pymatgen.io.vasp.inputs import VaspInput
    from pytest import MonkeyPatch
    from typing_extensions import Self


logger = logging.getLogger("atomate2")

_VFILES: Final = ("incar", "kpoints", "potcar", "poscar")
_REF_PATHS: dict[str, str | Path] = {}
_FAKE_RUN_VASP_KWARGS: dict[str, dict] = {}


def zpath(path: str | Path) -> Path:
    """Return the path of a zip file.

    Returns an existing (zipped or unzipped) file path given the unzipped
    version. If no path exists, returns the unmodified path.
    """
    return Path(monty_zpath(str(path)))


def monkeypatch_vasp(
    monkeypatch: MonkeyPatch, vasp_test_dir: Path, nelect: int = 12
) -> Generator[Callable[[Any, Any], Any]]:
    """Fake VASP calculations by copying reference files.

    This is provided as a generator and can be used as by conextmanagers and
    pytest.fixture.

    It works by monkeypatching (replacing) calls to run_vasp and
    VaspInputSet.write_inputs with versions that will work when the vasp executables or
    POTCAR files are not present.

    The primary idea is that instead of running VASP to generate the output files,
    reference files will be copied into the directory instead. As we do not want to
    test whether VASP is giving the correct output rather that the calculation inputs
    are generated correctly and that the outputs are parsed properly, this should be
    sufficient for our needs. Another potential issue is that the POTCAR files
    distributed with VASP are not present on the testing server due to licensing
    constraints. Accordingly, VaspInputSet.write_inputs will fail unless the
    "potcar_spec" option is set to True, in which case a POTCAR.spec file will be
    written instead.

    The pytext.fixture defined with this is stored at tests/vasp/conftest.py.
    For examples, see the tests in tests/vasp/makers/core.py.

    Parameters
    ----------
    monkeypatch: The a MonkeyPatch object from pytest, this is meant as a place-holder
        For the `monkeypatch` fixture in pytest.
    vasp_test_dir: The root directory for the VASP tests. This is where the reference
        test data is located.
    nelect: The number of electrons in a system is usually calculate using the POTCAR
        which we do not have direct access to during testing. So we have to patch it in.
        TODO: potcar_spec should have the nelect data somehow.
    """

    def mock_run_vasp(*_args, **_kwargs) -> None:
        name = CURRENT_JOB.job.name
        try:
            ref_path = vasp_test_dir / _REF_PATHS[name]
        except KeyError:
            raise ValueError(
                f"no reference directory found for job {name!r}; "
                f"reference paths received={_REF_PATHS}"
            ) from None

        if "json" in str(ref_path).lower():
            with TemporaryDirectory() as temp_ref_dir:
                ref_data = VaspTestData(**loadfn(ref_path))
                ref_data.reconstruct(out_path=temp_ref_dir)
                fake_run_vasp(Path(temp_ref_dir), **_FAKE_RUN_VASP_KWARGS.get(name, {}))
        else:
            fake_run_vasp(ref_path, **_FAKE_RUN_VASP_KWARGS.get(name, {}))

    get_input_set_orig = VaspInputGenerator.get_input_set

    def mock_get_input_set(self: VaspInputGenerator, *_args, **_kwargs) -> VaspInput:
        _kwargs["potcar_spec"] = True
        return get_input_set_orig(self, *_args, **_kwargs)

    def mock_nelect(*_args, **_kwargs) -> int:
        return nelect

    monkeypatch.setattr(atomate2.vasp.run, "run_vasp", mock_run_vasp)
    monkeypatch.setattr(atomate2.vasp.jobs.base, "run_vasp", mock_run_vasp)
    monkeypatch.setattr(atomate2.vasp.jobs.neb, "run_vasp", mock_run_vasp)
    if pmg_defects_installed:
        monkeypatch.setattr(atomate2.vasp.jobs.defect, "run_vasp", mock_run_vasp)
    monkeypatch.setattr(VaspInputSet, "get_input_set", mock_get_input_set)
    monkeypatch.setattr(VaspInputSet, "nelect", mock_nelect)

    def _run(ref_paths: dict, fake_run_vasp_kwargs: dict | None = None) -> None:
        _REF_PATHS.update(ref_paths)
        _FAKE_RUN_VASP_KWARGS.update(fake_run_vasp_kwargs or {})

    yield _run

    monkeypatch.undo()
    _REF_PATHS.clear()
    _FAKE_RUN_VASP_KWARGS.clear()


def fake_run_vasp(
    ref_path: Path,
    incar_settings: Sequence[str] | None = None,
    incar_exclude: Sequence[str] | None = None,
    check_inputs: Sequence[Literal["incar", "kpoints", "poscar", "potcar"]] = _VFILES,
    clear_inputs: bool = True,
) -> None:
    """
    Emulate running VASP and validate VASP input files.

    Parameters
    ----------
    ref_path
        Path to reference directory with VASP input files in the folder named 'inputs'
        and output files in the folder named 'outputs'.
    incar_settings
        A list of INCAR settings to check. Defaults to None which checks all settings.
        Empty list or tuple means no settings will be checked.
    incar_exclude
        A list of INCAR settings to exclude from checking. Defaults to None, meaning
        no settings will be excluded.
    check_inputs
        A list of vasp input files to check. Supported options are "incar", "kpoints",
        "poscar", "potcar", "wavecar".
    clear_inputs
        Whether to clear input files before copying in the reference VASP outputs.
    """
    logger.info("Running fake VASP.")

    if "incar" in check_inputs:
        _check_incar(ref_path, incar_settings, incar_exclude)

    if "kpoints" in check_inputs:
        _check_kpoints(ref_path)

    if "poscar" in check_inputs:
        if len(neb_sub_dirs := sorted((ref_path / "inputs").glob("[0-9][0-9]"))) > 0:
            for idx, neb_sub_dir in enumerate(neb_sub_dirs):
                _check_poscar(
                    zpath(neb_sub_dir / "POSCAR"),
                    user_poscar_path=zpath(Path(f"{idx:02}") / "POSCAR"),
                )
        else:
            _check_poscar(ref_path)

    if "potcar" in check_inputs:
        _check_potcar(ref_path)

    # This is useful to check if the WAVECAR has been copied
    if "wavecar" in check_inputs and not Path("WAVECAR").exists():
        raise ValueError("WAVECAR was not correctly copied")

    logger.info("Verified inputs successfully")

    if clear_inputs:
        _clear_vasp_inputs()
    _copy_vasp_outputs(ref_path)

    # pretend to run VASP by copying pre-generated outputs from reference dir
    logger.info("Generated fake vasp outputs")


def _check_incar(
    ref_path: Path,
    incar_settings: Sequence[str] | None,
    incar_exclude: Sequence[str] | None,
) -> None:
    """Check that INCAR settings are consistent with the reference calculation."""
    user_incar = Incar.from_file(zpath("INCAR"))
    ref_incar_path = zpath(ref_path / "inputs" / "INCAR")
    ref_incar = Incar.from_file(ref_incar_path)
    defaults = {"ISPIN": 1, "ISMEAR": 1, "SIGMA": 0.2}

    keys_to_check = (
        set(user_incar) if incar_settings is None else set(incar_settings)
    ) - set(incar_exclude or [])
    for key in keys_to_check:
        user_val = user_incar.get(key, defaults.get(key))
        ref_val = ref_incar.get(key, defaults.get(key))
        if user_val != ref_val:
            raise ValueError(
                f"\n\nINCAR value of {key} is inconsistent: expected {ref_val}, "
                f"got {user_val} \nin ref file {ref_incar_path}"
            )


def _check_kpoints(ref_path: Path) -> None:
    """Check that KPOINTS file is consistent with the reference calculation."""
    user_kpoints_exists = (user_kpt_path := zpath("KPOINTS")).exists()
    ref_kpoints_exists = (
        ref_kpt_path := zpath(ref_path / "inputs" / "KPOINTS")
    ).exists()

    if user_kpoints_exists and not ref_kpoints_exists:
        raise ValueError(
            "atomate2 generated a KPOINTS file but the reference calculation is using "
            "KSPACING"
        )
    if not user_kpoints_exists and ref_kpoints_exists:
        raise ValueError(
            "atomate2 is using KSPACING but the reference calculation is using "
            "a KPOINTS file"
        )
    if user_kpoints_exists and ref_kpoints_exists:
        user_kpts = Kpoints.from_file(user_kpt_path)
        ref_kpts = Kpoints.from_file(ref_kpt_path)
        if user_kpts.style != ref_kpts.style or user_kpts.num_kpts != ref_kpts.num_kpts:
            raise ValueError(
                f"\n\nKPOINTS files are inconsistent: {user_kpts.style} != "
                f"{ref_kpts.style} or {user_kpts.num_kpts} != {ref_kpts.num_kpts}\nin "
                f"ref file {ref_kpt_path}"
            )
    else:
        # check k-spacing
        user_incar = Incar.from_file(zpath("INCAR"))
        ref_incar_path = zpath(ref_path / "inputs" / "INCAR")
        ref_incar = Incar.from_file(ref_incar_path)

        user_ksp, ref_ksp = user_incar.get("KSPACING"), ref_incar.get("KSPACING")
        if user_ksp != ref_ksp:
            raise ValueError(
                f"\n\nKSPACING is inconsistent: expected {ref_ksp}, got {user_ksp} "
                f"\nin ref file {ref_incar_path}"
            )


def _check_poscar(
    ref_poscar_path: Path,
    user_poscar_path: Path | None = None,
) -> None:
    """Check that POSCAR information is consistent with the reference calculation."""
    user_poscar_path = user_poscar_path or zpath("POSCAR")
    if ref_poscar_path.is_dir():
        ref_poscar_path = zpath(ref_poscar_path / "inputs" / "POSCAR")

    user_poscar = Poscar.from_file(user_poscar_path)
    ref_poscar = Poscar.from_file(ref_poscar_path)

    user_frac_coords = user_poscar.structure.frac_coords
    ref_frac_coords = ref_poscar.structure.frac_coords

    # In some cases, the ordering of sites can change when copying input files.
    # To account for this, we check that the sites are the same, within a tolerance,
    # while accounting for PBC.
    coord_match = [
        in_coord_list_pbc(ref_frac_coords, coord, atol=1e-3)
        for coord in user_frac_coords
    ]
    if (
        user_poscar.natoms != ref_poscar.natoms
        or user_poscar.site_symbols != ref_poscar.site_symbols
        or not all(coord_match)
    ):
        no_match = [f"{idx}" for idx, v in enumerate(coord_match) if not v]
        aux_str = " on site(s) " + ", ".join(no_match) if no_match else ""
        raise ValueError(
            f"POSCAR files are inconsistent{aux_str}"
            f"\n\n{ref_poscar_path!s}\n{ref_poscar}"
            f"\n\n{user_poscar_path!s}\n{user_poscar}"
        )


def _check_potcar(ref_path: Path) -> None:
    """Check that POTCAR information is consistent with the reference calculation."""
    potcars = {"reference": None, "user": None}
    paths = {"reference": ref_path / "inputs", "user": Path(".")}
    for mode, path in paths.items():
        if (potcar_path := zpath(path / "POTCAR")).exists():
            potcars[mode] = Potcar.from_file(potcar_path).symbols
        elif (potcar_path := zpath(path / "POTCAR.spec")).exists():
            with zopen(potcar_path, "rt") as f:
                potcars[mode] = f.read().strip().split("\n")
        else:
            raise FileNotFoundError(f"no {mode} POTCAR or POTCAR.spec file found")

    if potcars["reference"] != potcars["user"]:
        raise ValueError(
            "POTCAR files are inconsistent: "
            f"{potcars['reference']} != {potcars['user']}"
        )


def _clear_vasp_inputs() -> None:
    """Clean up VASP input files."""
    for vasp_file in (
        "INCAR",
        "KPOINTS",
        "POSCAR",
        "POTCAR",
        "CHGCAR",
        "OUTCAR",
        "vasprun.xml",
        "CONTCAR",
    ):
        if (file_path := zpath(vasp_file)).exists():
            file_path.unlink()
    logger.info("Cleared vasp inputs")


def _copy_vasp_outputs(ref_path: Path) -> None:
    """Copy VASP output files from the reference directory."""
    output_path = ref_path / "outputs"
    neb_dirs = sorted(output_path.glob("[0-9][0-9]"))

    for output_file in output_path.iterdir():
        if output_file.is_file():
            shutil.copy(output_file, ".")

    for idx, neb_dir in enumerate(neb_dirs):
        (copied_neb_dir := Path(f"./{idx:02}")).mkdir(parents=True, exist_ok=True)
        for output_file in neb_dir.iterdir():
            if output_file.is_file():
                shutil.copy(output_file, copied_neb_dir)


class TestData(BaseModel):
    """
    Utility class to group VASP testing data.

    This is the base class, for creating an archive of test data,
    use `VaspTestData.from_directory` on a valid VASP calculation directory.

    This class also defines methods to establish appropriate directory
    structure for VASP test data, without user intervention:

    base_dir :
        - inputs
            - INCAR
            - KPOINTS (optional)
            - POSCAR
            - POTCAR.spec
        - outputs
            - INCAR
            - KPOINTS (optional)
            - POSCAR
            - POTCAR.spec
            - CONTCAR
            - OUTCAR
            - vasprun.xml
    """

    @model_validator(mode="before")
    @classmethod
    def serialize_from_str(cls, config: dict) -> dict:
        """Ensure class objects are serialized as defined in schema."""
        init_keys = list(config)
        for _k in init_keys:
            k = _k.replace(".", "_")
            field_class = cls._resolve_non_null_class(cls.model_fields[k].annotation)[0]
            if hasattr(field_class, "from_str") and isinstance(config[_k], str):
                config[k] = field_class.from_str(config[_k])
            if k != _k:
                config[k] = config.pop(_k)
        return config

    @staticmethod
    def flatten_dict(dct: dict, separator: str = ".") -> dict[str, str | bytes]:
        """
        Flatten an input dict with a nested structure.

        Parameters
        ----------
        dct : dict
        separator : str = "."
            The separator to use to flatten keys, e.g.:
                x = {"a": {"b": 1}}
            would get flattened into
                x = {"a.b": 1}

        Returns
        -------
        Flattened dict
        """
        f: dict[str, str | bytes] = {}

        def _flatten_dict(key: Any, value: Any, flattened: dict) -> Any:
            if hasattr(value, "items"):
                for k, v in value.items():
                    new_key = f"{key}{separator}{k}" if key else k
                    _flatten_dict(new_key, v, flattened)
            else:
                flattened[key] = value

        _flatten_dict("", dct, f)
        return f

    @staticmethod
    def _resolve_non_null_class(type_anno: Any) -> Any:
        """Resolve the possible non-null classes a type annotation includes."""
        anno_types = get_args(type_anno)
        return [typ for typ in anno_types if typ != NoneType]

    @classmethod
    def from_directory(
        cls, dir_name: str | Path, suffix: str | None = None, **kwargs
    ) -> Self:
        """
        Create an instance of TestData recursively from a directory.

        If a subclass includes nested TestData subclasses, they will also call
        `from_directory` recursively. This permits an appropriate input / output
        directory structure to be created automatically.

        This class also removes copyright-protected POTCAR data and converts it to
        a POTCAR.spec object.

        Note that any field names with "_" characters have these replaced by "."
        when parsing.

        Parameters
        ----------
        dir_name : str or Path
            Path to where the VASP files are located
        suffix : str or None (default)
            suffix of files, like ".orig" if inclusign original inputs.
        """
        dir_name = Path(dir_name)
        for fname, field_info in cls.model_fields.items():
            fpath = zpath(dir_name / f"{fname.replace('_', '.')}{suffix or ''}")
            field_class = cls._resolve_non_null_class(field_info.annotation)[0]
            if fpath.exists():
                if hasattr(field_class, "from_file"):
                    kwargs[fname] = field_class.from_file(fpath)
                else:
                    typ = "t" if isinstance(field_class, str) else "b"
                    with zopen(fpath, f"r{typ}") as f:
                        kwargs[fname] = f.read()

            elif issubclass(field_class, TestData):
                kwargs[fname] = field_class.from_directory(dir_name)

        return cls(**kwargs)

    def _to_dict(
        self,
        dct: dict[str, Any],
        prefix: str | None = None,
        suffix: str | None = None,
    ) -> dict[str, str]:
        """
        Recursively parse class to JSON-able dict, internal method only.

        Parameters
        ----------
        dct : dict[str,str or bytes]
        prefix : str or None (default)
            Hierarchical prefix, e.g., "inputs" or "outputs"
        suffix : str or None (default)
            Optional file suffix, e.g., ".orig"

        Returns
        -------
        dict[str,str] : JSON-able dict
        """
        if prefix and not dct.get(prefix):
            dct[prefix] = {}
            _dct = dct[prefix]
        else:
            _dct = dct

        for file_name, obj in dict(self).items():
            if not obj:
                continue

            if isinstance(obj, TestData):
                meta = file_name.split("_")
                sub_suffix = ""
                if len(meta) > 1:
                    sub_suffix = "." + ".".join(meta[1:])
                obj._to_dict(_dct, prefix=meta[0], suffix=sub_suffix)  # noqa: SLF001
                continue

            alias = f"{file_name.replace('_', '.')}{suffix or ''}"
            if "POTCAR" in file_name and "spec" not in file_name:
                fdata = "\n".join(
                    (Potcar.from_str(obj) if isinstance(obj, str) else obj).symbols
                )
                alias += ".spec"
            elif hasattr(obj, "__str__"):
                fdata = str(obj)
            else:
                fdata = obj

            _dct[alias] = fdata

        return dct

    def to_dict(
        self, prefix: str | None = None, suffix: str | None = None
    ) -> dict[str, Any]:
        """
        Convert the current test data to a JSON-able dict.

        Parameters
        ----------
        prefix : str or None (default)
            Hierarchical prefix, e.g., "inputs" or "outputs"
        suffix : str or None (default)
            Optional file suffix, e.g., ".orig"

        Returns
        -------
        dict[str,str] : JSON-able dict
        """
        return self._to_dict({}, prefix=prefix, suffix=suffix)

    def to_file(self, file_name: str | Path) -> None:
        """
        Dump the dict representation of the test data to a file.

        Parameters
        ----------
        file_name : str or Path
        """
        dumpfn(self.to_dict(), file_name)

    def reconstruct(
        self, out_path: str | Path | None = None, copy_input: bool = True
    ) -> None:
        """
        Write the files with correct directory structure to a directory.

        Parameters
        ----------
        out_path : str, Path, or None
            Optional output base directory to write files to. Defaults to "."
        copy_input : bool = True (default)
            Whether to copy all input files to the "output" directory.
            This is the default behavior for VASP test data.
        """
        out_path = Path(out_path or ".")
        for fpath, obj_str in self.flatten_dict(self.to_dict(), separator="/").items():
            p = out_path / fpath
            if not p.parent.exists():
                p.parent.mkdir(exist_ok=True, parents=True)
            with zopen(p, "wt") as f:
                f.write(obj_str)

        if copy_input:
            for p in (out_path / "inputs").glob("*"):
                if p.is_file():
                    new_p = Path(str(p).replace("input", "output"))
                    if not new_p.parent.exists():
                        new_p.parent.mkdir(exist_ok=True, parents=True)
                    shutil.copyfile(p, new_p)


class VaspInputTestData(TestData):
    """
    Schema for input VASP test data.

    Fields
    -------
    INCAR : pymatgen.io.vasp.inputs.Incar (optional)
    KPOINTS : pymatgen.io.vasp.inputs.Kpoints (optional)
    POSCAR : pymatgen.io.vasp.inputs.Poscar (optional)
    POTCAR : pymatgen.io.vasp.inputs.Potcar (optional)
    POTCAR_spec : str (optional)
        These are just the POTCAR symbols.
    """

    INCAR: Incar | None = None
    KPOINTS: Kpoints | None = None
    POSCAR: Poscar | None = None
    POTCAR: Potcar | None = None
    POTCAR_spec: str | None = None


class VaspOutputTestData(TestData):
    """
    Schema for output VASP test data.

    Fields
    -------
    CONTCAR : pymatgen.io.vasp.inputs.Poscar (optional)
    OUTCAR : str (optional)
    vasprun_xml : str (optional)
    WAVECAR : str (optional)
    CHGCAR : str (optional)
    """

    CONTCAR: Poscar | None = None
    OUTCAR: str | None = None
    vasprun_xml: str | None = None
    WAVECAR: str | None = None
    CHGCAR: str | None = None


class VaspTestData(TestData):
    """
    Schema for a single VASP calculation test data.

    Use this class to automatically generate test data
    from a single VASP calculation directory, using the
    `from_directory` method:

    ```python
    vasp_data = VaspTestData.from_directory("path to VASP calculation")
    vasp_data.to_file("name of output file.json")
    ```

    You can use any compression supported by monty.io.zopen.

    Note, if you want original inputs files, add this Field to a subclass:

    ```python
    inputs_orig: VaspInputTestData | None = None
    ```

    Fields
    -------
    inputs : VaspInputTestData (optional)
    outputs : VaspOutputTestData (optional)
    """

    inputs: VaspInputTestData | None = None
    outputs: VaspOutputTestData | None = None
