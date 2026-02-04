"""Define utility functions for processing MOF and zeolites.

This module include a wrapper for the zeo++ executable and
calculate pore properties
For information about the current flows, contact:
- Theo Jaffrelot Inizan (@tjaffrel)
- Aaron Kaplan (@esoteric-ephemera)

References
----------
Preprint:
@misc{inizan2025agenticaidiscoverymetalorganic,
    title={System of Agentic AI for the Discovery of Metal-Organic Frameworks},
    author={
        Theo Jaffrelot Inizan and Sherry Yang and Aaron Kaplan and Yen-hsu Lin
        and Jian Yin and Saber Mirzaei and Mona Abdelgaid and Ali H. Alawadhi
        and KwangHwan Cho and Zhiling Zheng and Ekin Dogus Cubuk and Christian Borgs
        and Jennifer T. Chayes and Kristin A. Persson and Omar M. Yaghi
    },
    year={2025},
    eprint={2504.14110},
    archivePrefix={arXiv},
    primaryClass={cond-mat.mtrl-sci},
    url={https://arxiv.org/abs/2504.14110},
}

Database:
https://next-gen.materialsproject.org/contribs/MOFGen_2025
"""

import logging
import multiprocessing
import os
import subprocess
from collections.abc import Callable
from shutil import which
from tempfile import TemporaryDirectory
from typing import Any

from jobflow import job
from pydantic import BaseModel
from pymatgen.core import Structure

logger = logging.getLogger(__name__)

_installed_extra = {"mofid": True}
try:
    from mofid.run_mofid import cif2mofid
except ImportError:
    _installed_extra["mofid"] = False


class MofIdEntry(BaseModel):
    """
    Interface for running MOFid calculations.

    This class wraps the mofid executable to extract key MOF components.
    """

    smiles: str | None = None
    Topology: str | None = None
    SmilesLinkers: list[str] | None = None
    SmilesNodes: list[str] | None = None
    Mofkey: str | None = None
    Mofid: str | None = None

    @classmethod
    def from_structure(cls, structure: Structure, **kwargs) -> "MofIdEntry":
        """
        Run MOFid, `cif2mofid` function, in a temporary directory.

        Store MOFid information: MOF topology, linker and metal nodes SMILES.
        """
        if not _installed_extra["mofid"]:
            logger.debug("MOFid not found, skipping MOFid analysis.")
            return cls()
        old_cwd = os.getcwd()
        try:
            with TemporaryDirectory() as tmp:
                os.chdir(tmp)
                structure.to("tmp.cif")
                mofid_out = cif2mofid("tmp.cif", **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MOFid failed: %s", exc)
            return cls()
        os.chdir(old_cwd)

        remap = {
            "Smiles": "smiles",
            "Topology": "topology",
            "SmilesLinkers": "smiles_linkers",
            "SmilesNodes": "smiles_nodes",
            "MofKey": "mofkey",
            "MofId": "mofid",
        }
        return cls(**{k: mofid_out.get(v) for k, v in remap.items()})


class ZeoPlusPlus:
    """
    Interface for running zeo++ calculations for MOF or zeolites.

    This class wraps the zeo++ executable to calculate pore properties
    (e.g, Probe-occupiable volume, Pore diameters - see zeoplusplus.org)
    using given sorbate species.
    """

    def __init__(
        self,
        cif_path: str,
        zeopp_path: str | None = None,
        working_dir: str | None = None,
        sorbates: list[str] | str | None = None,
    ) -> None:
        if sorbates is None:
            sorbates = ["N2", "CO2", "H2O"]
        elif isinstance(sorbates, str):
            sorbates = [sorbates]
        self._cif_path = cif_path
        self.cif_name = os.path.basename(cif_path.split(".cif", maxsplit=1)[0])
        self.zeopp_path = zeopp_path or which("zeo++") or os.environ.get("ZEO_PATH")
        self.sorbates: list[str] = sorbates
        self.working_dir = working_dir or os.path.dirname(cif_path)
        self._zeopp_path = zeopp_path

    @classmethod
    def from_structure(
        cls,
        structure: Structure,
        cif_path: str,
        zeopp_path: str | None = None,
        working_dir: str | None = None,
        sorbates: list[str] | str | None = None,
    ) -> "ZeoPlusPlus":
        """
        Create a ZeoPlusPlus instance from a pymatgen Structure.

        Parameters
        ----------
        structure : Structure
            Input pymatgen structure object.
        cif_path : str
            Path to write the CIF and output files.
        zeopp_path : str, optional
            Path to zeo++ executable.
            For ease of use, set ZEO_PATH in your bashrc, e.g:
            export ZEO_PATH="/my/path/zeopp-lsmo/zeo++/network"
            or
            zeopp_path = "/my/path/zeopp-lsmo/zeo++/network"
        working_dir : str, optional
            Directory for temporary files.
        sorbates : list[str] or str, optional
            List of multiple or single sorbate.

        Returns
        -------
        ZeoPlusPlus
            An instance of the ZeoPlusPlus class.
        """
        structure.to(cif_path)
        return cls(
            cif_path=cif_path,
            zeopp_path=zeopp_path,
            working_dir=working_dir,
            sorbates=sorbates,
        )

    def run(
        self,
        zeopp_args: list[str] | None = None,
        nproc: int = 1,
    ) -> None:
        """
        Run the zeo++ calculations on multi-processor.

        Parameters
        ----------
        zeopp_args : list[str], optional
            Additional arguments for zeo++.
        nproc : int, optional
            Number of processes to run in parallel.
        """
        nproc = min(nproc, len(self.sorbates))
        sorbate_batches: list[list[str]] = [[] for _ in range(nproc)]
        iproc = 0
        for sorbate in self.sorbates:
            sorbate_batches[iproc].append(sorbate)
            iproc = (iproc + 1) % nproc

        manager = multiprocessing.Manager()
        output_file_path = manager.dict()
        output = manager.dict()

        procs = []
        for iproc in range(nproc):
            proc = multiprocessing.Process(
                target=self._run_zeopp_many,
                kwargs={
                    "sorbates": sorbate_batches[iproc],
                    "file_paths_shared": output_file_path,
                    "output_shared": output,
                    "zeopp_args": zeopp_args,
                },
            )
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

        self.output_file_path = dict(output_file_path)
        self.output = dict(output)

    def _run_zeopp_many(
        self,
        sorbates: list[str],
        file_paths_shared: dict[str, Any],
        output_shared: dict[str, Any],
        zeopp_args: list[str] | None = None,
    ) -> None:
        """
        Run zeo++ for multiple sorbates.

        Parameters
        ----------
        sorbates : list[str]
            List of sorbates.
        file_paths_shared : dict
            Shared dictionary for output file paths.
        output_shared : dict
            Shared dictionary for outputs.
        zeopp_args : list[str], optional
            Additional arguments for zeo++.
        """
        for sorbate in sorbates:
            self._run_zeopp_single(
                sorbate, file_paths_shared, output_shared, zeopp_args=zeopp_args
            )

    def _run_zeopp_single(
        self,
        sorbate: str,
        file_paths_shared: dict[str, Any],
        output_shared: dict[str, Any],
        zeopp_args: list[str] | None = None,
    ) -> None:
        """
        Run zeo++ for a single sorbate.

        Parameters
        ----------
        sorbate : str
            String of a single sorbate.
        file_paths_shared : dict
            Shared dictionary for output file paths.
        output_shared : dict
            Shared dictionary for outputs.
        zeopp_args : list[str], optional
            Additional arguments for zeo++.
        """
        radius_sorbate = self.get_sorbate_radius(sorbate)
        parse_func = None
        flag_to_func: dict[str, Any] = {
            "res": self._parse_res,
            "volpo": self._parse_volpo,
        }
        zeopp_args = zeopp_args or [
            "-ha",
            "-volpo",
            str(radius_sorbate),
            str(radius_sorbate),
            "50000",
        ]

        output_file_path = ""
        for flag, _func in flag_to_func.items():
            if f"-{flag}" in zeopp_args:
                output_file_path = (
                    os.path.join(self.working_dir, self.cif_name) + f"_{sorbate}.{flag}"
                )
                parse_func = _func

        zeopp_args = [self.zeopp_path, *zeopp_args, output_file_path, self._cif_path]

        with subprocess.Popen(
            zeopp_args,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            close_fds=True,
        ) as proc:
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(
                    f"exit code: {proc.returncode}, error: {stderr!s}.\n"
                    f"stdout: {stdout!s}. Check zeo++ installation."
                )

        output: dict[str, Any] = parse_func(output_file_path)

        if output == {}:
            raise ValueError(
                f"zeopp_args must contain either -res or -volpo, not {zeopp_args}"
            )

        try:
            output["structure"] = Structure.from_file(self._cif_path)
        except (OSError, ValueError) as exc:
            output["structure"] = f"Exception: {exc}"

        file_paths_shared[sorbate] = output_file_path
        output_shared[sorbate] = output

    @staticmethod
    def _parse_volpo(volpo_path: str) -> dict[str, Any]:
        """
        Parse the output from a volpo calculation.

        Parameters
        ----------
        volpo_path : str
            Path to the volpo output file.

        Returns
        -------
        dict[str, Any]
            Parsed output.
        """
        with open(volpo_path) as f:
            data = f.read().split("\n")

        output: dict[str, Any] = {}
        for line in data:
            if "PROBE_OCCUPIABLE" in line:
                continue

            read_value = False
            for token in line.split():
                if ":" in token:
                    key, *_ = token.split(":", 1)
                    read_value = True
                elif read_value:
                    try:
                        value: float | str = float(token)
                    except ValueError:
                        value = token
                    output[key] = value
                    read_value = False
        return output

    @staticmethod
    def _parse_res(res_path: str) -> dict[str, Any]:
        """
        Parse the output from a res calculation.

        Parameters
        ----------
        res_path : str
            Path to the res output file.

        Returns
        -------
        dict[str, Any]
            Parsed output.
        """
        with open(res_path) as f:
            data = f.read().split()
        return {"LCD": float(data[1]), "PLD": float(data[2])}

    @staticmethod
    def get_sorbate_radius(sorbate: str) -> float:
        """
        Get the half of the kinetic diameter for a sorbate.

        Parameters
        ----------
        sorbate : str
            The sorbate species.

        Returns
        -------
        float
            The radius (kinetic diameter / 2) in Angstrom.

        Raises
        ------
        KeyError
            If the sorbate is not known.
        """
        kinetic_diameter = {
            "He": 2.551,
            "Ne": 2.82,
            "Ar": 3.542,
            "Kr": 3.655,
            "Xe": 4.047,
            "H2": 2.8585,
            "D2": 2.8585,
            "N2": 3.72,
            "O2": 3.467,
            "Cl2": 4.217,
            "Br2": 4.296,
            "CO": 3.69,
            "CO2": 3.3,
            "NO": 3.492,
            "N2O": 3.838,
            "SO2": 4.112,
            "COS": 4.130,
            "H2O": 2.641,
            "CH4": 3.758,
            "NH3": 3.62,
            "H2S": 3.623,
        }
        try:
            return kinetic_diameter[sorbate] * 0.5
        except Exception:
            logger.exception("Unknown sorbate %s.", sorbate)
            raise


@job
def run_zeopp_assessment(
    structure: Structure | str,
    zeopp_path: str | None = None,
    working_dir: str | None = None,
    sorbates: list[str] | str | None = None,
    cif_name: str | None = None,
    nproc: int = 1,
    rules: dict[str, Callable[[dict[str, Any]], bool]] | None = None,
) -> dict[str, Any]:
    """
    Run zeo++ on a structure with user-defined rules.

    Parameters
    ----------
    structure : Structure or str
        Either a pymatgen Structure or a path to a CIF file.
    zeopp_path : str, optional
        Path to the zeo++ executable.
    working_dir : str, optional
        Directory for intermediate files.
    sorbates : list[str] or str, optional
        List of sorbate species or a single species.
    cif_name : str, optional
        Filename for the CIF if structure is a Structure.
    nproc : int, optional
        Number of processes to use.
    rules : dict[str, Callable[[dict[str, Any]], bool]], optional
        Mapping of names to functions that take the full output dict
        and return True/False if the structure passes each rule.

    Returns
    -------
    dict[str, Any]
        Zeo++ outputs (per sorbate) and boolean result for the rule.

    Examples
    --------
    Example of custom rules to assess a candidate MOF structure:

    ```python
    from atomate2.common.jobs.mof import run_zeopp_assessment


    def custom_mof_rule(out):
        props = out["N2"]
        keys = ["PLD", "POAV_A^3", "PONAV_A^3"]
        if not all(k in props for k in keys):
            return False
        return props["PLD"] > 3.0


    response = run_zeopp_assessment(
        structure=my_struct,
        sorbates="N2",
        rules={"is_mof": custom_mof_rule},
    )
    # response.output["is_mof"] will be True/False
    ```
    """
    if sorbates is None:
        sorbates = ["N2", "CO2", "H2O"]
    if isinstance(structure, str) and os.path.isfile(structure):
        maker = ZeoPlusPlus(
            cif_path=structure,
            zeopp_path=zeopp_path,
            working_dir=working_dir,
            sorbates=sorbates,
        )
    elif isinstance(structure, Structure):
        cif_name = cif_name or "structure.cif"
        maker = ZeoPlusPlus.from_structure(
            structure=structure,
            cif_path=cif_name,
            zeopp_path=zeopp_path,
            working_dir=working_dir,
            sorbates=sorbates,
        )

    sorbate_list: list[str] = (
        maker.sorbates if isinstance(sorbates, list) else [sorbates]
    )
    output: dict[str, Any] = {s: {} for s in sorbate_list}
    for args in [[], ["-ha", "-res"]]:
        maker.run(zeopp_args=args, nproc=nproc)
        for sorbate in maker.sorbates:
            output[sorbate].update(maker.output[sorbate])

    if rules is not None:
        for name, rule_func in rules.items():
            try:
                output[name] = bool(rule_func(output))
            except Exception as e:  # noqa: BLE001
                output[name] = f"rule_error: {e!s}"

    return output
