"""Module to define various calculation types as Enums for VASP."""

from pathlib import Path
from typing import Dict, Literal

from monty.serialization import loadfn

from atomate2.vasp.schemas.calc_types.enums import CalcType, RunType, TaskType

_RUN_TYPE_DATA = loadfn(str(Path(__file__).parent.joinpath("run_types.yaml").resolve()))

__all__ = ["run_type", "task_type", "calc_type"]


def run_type(vasp_parameters: Dict) -> RunType:
    """
    Determine run_type from the VASP parameters dict.

    This is adapted from pymatgen to be far less unstable

    Parameters
    ----------
    vasp_parameters
        Dictionary of VASP parameters from vasprun.xml.

    Returns
    -------
    RunType
        The run type.
    """
    if vasp_parameters.get("LDAU", False):
        is_hubbard = "+U"
    else:
        is_hubbard = ""

    def _variant_equal(v1, v2) -> bool:
        """Check two strings equal."""
        if isinstance(v1, str) and isinstance(v2, str):
            return v1.strip().upper() == v2.strip().upper()
        else:
            return v1 == v2

    # This is to force an order of evaluation
    for functional_class in ["HF", "VDW", "METAGGA", "GGA"]:
        for special_type, params in _RUN_TYPE_DATA[functional_class].items():
            if all(
                [
                    _variant_equal(vasp_parameters.get(param, None), value)
                    for param, value in params.items()
                ]
            ):
                return RunType(f"{special_type}{is_hubbard}")

    return RunType(f"LDA{is_hubbard}")


def task_type(
    inputs: Dict[Literal["incar", "poscar", "kpoints", "potcar"], Dict]
) -> TaskType:
    """
    Determine the task from vasp inputs.

    Parameters
    ----------
    inputs
        Inputs dict with an incar, kpoints, potcar, and poscar dictionaries.

    Returns
    -------
    TaskType
        The task type.
    """
    acalc_type = []
    incar = inputs.get("incar", {})

    if incar.get("ICHARG", 0) > 10:
        try:
            kpts = inputs.get("kpoints") or {}
            kpt_labels = kpts.get("labels") or []
            num_kpt_labels = len(list(filter(None.__ne__, kpt_labels)))
        except Exception as e:
            raise Exception(f"Couldn't identify total number of kpt labels: {e}")

        if num_kpt_labels > 0:
            acalc_type.append("NSCF Line")
        else:
            acalc_type.append("NSCF Uniform")

    elif incar.get("LEPSILON", False):
        if incar.get("IBRION", 0) > 6:
            acalc_type.append("DFPT")
        acalc_type.append("Dielectric")

    elif incar.get("IBRION", 0) > 6:
        acalc_type.append("DFPT")

    elif incar.get("LCHIMAG", False):
        acalc_type.append("NMR Nuclear Shielding")

    elif incar.get("LEFG", False):
        acalc_type.append("NMR Electric Field Gradient")

    elif incar.get("NSW", 1) == 0:
        acalc_type.append("Static")

    elif incar.get("ISIF", 2) == 3 and incar.get("IBRION", 0) > 0:
        acalc_type.append("Structure Optimization")

    elif incar.get("ISIF", 3) == 2 and incar.get("IBRION", 0) > 0:
        acalc_type.append("Deformation")

    if len(acalc_type) == 0:
        return TaskType("Unrecognized")

    return TaskType(" ".join(acalc_type))


def calc_type(
    inputs: Dict[Literal["incar", "poscar", "kpoints", "potcar"], Dict],
    vasp_parameters: Dict,
) -> CalcType:
    """
    Determine the calc type.

    Parameters
    ----------
    inputs
        Inputs dict with an incar, kpoints, potcar, and poscar dictionaries.
    vasp_parameters
        Dictionary of VASP parameters from vasprun.xml.

    Returns
    -------
    CalcType
        The calculation type.
    """
    rt = run_type(vasp_parameters).value
    tt = task_type(inputs).value
    return CalcType(f"{rt} {tt}")
