"""Module to define various calculation types as Enums for CP2K."""

from collections.abc import Iterable, Sequence
from pathlib import Path

from monty.serialization import loadfn
from pymatgen.io.cp2k.inputs import Keyword, KeywordList

from atomate2.cp2k.schemas.calc_types import CalcType, RunType, TaskType

_RUN_TYPE_DATA = loadfn(str(Path(__file__).parent.joinpath("run_types.yaml").resolve()))


def run_type(inputs: dict) -> RunType:
    """
    Determine the run_type from the CP2K input dict.

    This is adapted from pymatgen to be far less unstable.

    Args:
        dft: dictionary of DFT parameters (standard from task doc)
    """
    dft = inputs.get("dft")

    def _variant_equal(v1: Sequence, v2: Sequence) -> bool:
        """Determine if two run_types are equal."""
        if isinstance(v1, str) and isinstance(v2, str):
            return v1.strip().upper() == v2.strip().upper()
        if isinstance(v1, Iterable) and isinstance(v2, Iterable):
            return set(v1) == set(v2)
        return v1 == v2

    is_hubbard = "+U" if dft.get("dft_plus_u") else ""
    vdw = f"-{dft.get('vdw')}" if dft.get("vdw") else ""

    parameters = {
        "FUNCTIONAL": dft.get("functional"),
        "INTERACTION_POTENTIAL": dft.get("hfx", {}).get("Interaction_Potential"),
        "FRACTION": dft.get("hfx", {}).get("FRACTION", 0),
    }

    # Standard calc will only have one functional. If there are multiple functionals
    # used this is either a hybrid calc or a non-generic mixed calculation.
    if len(parameters["FUNCTIONAL"]) == 1:
        parameters["FUNCTIONAL"] = parameters["FUNCTIONAL"][0]

    # If all parameters in for the functional_class.special_type located in
    # run_types.yaml are met, then that is the run type.
    for functional_class in _RUN_TYPE_DATA:
        for special_type, params in _RUN_TYPE_DATA[functional_class].items():
            if all(
                _variant_equal(parameters.get(param, True), value)
                for param, value in params.items()
            ):
                return RunType(f"{special_type}{vdw}{is_hubbard}")

    # TODO elegant way to handle this?
    # This is a hack to get the non-standard hybrids to work
    if parameters.get("FRACTION"):
        return RunType(f"HYBRID{vdw}{is_hubbard}")

    return RunType(f"LDA{is_hubbard}")


def task_type(inputs: dict) -> TaskType:
    """
    Determine the task type.

    Args:
        inputs: Input dictionary
    """
    calc_type = []
    cp2k_run_type = inputs.get("cp2k_global", {}).get("Run_type", "")
    ci = inputs["cp2k_input"]

    if cp2k_run_type.upper() in (
        "ENERGY",
        "ENERGY_FORCE",
        "WAVEFUNCTION_OPTIMIZATION",
        "WFN_OPT",
    ):
        if ci.check("FORCE_EVAL/DFT/SCF"):
            tmp = ci["force_eval"]["dft"]["scf"].get("MAX_SCF", Keyword("", 50))
            if tmp.values[0] == 1:
                if ci.check("force_eval/dft/print/band_structure"):
                    kpt_sets = ci.by_path("force_eval/dft/print/band_structure").get(
                        "kpoint_set", []
                    )
                    spcl = kpt_sets.get("SPECIAL_POINT")
                    label = (
                        spcl[0].values[0]
                        if isinstance(spcl, KeywordList)
                        else spcl.values[0]
                    )
                    if label is not None:
                        calc_type.append("NSCF Line")
                    else:
                        calc_type.append("NSCF Uniform")
            else:
                calc_type.append("Static")
        else:
            calc_type.append("Static")

    elif cp2k_run_type.upper() in ("GEO_OPT", "GEOMETRY_OPTIMIZATION", "CELL_OPT"):
        calc_type.append("Structure Optimization")

    elif cp2k_run_type.upper() == "BAND":
        calc_type.append("Band")

    elif cp2k_run_type.upper() in ("MOLECULAR_DYNAMICS", "MD"):
        calc_type.append("Molecular Dynamics")

    elif cp2k_run_type.upper() in ("MONTE_CARLO", "MC", "TMC", "TAMC"):
        calc_type.append("Monte Carlo")

    elif cp2k_run_type.upper() in ("LINEAR_RESPONSE", "LR"):
        calc_type.append("Linear Response")

    elif cp2k_run_type.upper() in ("VIBRATIONAL_ANALYSIS", "NORMAL_MODES"):
        calc_type.append("Vibrational Analysis")

    elif cp2k_run_type.upper() in ("ELECTRONIC_SPECTRA", "SPECTRA"):
        calc_type.append("Electronic Spectra")

    elif cp2k_run_type.upper() == "NEGF":
        calc_type.append("Non-equilibrium Green's Function")

    elif cp2k_run_type.upper() in ("PINT", "DRIVER"):
        calc_type.append("Path Integral")

    elif cp2k_run_type.upper() in ("RT_PROPAGATION", "EHRENFEST_DYN"):
        calc_type.append("Real-time propagation")

    elif cp2k_run_type.upper() == "BSSE":
        calc_type.append("Base set superposition error")

    elif cp2k_run_type.upper() == "DEBUG":
        calc_type.append("Debug analysis")

    elif cp2k_run_type.upper() == "NONE":
        calc_type.append("None")

    return TaskType(" ".join(calc_type))


def calc_type(
    inputs: dict,
) -> CalcType:
    """
    Determine the calc type.

    Args:
        inputs: dict from InputSummary containing necessary data for determining
            calc type
    """
    rt = run_type(inputs).value
    tt = task_type(inputs).value
    return CalcType(f"{rt} {tt}")
