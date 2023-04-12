""" Utilities to determine level of theory, task type, and calculation type for Q-Chem calculations"""
from typing import Any, Dict, Optional
from jobflow.utils import ValueEnum
from atomate2.qchem.schemas.calc_types import LevelOfTheory, CalcType, TaskType
from atomate2.qchem.schemas.calc_types.calc_types import (
    FUNCTIONALS,
    BASIS_SETS,
    SOLVENTS,
    PCM_DIELECTRICS,
    SMD_PARAMETERS,
)


__author__ = "Evan Spotte-Smith <ewcspottesmith@lbl.gov>"


functional_synonyms = {
    "b97mv": "b97m-v",
    "b97mrv": "b97m-rv",
    "wb97xd": "wb97x-d",
    "wb97xd3": "wb97x-d3",
    "wb97xv": "wb97x-v",
    "wb97mv": "wb97m-v",
}

smd_synonyms = {
    "SOLVENT=WATER": "water",
    "SOLVENT=THF": "thf",
    "DIELECTRIC=7,230;N=1,410;ALPHA=0,000;BETA=0,859;GAMMA=36,830;PHI=0,000;PSI=0,000": "diglyme",
    "DIELECTRIC=18,500;N=1,415;ALPHA=0,000;BETA=0,735;GAMMA=20,200;PHI=0,000;PSI=0,143": "3:7 EC:EMC"
}

def level_of_theory(
    parameters: Dict[str, Any], custom_smd: Optional[str] = None
) -> LevelOfTheory:
    """

    Returns the level of theory for a calculation,
    based on the input parameters given to Q-Chem

    Args:
        parameters: Dict of Q-Chem input parameters
        custom_smd: (Optional) string representing SMD parameters for a
        non-stadard solvent

    """

    funct_raw = parameters["rem"].get("method")
    basis_raw = parameters["rem"].get("basis")

    if funct_raw is None or basis_raw is None:
        raise ValueError(
            'Method and basis must be included in "rem" section ' "of parameters!"
        )

    disp_corr = parameters["rem"].get("dft_d")

    if disp_corr is None:
        funct_lower = funct_raw.lower()
        funct_lower = functional_synonyms.get(funct_lower, funct_lower)
    else:
        # Replace Q-Chem terms for D3 tails with more common expressions
        disp_corr = disp_corr.replace("_bj", "(bj)").replace("_zero", "(0)")
        funct_lower = f"{funct_raw}-{disp_corr}"

    basis_lower = basis_raw.lower()

    functional = [f for f in FUNCTIONALS if f.lower() == funct_lower]
    if not functional:
        raise ValueError(f"Unexpected functional {funct_lower}!")

    functional = functional[0]

    basis = [b for b in BASIS_SETS if b.lower() == basis_lower]
    if not basis:
        raise ValueError(f"Unexpected basis set {basis_lower}!")

    basis = basis[0]

    solvent_method = parameters["rem"].get("solvent_method")
    if solvent_method is None:
        solvation = "VACUUM"
    elif solvent_method in ["pcm", "isosvp", "cosmo"]:
        dielectric = float(parameters["solvent"].get("dielectric", 78.39))
        solvent = None
        for s, d in PCM_DIELECTRICS.items():
            if round(d, 2) == round(dielectric, 2):
                solvent = s
                break

        solvation = f"PCM({solvent or str(dielectric)})"
    elif solvent_method == "smd":
        solvent = parameters["smx"].get("solvent", "unknown")

        if solvent == "other":
            if custom_smd is None:
                raise ValueError(
                    "SMD calculation with solvent=other requires custom_smd!"
                )

            match = False
            custom_mod = custom_smd.replace(".0,", ".00,")
            if custom_mod.endswith(".0"):
                custom_mod += "0"
            for s, p in SMD_PARAMETERS.items():
                if p == custom_mod:
                    solvent = s
                    match = True
                    break
            if not match:
                raise ValueError(f"Unknown solvent with SMD parameters {custom_smd}!")
        solvation = f"SMD({solvent.upper()})"
    else:
        raise ValueError(f"Unexpected implicit solvent method {solvent_method}!")

    lot = f"{functional}/{basis}/{solvation}"

    return LevelOfTheory(lot)

def solvent(parameters: Dict[str, Any], custom_smd: Optional[str] = None) -> str:
    """
    Returns the solvent used for this calculation.
    Args:
        parameters: Dict of Q-Chem input parameters
        custom_smd: (Optional) string representing SMD parameters for a
        non-standard solvent
    """

    lot = level_of_theory(parameters)
    solvation = lot.value.split("/")[-1]

    if solvation == "PCM":
        dielectric = float(parameters.get("solvent", {}).get("dielectric", 78.39))
        dielectric_string = f"{dielectric:.2f}".replace(".", ",")
        return f"DIELECTRIC={dielectric_string}"
    # TODO: Add this once added into pymatgen and atomate
    # elif solvation == "ISOSVP":
    #     dielectric = float(parameters.get("svp", {}).get("dielst", 78.39))
    #     rho = float(parameters.get("svp", {}).get("rhoiso", 0.001))
    #     return f"DIELECTRIC={round(dielectric, 2)},RHO={round(rho, 4)}"
    # elif solvation == "CMIRS":
    #     dielectric = float(parameters.get("svp", {}).get("dielst", 78.39))
    #     rho = float(parameters.get("svp", {}).get("rhoiso", 0.001))
    #     a = parameters.get("pcm_nonels", {}).get("a")
    #     b = parameters.get("pcm_nonels", {}).get("b")
    #     c = parameters.get("pcm_nonels", {}).get("c")
    #     d = parameters.get("pcm_nonels", {}).get("d")
    #     solvrho = parameters.get("pcm_nonels", {}).get("solvrho")
    #     gamma = parameters.get("pcm_nonels", {}).get("gamma")
    #
    #     string = f"DIELECTRIC={round(dielectric, 2)},RHO={round(rho, 4)}"
    #     for name, (piece, digits) in {"A": (a, 6), "B": (b, 6), "C": (c, 1), "D": (d, 3),
    #                                   "SOLVRHO": (solvrho, 2), "GAMMA": (gamma, 1)}.items():
    #         if piece is None:
    #             piecestring = "NONE"
    #         else:
    #             piecestring = f"{name}={round(float(piece), digits)}"
    #         string += "," + piecestring
    #     return string
    elif solvation == "SMD":
        solvent = parameters.get("smx", {}).get("solvent", "water")
        if solvent == "other":
            if custom_smd is None:
                raise ValueError(
                    "SMD calculation with solvent=other requires custom_smd!"
                )

            names = ["DIELECTRIC", "N", "ALPHA", "BETA", "GAMMA", "PHI", "PSI"]
            numbers = [float(x) for x in custom_smd.split(",")]

            string = ""
            for name, number in zip(names, numbers):
                string += f"{name}={number:.3f};"
            return string.rstrip(",").rstrip(";").replace(".", ",")
        else:
            return f"SOLVENT={solvent.upper()}"
    else:
        return "NONE"


def lot_solvent_string(
    parameters: Dict[str, Any], custom_smd: Optional[str] = None
) -> str:
    """
    Returns a string representation of the level of theory and solvent used for this calculation.
    Args:
        parameters: Dict of Q-Chem input parameters
        custom_smd: (Optional) string representing SMD parameters for a
        non-standard solvent
    """

    lot = level_of_theory(parameters).value
    solv = solvent(parameters, custom_smd=custom_smd)
    return f"{lot}({solv})"

def task_type(orig: Dict[str, Any], special_run_type: Optional[str] = None) -> TaskType:
    if special_run_type == "frequency_flattener":
        return TaskType("Frequency Flattening Geometry Optimization")
    elif special_run_type == "ts_frequency_flattener":
        return TaskType("Frequency Flattening Transition State Geometry Optimization")

    if orig["rem"].get("job_type") == "sp":
        return TaskType("Single Point")
    elif orig["rem"].get("job_type") == "opt":
        return TaskType("Geometry Optimization")
    elif orig["rem"].get("job_type") == "ts":
        return TaskType("Transition State Geometry Optimization")
    elif orig["rem"].get("job_type") == "freq":
        return TaskType("Frequency Analysis")

    return TaskType("Unknown")


def calc_type(
    special_run_type: str, orig: Dict[str, Any], custom_smd: Optional[str] = None
) -> CalcType:
    """
    Determines the calc type

    Args:
        inputs: inputs dict with an incar, kpoints, potcar, and poscar dictionaries
        parameters: Dictionary of VASP parameters from Vasprun.xml
    """
    rt = level_of_theory(orig, custom_smd=custom_smd).value
    tt = task_type(orig, special_run_type=special_run_type).value
    return CalcType(f"{rt} {tt}")

def get_enum_source(enum_name, doc, items):
    header = f"""
class {enum_name}(ValueEnum):
    \"\"\" {doc} \"\"\"\n
"""
    items = [f'    {const} = "{val}"' for const, val in items.items()]

    return header + "\n".join(items)


