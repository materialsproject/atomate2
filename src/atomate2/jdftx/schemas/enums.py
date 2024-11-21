"""Enums for constants across JDFTx schemas."""

from emmet.core.utils import ValueEnum


class JDFTxStatus(ValueEnum):
    """JDFTx Calculation State."""

    SUCCESS = "successful"
    FAILED = "unsuccessful"


class CalcType(ValueEnum):
    """JDFTx calculation type."""

    PBE = "PBE"
    HSE = "HSE"


class TaskType(ValueEnum):
    """JDFTx task type."""

    SINGLEPOINT = "Single Point"
    LATTICEOPT = "Lattice Optimization"
    IONOPT = "Ionic Optimization"
    FREQ = "Frequency"
    SOFTSPHERE = "SoftSphere"
    DYNAMICS = "Molecular Dynamics"

class SolvationType(ValueEnum):
    """JDFTx solvent type"""
    
    NONE = "None"
    SALSA = "SaLSA"
    CDFT = "Classical DFT"
    CANON = "CANON"
    LINEAR_CANDLE = "LinearPCM CANDLE"
    LINEAR_SCCS_ANION = "LinearPCM SCCS_anion"
    LINEAR_SCCS_CATION = "LinearPCM SCCS_anion"
    LINEAR_SCCS_ANION = "LinearPCM SCCS_cation"
    LINEAR_SCCS_G03 = "LinearPCM SCCS_g03"
    LINEAR_SCCS_G03BETA = "LinearPCM SCCS_g03beta"
    LINEAR_SCCS_G03P = "LinearPCM SCCS_g03p"
    LINEAR_SCCS_G03PBETA = "LinearPCM SCCS_g03pbeta"
    LINEAR_SCCS_G09 = "LinearPCM SCCS_g09"
    LINEAR_SCCS_G09BETA = "LinearPCM SCCS_g09beta"
    LINEAR_SGA13 = "LinearPCM SGA13"
    LINEAR_SOFTSPHERE = "LinearPCM SoftSphere"
    NONLINEAR_SGA13 = "NonlinearPCM SGA13"



