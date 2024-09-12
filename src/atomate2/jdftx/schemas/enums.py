"""Enums for constants across JDFTx schemas."""

from emmet.core.utils import ValueEnum

class JDFTxStatus(ValueEnum):
    """JDFTx Calculation State."""
    SUCCESS = "successful"
    FAILED = "unsuccessful"

class CalcType(ValueEnum):
    """JDFTx calculation type."""
    PBE_CANDLE = "PBE CANDLE"
    PBE_VACUUM = "PBE Vacuum"
    PBE_SALSA = "PBE SaLSA"
    PBE_GLSSA13 = "PBE GLSSA13"

class TaskType(ValueEnum):
    """JDFTx task type."""
    SINGLEPOINT = "Single Point"
    GEOMOPT = "Geometry Optimization"
    FREQ = "Frequency"
    DYNAMICS = "Molecular Dynamics"