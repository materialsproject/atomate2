"""Enums for constants across JDFTx schemas."""

from emmet.core.types.enums import ValueEnum


class JDFTxStatus(ValueEnum):
    """JDFTx Calculation State."""

    SUCCESS = "successful"
    FAILED = "unsuccessful"


class CalcType(ValueEnum):
    """JDFTx calculation type."""

    GGA = "gga"
    GGA_PBE = "gga-PBE"
    GGA_PBESOL = ("gga-PBEsol",)
    GGA_PW91 = ("gga-PW91",)
    HARTREE_FOCK = ("Hartree-Fock",)
    HYB_HSE06 = ("hyb-HSE06",)
    HYB_HSE12 = ("hyb-HSE12",)
    HYB_HSE12S = ("hyb-HSE12s",)
    HYB_PBE0 = ("hyb-PBE0",)
    LDA = ("lda",)
    LDA_PW = ("lda-PW",)
    LDA_PW_PREC = ("lda-PW-prec",)
    LDA_PZ = ("lda-PZ",)
    LDA_TETER = ("lda-Teter",)
    LDA_VWN = ("lda-VWN",)
    MGGA_REVTPSS = ("mgga-revTPSS",)
    MGGA_TPSS = ("mgga-TPSS",)
    ORB_GLLBSC = ("orb-GLLBsc",)
    POT_LB94 = "pot-LB94"


class TaskType(ValueEnum):
    """JDFTx task type."""

    SINGLEPOINT = "Single Point"
    LATTICEOPT = "Lattice Optimization"
    IONOPT = "Ionic Optimization"
    FREQ = "Frequency"
    SOFTSPHERE = "SoftSphere"
    DYNAMICS = "Molecular Dynamics"


class SolvationType(ValueEnum):
    """JDFTx solvent type."""

    NONE = "None"
    SALSA = "SaLSA"
    CDFT = "Classical DFT"
    CANON = "CANON"
    LINEAR_CANDLE = "LinearPCM CANDLE"
    LINEAR_SCCS_ANION = "LinearPCM SCCS_anion"
    LINEAR_SCCS_CATION = "LinearPCM SCCS_anion"
    LINEAR_SCCS_G03 = "LinearPCM SCCS_g03"
    LINEAR_SCCS_G03BETA = "LinearPCM SCCS_g03beta"
    LINEAR_SCCS_G03P = "LinearPCM SCCS_g03p"
    LINEAR_SCCS_G03PBETA = "LinearPCM SCCS_g03pbeta"
    LINEAR_SCCS_G09 = "LinearPCM SCCS_g09"
    LINEAR_SCCS_G09BETA = "LinearPCM SCCS_g09beta"
    LINEAR_SGA13 = "LinearPCM SGA13"
    LINEAR_SOFTSPHERE = "LinearPCM SoftSphere"
    NONLINEAR_SGA13 = "NonlinearPCM SGA13"
