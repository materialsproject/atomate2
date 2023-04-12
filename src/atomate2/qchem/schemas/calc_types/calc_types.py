"""Task types and level of theory components for Q-Chem calculations"""


__author__ = "Evan Spotte-Smith <ewcspottesmith@lbl.gov>"


TASK_TYPES = [
    "Single Point",
    "Force",
    "Geometry Optimization",
    "Frequency Analysis",
    "Frequency Flattening Geometry Optimization",
    "Transition State Geometry Optimization",
    "Frequency Flattening Transition State Geometry Optimization",
    "Unknown",
]

FUNCTIONAL_CLASSES = {
    "gga": [
        "PBE",
        # "PBE-D3(BJ)",
        # "BLYP",
        # "BLYP-D3(BJ)",
        "B97-D",
        "B97-D3",
        # "mPW91",
        # "mPW91-D3(BJ)",
        # "VV10",
        # "rVV10"
    ],
    "meta-gga": [
        # "M06-L",
        # "M06-L-D3(0)",
        # "SCAN",
        # "SCAN-D3(BJ)",
        # "TPSS",
        # "TPSS-D3(BJ)",
        # "MN12-L",
        # "MN12-L-D3(BJ)",
        "B97M-rV"
    ],
    "hybrid-gga": [
        # "PBE0",
        # "PBE0-D3(BJ)",
        "B3LYP",
        # "B3LYP-D3(BJ)",
        # "CAM-B3LYP",
        # "CAM-B3LYP-D3(0)",
        # "mPW1PW91",
        # "mPW1PW91-D3(BJ)",
        # "wB97X",
        "wB97X-D",
        "wB97X-D3",
        "wB97X-V",
    ],
    "hybrid-meta-gga": [
        # "M06-2X",
        # "M06-2X-D3(0)",
        # "M06-HF",
        # "M08-SO",
        # "M11",
        # "MN15",
        # "BMK",
        # "BMK-D3(BJ)",
        # "TPSSh",
        # "TPSSh-D3(BJ)",
        # "SCAN0",
        # "mPWB1K",
        # "mPWB1K-D3(BJ)",
        "wB97M-V"
    ],
}

FUNCTIONALS = [
    rt
    for functional_class in FUNCTIONAL_CLASSES
    for rt in FUNCTIONAL_CLASSES[functional_class]
]

BASIS_SETS = [
    "6-31g*",
    "def2-SVPD",
    "def2-TZVP",
    "def2-TZVPD",
    "def2-TZVPP",
    "def2-TZVPPD",
    "def2-QZVPD",
    "def2-QZVPPD",
]

SOLVENT_MODELS = ["VACUUM", "PCM", "ISOSVP", "SMD", "CMIRS"]

SOLVENTS = ["WATER", "THF", "G2", "DME", "EC", "EC/EMC", "PC", "DMSO", "ACETONITRILE", "BENZENE", "METHANOL"]

PCM_DIELECTRICS = {
    "WATER": 78.39,
    "THF": 7.43,
    "G2": 7.23,
    "DME": 7.2,
    "EC": 95.3,
    "EC/EMC": 18.5,
    "PC": 65.5,
}

SMD_PARAMETERS = {
    "G2": "7.23,1.4097,0.00,0.859,36.83,0.00,0.00",
    "DME": "7.2,1.379,0.00,0.651,30.8,0.00,0.00",
    "EC": "90.0,1.415,0.00,0.735,20.2,0.00,0.00",
    "EC/EMC": "18.5,1.415,0.00,0.735,20.2,0.00,0.00",
    "PC": "64.9,1.421,0.0,0.730,29.13,0.0,0.0",
}
