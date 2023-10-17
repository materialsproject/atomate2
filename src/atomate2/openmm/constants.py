from enum import Enum


class OpenMMConstants(Enum):
    TOPOLOGY_PDD_FILE_NAME: str = "topology_pdb"
    STATE_XML_FILE_NAME: str = "state_xml"
    STATE_REPORT_CSV_FILE_NAME: str = "state_csv"
    SYSTEM_XML_FILE_NAME: str = "system_xml"
    INTEGRATOR_XML_FILE_NAME: str = "integrator_xml"
    CONTENTS_JOSN_FILE_NAME: str = "contents_json"
    TRAJECTORY_DCD_FILE_NAME: str = "trajectory_dcd"
    STATE_REPORT_SCHEMA = [
        "steps",
        "potential_energy",
        "kinetic_energy",
        "total_energy",
        "temperature",
        "volume",
        "density"
    ]
