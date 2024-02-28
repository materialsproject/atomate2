from atomate2.qchem.run import _DEFAULT_HANDLERS, JobType, run_qchem, should_stop_children
from atomate2.qchem.drones import QChemDrone

def test_stop_children(qchem_test_dir):
    drone = QChemDrone()
    task_doc = drone.assimilate(qchem_test_dir / "water_single_point" / "outputs")
    chk_stop_children = should_stop_children(task_document=task_doc, handle_unsuccessful=False)

    assert chk_stop_children == False



