import pytest

from atomate2.qchem.drones import QChemDrone
from atomate2.qchem.run import should_stop_children


def test_stop_children_val_td(qchem_test_dir):
    drone = QChemDrone()
    task_doc = drone.assimilate(qchem_test_dir / "water_single_point" / "outputs")
    chk_stop_children = should_stop_children(
        task_document=task_doc, handle_unsuccessful=False
    )

    assert isinstance(chk_stop_children, bool)
    with pytest.raises(RuntimeError) as exc_info:
        should_stop_children(task_document=task_doc, handle_unsuccessful="error")

    error_message = "Job was successful but children jobs need to be stopped!"

    assert str(exc_info.value) == error_message


def test_stop_children_inval_td(qchem_test_dir):
    drone = QChemDrone()
    task_doc = drone.assimilate(qchem_test_dir / "failed_qchem_task_dir" / "outputs")

    with pytest.raises(RuntimeError) as exc_info:
        should_stop_children(task_document=task_doc, handle_unsuccessful="error")

    error_message = (
        "Job was not successful "
        "(perhaps your job did not converge within the "
        "limit of electronic/ionic iterations)!"
    )

    assert str(exc_info.value) == error_message
