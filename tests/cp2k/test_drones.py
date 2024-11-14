import re

import pytest


def test_structure_optimization(cp2k_test_dir):
    from atomate2.cp2k.drones import Cp2kDrone
    from atomate2.cp2k.schemas.task import TaskDocument

    drone = Cp2kDrone()
    doc = drone.assimilate(cp2k_test_dir / "Si_band_structure" / "static" / "outputs")
    assert isinstance(doc, TaskDocument)
    assert doc.output.energy == pytest.approx(-197.4, abs=1e-1)
    assert doc.output.forces == pytest.approx(
        [(-1e-08, -1e-08, -1e-08), (2e-08, 2e-08, 2e-08)], abs=1e-1
    )
    assert doc.output.stress == (
        (10.7732045483, -1.12172149101, -0.432646843218),
        (-1.12172149101, 11.8286321403, 2.48811110025),
        (-0.432646843218, 2.48811110025, 8.74606312102),
    )
    assert doc.output.bandgap is None
    assert doc.output.cbm == pytest.approx(7.65104451, abs=1e-1)
    assert doc.output.vbm == pytest.approx(5.09595793, abs=1e-1)
    assert doc.output.structure.lattice.a == pytest.approx(3.8669, abs=1e-1)
    assert re.match(  # example completed_at: 2023-11-18 00:05:01.267082+00:00
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}\+\d{2}:\d{2}$", doc.completed_at
    )
    assert doc.state.value == "successful"
    assert doc.task_label is None
    assert doc.tags is None
    assert {*doc.run_stats} == {"overall", "standard"}
