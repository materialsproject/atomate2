from atomate2.qchem.drones import QChemDrone


def test_structure_optimization(qchem_test_dir):
    drone = QChemDrone()
    doc = drone.assimilate(qchem_test_dir / "water_single_point" / "outputs")
    assert doc
