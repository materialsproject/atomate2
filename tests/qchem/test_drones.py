from atomate2.qchem.drones import QChemDrone


def test_structure_optimization(qchem_test_dir):
    drone = QChemDrone()
    doc = drone.assimilate(qchem_test_dir / "water_single_point" / "outputs")
    assert doc


def test_valid_paths(qchem_test_dir):
    drone = QChemDrone()
    valid_paths = drone.get_valid_paths(
        [
            str(qchem_test_dir) + "/water_frequency",
            ["inputs/", "outputs/"],
            ["mol.in", "mol.out"],
        ]
    )
    assert valid_paths
