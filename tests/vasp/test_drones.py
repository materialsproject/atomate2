from atomate2.vasp.drones import VaspDrone


def test_structure_optimization(vasp_test_dir):
    drone = VaspDrone()
    doc = drone.assimilate(vasp_test_dir / "Si_band_structure" / "static" / "outputs")
    assert doc
