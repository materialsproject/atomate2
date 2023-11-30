def test_structure_optimization(cp2k_test_dir):
    from atomate2.cp2k.drones import Cp2kDrone

    drone = Cp2kDrone()
    doc = drone.assimilate(cp2k_test_dir / "Si_band_structure" / "static" / "outputs")
    assert doc
