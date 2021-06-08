from monty.json import jsanitize


def test_structure_optimization(vasp_test_dir):
    from atomate2.vasp.drones import VaspDrone

    drone = VaspDrone()
    doc = drone.assimilate(
        vasp_test_dir / "Si_structure_optimization_double" / "outputs"
    )
    from pprint import pprint

    pprint(jsanitize(doc, strict=True, allow_bson=True))
