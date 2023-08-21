from pymatgen.core import Structure

from atomate2.vasp.sets.core import StaticSetGenerator


def test_user_incar_settings():
    structure = Structure([[1, 0, 0], [0, 1, 0], [0, 0, 1]], ["H"], [[0, 0, 0]])

    # check to see if user incar settings (even when set to nonsensical values, as done
    # below) are always preserved.
    uis = {
        "ALGO": "VeryFast",
        "EDIFF": 1e-30,
        "EDIFFG": -1e-10,
        "ENAUG": 20000,
        "ENCUT": 15000,
        "GGA": "PE",
        "IBRION": 1,
        "ISIF": 1,
        "ISPIN": False,  # wrong type, should be integer (only 1 or 2)
        "LASPH": False,
        "ISMEAR": -2,
        "LCHARG": 50,  # wrong type, should be bool.
        "LMIXTAU": False,
        "LORBIT": 14,
        "LREAL": "On",
        "MAGMOM": {"H": 100},
        "NELM": 5,
        "NELMIN": 10,  # should not be greater than NELM
        "NSW": 5000,
        "PREC": 10,  # wrong type, should be string.
        "SIGMA": 20,
    }

    static_set_generator = StaticSetGenerator(user_incar_settings=uis)
    incar = static_set_generator.get_input_set(structure).incar

    for key in uis:
        if isinstance(incar[key], str):
            assert incar[key].lower() == uis[key].lower()
        else:
            assert incar[key] == uis[key]
