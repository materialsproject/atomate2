import pytest
from pymatgen.core import Lattice, Species, Structure
from pymatgen.io.vasp.sets import MPScanRelaxSet

from atomate2.vasp.sets.core import StaticSetGenerator


@pytest.fixture(scope="module")
def struct_no_magmoms() -> Structure:
    """Dummy FeO structure with expected +U corrections but no magnetic moments
    defined."""
    return Structure(
        lattice=Lattice.cubic(3),
        species=("Fe", "O"),
        coords=((0, 0, 0), (0.5, 0.5, 0.5)),
    )


@pytest.fixture(scope="module")
def struct_with_spin() -> Structure:
    """Dummy FeO structure with spins defined."""
    iron = Species("Fe2+", spin=4)
    oxi = Species("O2-", spin=0.63)

    return Structure(
        lattice=Lattice.cubic(3),
        species=(iron, oxi),
        coords=((0, 0, 0), (0.5, 0.5, 0.5)),
    )


@pytest.fixture(scope="module")
def struct_with_magmoms(struct_no_magmoms) -> Structure:
    """Dummy FeO structure with magmoms defined."""
    struct = struct_no_magmoms.copy()
    struct.add_site_property("magmom", [4.7, 0.0])
    return struct


@pytest.fixture(scope="module")
def struct_no_u_params() -> Structure:
    """Dummy SiO structure with no anticipated +U corrections"""
    return Structure(
        lattice=Lattice.cubic(3),
        species=("Si", "O"),
        coords=((0, 0, 0), (0.5, 0.5, 0.5)),
    )


def test_user_incar_settings():
    structure = Structure([[1, 0, 0], [0, 1, 0], [0, 0, 1]], ["H"], [[0, 0, 0]])

    # check to see if user incar settings (even when set to nonsensical values, as done
    # below) are always preserved.
    uis = {
        "ALGO": "VeryFast",
        "EDIFF": 1e-30,
        "EDIFFG": -1e-10,
        "ENAUG": 20_000,
        "ENCUT": 15_000,
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
        "NSW": 5_000,
        "PREC": 10,  # wrong type, should be string.
        "SIGMA": 20,
        "LDAUU": {"H": 5.0},
        "LDAUJ": {"H": 6.0},
        "LDAUL": {"H": 3.0},
        "LDAUTYPE": 2,
    }

    static_set_generator = StaticSetGenerator(user_incar_settings=uis)
    incar = static_set_generator.get_input_set(structure, potcar_spec=True)["INCAR"]

    for key in uis:
        if isinstance(incar[key], str):
            assert incar[key].lower() == uis[key].lower()
        elif isinstance(uis[key], dict):
            assert incar[key] == [uis[key][str(site.specie)] for site in structure]
        else:
            assert incar[key] == uis[key]


@pytest.mark.parametrize(
    "structure,user_incar_settings",
    [
        ("struct_no_magmoms", {}),
        ("struct_with_magmoms", {}),
        ("struct_with_spin", {}),
        ("struct_no_magmoms", {"MAGMOM": {"Fe": 3.7, "O": 0.8}}),
        ("struct_with_magmoms", {"MAGMOM": {"Fe": 3.7, "O": 0.8}}),
        ("struct_with_spin", {"MAGMOM": {"Fe2+,spin=4": 3.7, "O2-,spin=0.63": 0.8}}),
    ],
)
def test_incar_magmoms_precedence(structure, user_incar_settings, request) -> None:
    """
    According to VaspInputGenerator._get_magmoms, the magmoms for a new input set are
    determined given the following precedence:

    1. user incar settings
    2. magmoms in input struct
    3. spins in input struct
    4. job config dict
    5. set all magmoms to 0.6

    Here, we use the StaticSetGenerator as an example, but any input generator that has
    an implemented get_incar_updates() method could be used.
    """
    structure = request.getfixturevalue(structure)

    input_gen = StaticSetGenerator(user_incar_settings=user_incar_settings)
    incar = input_gen.get_input_set(structure, potcar_spec=True)["INCAR"]
    incar_magmom = incar["MAGMOM"]

    has_struct_magmom = structure.site_properties.get("magmom")
    has_struct_spin = getattr(structure.species[0], "spin", None) is not None

    if user_incar_settings:  # case 1
        assert incar_magmom == [
            user_incar_settings["MAGMOM"][str(site.specie)] for site in structure
        ]
    elif has_struct_magmom:  # case 2
        assert incar_magmom == structure.site_properties["magmom"]
    elif has_struct_spin:  # case 3
        assert incar_magmom == [s.spin for s in structure.species]
    else:  # case 4 and 5
        assert incar_magmom == [
            input_gen.config_dict["INCAR"]["MAGMOM"].get(str(s), 0.6)
            for s in structure.species
        ]


@pytest.mark.parametrize("structure", ["struct_no_magmoms", "struct_no_u_params"])
def test_set_u_params(structure, request) -> None:
    structure = request.getfixturevalue(structure)
    input_gen = StaticSetGenerator()
    incar = input_gen.get_input_set(structure, potcar_spec=True)["INCAR"]

    has_nonzero_u = (
        any(
            input_gen.config_dict["INCAR"]["LDAUU"]["O"].get(str(site.specie), 0) > 0
            for site in structure
        )
        and input_gen.config_dict["INCAR"]["LDAU"]
    )

    if has_nonzero_u:
        # if at least one site has a nonzero U value in the config_dict,
        # ensure that there are LDAU* keys, and that they match expected values
        # in config_dict
        assert len([key for key in incar if key.startswith("LDAU")]) > 0
        for ldau_key in ("LDAUU", "LDAUJ", "LDAUL"):
            for idx, site in enumerate(structure):
                assert incar[ldau_key][idx] == input_gen.config_dict["INCAR"][ldau_key][
                    "O"
                ].get(str(site.specie), 0)
    else:
        # if no sites have a nonzero U value in the config_dict,
        # ensure that no keys starting with LDAU are in the INCAR
        assert len([key for key in incar if key.startswith("LDAU")]) == 0


@pytest.mark.parametrize(
    "bandgap, bandgap_tol, expected_params",
    [
        (0, 1.0e-4, {"KSPACING": 0.22, "ISMEAR": 2, "SIGMA": 0.2}),
        (0.1, 1.0e-4, {"KSPACING": 0.26969561, "ISMEAR": -5, "SIGMA": 0.05}),
        (0.1, 0.1, {"KSPACING": 0.22, "ISMEAR": 2, "SIGMA": 0.2}),
        (0.1, 0.2, {"KSPACING": 0.22, "ISMEAR": 2, "SIGMA": 0.2}),
        (1, 1.0e-4, {"KSPACING": 0.30235235, "ISMEAR": -5, "SIGMA": 0.05}),
        (2, 1.0e-4, {"KSPACING": 0.34935513, "ISMEAR": -5, "SIGMA": 0.05}),
        (5, 1.0e-4, {"KSPACING": 0.44, "ISMEAR": -5, "SIGMA": 0.05}),
        (10, 1.0e-4, {"KSPACING": 0.44, "ISMEAR": -5, "SIGMA": 0.05}),
    ],
)
def test_set_kspacing_bandgap_tol_and_auto_ismear(
    struct_no_magmoms, bandgap, bandgap_tol, expected_params, monkeypatch
):
    static_set = MPScanRelaxSet(
        auto_ismear=True,
        auto_kspacing=True,
        structure=struct_no_magmoms,
        bandgap=bandgap,
        bandgap_tol=bandgap_tol,
    )

    incar = static_set.incar

    actual = {key: incar[key] for key in expected_params}
    assert actual == pytest.approx(expected_params)
