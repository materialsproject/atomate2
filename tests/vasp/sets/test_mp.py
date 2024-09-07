"""Confirm with @janosh before changing any of the expected values below."""

import pytest

from atomate2.vasp.sets.base import VaspInputGenerator
from atomate2.vasp.sets.mp import (
    MPGGARelaxSetGenerator,
    MPGGAStaticSetGenerator,
    MPMetaGGARelaxSetGenerator,
    MPMetaGGAStaticSetGenerator,
)


@pytest.mark.parametrize(
    "set_generator",
    [
        MPGGAStaticSetGenerator,
        MPMetaGGAStaticSetGenerator,
        MPGGARelaxSetGenerator,
        MPMetaGGARelaxSetGenerator,
    ],
)
def test_mp_sets(set_generator: VaspInputGenerator) -> None:
    with pytest.warns(FutureWarning):
        mp_set: VaspInputGenerator = set_generator()
    assert {*mp_set.as_dict()} >= {
        "@class",
        "@module",
        "@version",
        "auto_ismear",
        "auto_ispin",
        "auto_kspacing",
        "auto_lreal",
        "auto_metal_kpoints",
        "config_dict",
        "constrain_total_magmom",
        "force_gamma",
        "inherit_incar",
        "sort_structure",
        "sym_prec",
        "use_structure_charge",
        "user_incar_settings",
        "user_kpoints_settings",
        "user_potcar_functional",
        "user_potcar_settings",
        "validate_magmom",
        "vdw",
    }
    assert (
        mp_set.potcar_functional == "PBE_54"
        if "Meta" in set_generator.__name__
        else "PBE"
    )
    assert mp_set.inherit_incar is False
    assert mp_set.auto_ismear is False
    assert mp_set.auto_kspacing is ("Meta" in set_generator.__name__)
    assert mp_set.auto_lreal is False
    assert mp_set.auto_metal_kpoints is ("Meta" not in set_generator.__name__)
    assert mp_set.force_gamma is ("Meta" not in set_generator.__name__)
    assert mp_set.sort_structure is True
    assert mp_set.sym_prec == 0.1
    assert mp_set.use_structure_charge is False
    assert mp_set.vdw is None
    bandgap_tol = getattr(mp_set, "bandgap_tol", None)
    assert bandgap_tol == (1e-4 if "Meta" in set_generator.__name__ else None)
