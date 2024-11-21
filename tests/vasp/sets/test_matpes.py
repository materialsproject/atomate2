"""Confirm with @janosh before changing any of the expected values below."""

import pytest
from pymatgen.io.vasp.sets import MatPESStaticSet

from atomate2.vasp.sets.base import VaspInputGenerator
from atomate2.vasp.sets.matpes import (
    MatPesGGAStaticSetGenerator,
    MatPesMetaGGAStaticSetGenerator,
)


@pytest.mark.parametrize(
    "set_generator",
    [MatPesGGAStaticSetGenerator, MatPesMetaGGAStaticSetGenerator],
)
def test_matpes_sets(set_generator: VaspInputGenerator) -> None:
    with pytest.warns(FutureWarning):
        matpes_set: VaspInputGenerator = set_generator()

    assert {*matpes_set.as_dict()} >= {
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
    assert matpes_set.potcar_functional == "PBE_64"
    assert isinstance(matpes_set.inherit_incar, list | tuple)
    assert set(matpes_set.inherit_incar) == set(MatPESStaticSet.inherit_incar)
    assert matpes_set.auto_ismear is False
    assert matpes_set.auto_kspacing is False
    assert matpes_set.auto_lreal is False
    assert matpes_set.sort_structure is True
    assert matpes_set.sym_prec == 0.1
    assert matpes_set.use_structure_charge is False
    assert matpes_set.vdw is None
