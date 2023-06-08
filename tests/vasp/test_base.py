from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from atomate2.vasp.sets.base import _get_magmoms


@pytest.mark.parametrize("base_magmoms", [None, {"Co": 0.9, "Fe": 2}])
def test_get_magmoms(base_magmoms: dict[str, float]) -> None:
    magmoms = {"Co": 0.8, "Fe": 2.2}  # dummy magmoms

    # structure with Co that will be assigned magmoms
    structure_with_magmoms = Structure(
        lattice=Lattice.cubic(3),
        species=["Co", "Fe"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    # structure that does not have magmoms but has 'Co'
    structure_without_magmoms = structure_with_magmoms.copy()
    for site in structure_with_magmoms:
        site.properties["magmom"] = magmoms[site.species_string]

    msg = "Co without an oxidation state is initialized as low spin by default"
    # check there are no warnings
    for struct in [structure_with_magmoms, structure_without_magmoms]:
        with pytest.warns(UserWarning) as warns:
            out = _get_magmoms(struct, magmoms=magmoms, base_magmoms=base_magmoms)

        expected_magmoms = list((magmoms or base_magmoms).values())
        assert out == expected_magmoms
        assert len(warns) == 1
        assert str(warns[0].message).startswith(msg)
