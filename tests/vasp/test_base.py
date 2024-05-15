from __future__ import annotations

import pytest
from custodian.vasp.handlers import ErrorHandler
from pymatgen.core import Lattice, Structure

from atomate2.vasp.run import DEFAULT_HANDLERS
from atomate2.vasp.sets.base import get_magmoms


@pytest.mark.parametrize("magmoms", [None, {"Co": 0.8, "Fe": 2.2}])
@pytest.mark.parametrize("config_magmoms", [None, {"Co": 0.9, "Fe": 2}])
def test_get_magmoms(
    magmoms: dict[str, float], config_magmoms: dict[str, float]
) -> None:
    # structure with Co that will be assigned magmoms
    struct = Structure(
        lattice=Lattice.cubic(3),
        species=["Co", "Fe"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    # structure that does not have magmoms but has 'Co'
    if magmoms:
        for site in struct:
            site.properties["magmom"] = magmoms[site.species_string]

    msg = "Co without an oxidation state is initialized as low spin by default"
    # check there are no warnings
    with pytest.warns(None if magmoms else UserWarning) as warns:
        out = get_magmoms(struct, magmoms=magmoms, config_magmoms=config_magmoms)

        expected_magmoms = list(
            (magmoms or config_magmoms or {"Co": 0.6, "Fe": 0.6}).values()
        )
        assert out == expected_magmoms
        if magmoms is None:
            assert len(warns) == 1
            assert str(warns[0].message).startswith(msg)


def test_get_magmoms_with_specie() -> None:
    # Setting the species to Co2+ and Fe3+ will change the `site.specie` to
    # `Specie` instead of `Element`. This will allow us to test the part of
    # the code that checks for `Specie.spin`.
    # @jmmshn
    struct = Structure(
        lattice=Lattice.cubic(3),
        species=["Co2+", "Fe3+"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    site = struct.sites[0]
    assert not hasattr(site, "magmom")
    assert struct[0].specie.spin is None
    out = get_magmoms(struct)
    assert out == [0.6, 0.6]


def test_default_handlers():
    assert len(DEFAULT_HANDLERS) >= 8
    assert all(isinstance(handler, ErrorHandler) for handler in DEFAULT_HANDLERS)
