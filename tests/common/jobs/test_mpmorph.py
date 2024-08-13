"""Test common MPMorph features / jobs."""

from shutil import which

import pytest
from jobflow import run_locally
from monty.serialization import loadfn
from pandas import read_json
from pymatgen.core import Composition

from atomate2.common.jobs.mpmorph import (
    _DEFAULT_ICSD_AVG_VOL_FILE,
    _DEFAULT_MP_AVG_VOL_FILE,
    _get_chem_env_key_from_composition,
    get_average_volume_from_icsd,
    get_average_volume_from_mp,
    get_random_packed_structure,
)


@pytest.mark.parametrize(
    "db, ignore_oxi_states", [("icsd", [True, False]), ("mp", [True])]
)
def test_get_average_volume_from_icsd(db: str, ignore_oxi_states: list[bool]):
    comp = Composition({"Ag+": 1, "Cl5+": 1, "O2-": 3})

    if db == "icsd":
        avg_vols = read_json(_DEFAULT_ICSD_AVG_VOL_FILE)
        ref_vols = {0: 14.204405661000004, 1: 14.244954961061925}
        get_avg_vol_func = get_average_volume_from_icsd
    elif db == "mp":
        avg_vols = read_json(_DEFAULT_MP_AVG_VOL_FILE)
        ref_vols = {1: 17.845894151307604}
        get_avg_vol_func = get_average_volume_from_mp
    else:
        raise ValueError(f"Unknown database {db}")

    for ignore_oxi in ignore_oxi_states:
        chem_env = "Ag__Cl__O" if ignore_oxi else "Ag+__Cl5+__O2-"

        kwargs = {"ignore_oxi_states": ignore_oxi} if db == "icsd" else {}
        assert (
            _get_chem_env_key_from_composition(comp, ignore_oxi_states=ignore_oxi)
            == chem_env
        )
        assert (
            len(
                avg_vols[
                    (avg_vols["chem_env"] == chem_env)
                    & (~avg_vols["with_oxi"] if ignore_oxi else avg_vols["with_oxi"])
                ]
            )
            > 0
        )

        assert get_avg_vol_func(comp, **kwargs) == pytest.approx(ref_vols[ignore_oxi])

    comp = Composition({"Ag+": 1, "Cu2+": 1, "Cl-": 3})
    if db == "icsd":
        ref_vols = {0: 19.471460609572503, 1: 18.989196360342223}
    elif db == "mp":
        ref_vols = {1: 21.972236147327344}

    for ignore_oxi in ignore_oxi_states:
        chem_env = "Ag__Cl__Cu" if ignore_oxi else "Ag+__Cl-__Cu2+"
        kwargs = {"ignore_oxi_states": ignore_oxi} if db == "icsd" else {}
        assert (
            _get_chem_env_key_from_composition(comp, ignore_oxi_states=ignore_oxi)
            == chem_env
        )
        assert get_avg_vol_func(comp, **kwargs) == pytest.approx(ref_vols[ignore_oxi])


@pytest.mark.skipif(
    which("packmol") is None, reason="packmol must be installed to run this test."
)
def test_get_random_packed_structure(test_dir):
    comp = Composition({"Mg2+": 6, "Si4+": 3, "O2-": 12})
    ref_struct = loadfn(test_dir / "structures" / "packmol_123456.json.gz")

    kwargs = {
        "composition": comp,
        "target_atoms": 64,
        "vol_per_atom_source": "icsd",
        "packmol_seed": 123456,
    }
    random_struct = get_random_packed_structure(**kwargs)
    assert random_struct == ref_struct

    random_struct_job = get_random_packed_structure(**kwargs, return_as_job=True)
    response = run_locally(random_struct_job)
    random_struct = response[random_struct_job.uuid][1].output
    assert random_struct == ref_struct
