""" Test common MPMorph features / jobs."""

from jobflow import run_locally
from monty.serialization import loadfn
from pymatgen.core import Composition
import pytest
from shutil import which

from atomate2.common.jobs.mpmorph import (
    _DEFAULT_ICSD_AVG_VOL_FILE,
    _get_chem_env_key_from_composition,
    get_average_volume_from_icsd,
    get_random_packed_structure,
)


def test_get_average_volume_from_icsd():

    avg_vols = loadfn(_DEFAULT_ICSD_AVG_VOL_FILE)

    comp = Composition({"Ag+": 4, "Cu2+": 2, "O2-": 4})

    ref_vols = {0: 13.246317640839177, 1: 12.797510860470629}
    for ignore_oxi in [True,False]:
        if ignore_oxi:
            chem_env = "Ag__Cu__O"
        else:
            chem_env = "Ag+__Cu2+__O2-"
        
        assert _get_chem_env_key_from_composition(comp,ignore_oxi_states=ignore_oxi) == chem_env
        assert chem_env in avg_vols[("without" if ignore_oxi else "with") + "_oxi"]
        assert get_average_volume_from_icsd(comp, ignore_oxi_states = ignore_oxi) == pytest.approx(ref_vols[ignore_oxi])

    comp = Composition({"Ag+": 2, "Cu2+": 2, "Cl-": 6})
    ref_vols = {0: 19.471460609572503, 1: 18.989196360342223}
    for ignore_oxi in [True,False]:
        if ignore_oxi:
            chem_env = "Ag__Cl__Cu"
        else:
            chem_env = "Ag+__Cl-__Cu2+"
        assert _get_chem_env_key_from_composition(comp,ignore_oxi_states=ignore_oxi) == chem_env
        assert get_average_volume_from_icsd(comp, ignore_oxi_states = ignore_oxi) == pytest.approx(ref_vols[ignore_oxi])

@pytest.mark.skipif(
    which("packmol") is None,
    reason = "packmol must be installed to run this test."
)
def test_get_random_packed_structure(test_dir):

    comp = Composition({"Mg2+": 6, "Si4+": 3, "O2-": 12,})
    ref_struct = loadfn(test_dir / "structures" / "packmol_123456.json.gz")

    kwargs = {
        "composition" : comp,
        "target_atoms" : 64,
        "vol_per_atom_source": "icsd",
        "packmol_seed": 123456

    }
    random_struct = get_random_packed_structure(**kwargs)
    assert random_struct == ref_struct

    random_struct_job = get_random_packed_structure(**kwargs, return_as_job=True)
    response = run_locally(random_struct_job)
    random_struct = response[random_struct_job.uuid][1].output
    assert random_struct == ref_struct