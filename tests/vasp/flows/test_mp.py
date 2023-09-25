import pytest
from jobflow import Maker
from pymatgen.core import Structure

from atomate2.vasp.flows.mp import MPMetaGGADoubleRelaxStatic
from atomate2.vasp.jobs.mp import (
    MPMetaGGARelaxMaker,
    MPPreRelaxMaker,
)


@pytest.mark.parametrize("name", ["test", None])
@pytest.mark.parametrize(
    "relax_maker, static_maker",
    [
        (MPPreRelaxMaker(), MPMetaGGARelaxMaker()),
        (MPPreRelaxMaker(), None),
        (None, MPMetaGGARelaxMaker()),
        (None, None),  # shouldn't raise without optional makers
    ],
)
def test_mp_meta_gga_relax_custom_values(
    name: str, relax_maker: Maker | None, static_maker: Maker | None
):
    kwargs = {}
    if name:
        kwargs["name"] = name
    flow = MPMetaGGADoubleRelaxStatic(
        relax_maker=relax_maker, static_maker=static_maker, **kwargs
    )
    assert isinstance(flow.relax_maker, type(relax_maker))
    if relax_maker:
        assert flow.relax_maker.name == "MP pre-relax"

    assert isinstance(flow.static_maker, type(static_maker))
    if static_maker:
        assert flow.static_maker.name == "MP meta-GGA relax"

    assert flow.name == (name or "MP meta-GGA relax")


def test_mp_meta_gga_relax(mock_vasp, clean_dir, vasp_test_dir):
    from emmet.core.tasks import TaskDoc
    from jobflow import run_locally

    # map from job name to directory containing reference output files
    pre_relax_dir = "Si_mp_metagga_relax/pbesol_pre_relax"
    ref_paths = {
        "MP pre-relax 1": pre_relax_dir,
        "MP meta-GGA relax 2": "Si_mp_metagga_relax/r2scan_relax",
        "MP meta-GGA static": "Si_mp_metagga_relax/r2scan_final_static",
    }
    si_struct = Structure.from_file(f"{vasp_test_dir}/{pre_relax_dir}/inputs/POSCAR")

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        key: {"incar_settings": ["LWAVE", "LCHARG"]} for key in ref_paths
    }

    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # generate flow
    flow = MPMetaGGADoubleRelaxStatic().make(si_struct)

    # ensure flow runs successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # validate output
    output = responses[flow.jobs[-1].uuid][1].output
    assert isinstance(output, TaskDoc)
    assert output.output.energy == pytest.approx(-10.85043620)
