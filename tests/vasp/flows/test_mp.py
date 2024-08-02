from __future__ import annotations

import pytest
from emmet.core.tasks import TaskDoc
from jobflow import Maker, run_locally
from pymatgen.core import Structure

from atomate2.vasp.flows.mp import (
    MPGGADoubleRelaxMaker,
    MPGGADoubleRelaxStaticMaker,
    MPMetaGGADoubleRelaxStaticMaker,
)
from atomate2.vasp.jobs.mp import MPMetaGGARelaxMaker, MPPreRelaxMaker


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
    flow = MPMetaGGADoubleRelaxStaticMaker(
        relax_maker=relax_maker, static_maker=static_maker, **kwargs
    )
    assert isinstance(flow.relax_maker, type(relax_maker))
    if relax_maker:
        assert flow.relax_maker.name == "MP pre-relax"

    assert isinstance(flow.static_maker, type(static_maker))
    if static_maker:
        assert flow.static_maker.name == "MP meta-GGA relax"

    assert flow.name == (name or "MP meta-GGA relax")


def test_mp_meta_gga_double_relax_static(mock_vasp, clean_dir, vasp_test_dir):
    # map from job name to directory containing reference input/output files
    pre_relax_dir = "Si_mp_meta_gga_relax/pbesol_pre_relax"
    ref_paths = {
        "MP pre-relax 1": pre_relax_dir,
        "MP meta-GGA relax 2": "Si_mp_meta_gga_relax/r2scan_relax",
        "MP meta-GGA static": "Si_mp_meta_gga_relax/r2scan_final_static",
    }
    si_struct = Structure.from_file(f"{vasp_test_dir}/{pre_relax_dir}/inputs/POSCAR.gz")

    mock_vasp(ref_paths)

    # generate flow
    flow = MPMetaGGADoubleRelaxStaticMaker().make(si_struct)

    # ensure flow runs successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # validate output
    task_doc = responses[flow.jobs[-1].uuid][1].output
    assert isinstance(task_doc, TaskDoc)
    assert task_doc.output.energy == pytest.approx(-46.8613738)


def test_mp_gga_double_relax_static(mock_vasp, clean_dir, vasp_test_dir):
    # map from job name to directory containing reference input/output files
    pre_relax_dir = "Si_mp_gga_relax/GGA_Relax_1"
    ref_paths = {
        "MP GGA relax 1": pre_relax_dir,
        "MP GGA relax 2": "Si_mp_gga_relax/GGA_Relax_2",
        "MP GGA static": "Si_mp_gga_relax/GGA_Static",
    }
    si_struct = Structure.from_file(f"{vasp_test_dir}/{pre_relax_dir}/inputs/POSCAR.gz")

    mock_vasp(ref_paths)

    # generate flow
    flow = MPGGADoubleRelaxStaticMaker().make(si_struct)

    # ensure flow runs successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # validate output
    task_doc = responses[flow.jobs[-1].uuid][1].output
    assert isinstance(task_doc, TaskDoc)
    assert task_doc.output.energy == pytest.approx(-10.84060922)


def test_mp_gga_double_relax(mock_vasp, clean_dir, vasp_test_dir):
    # map from job name to directory containing reference input/output files
    pre_relax_dir = "Si_mp_gga_relax/GGA_Relax_1"
    ref_paths = {
        "MP GGA relax 1": pre_relax_dir,
        "MP GGA relax 2": "Si_mp_gga_relax/GGA_Relax_2",
    }
    si_struct = Structure.from_file(f"{vasp_test_dir}/{pre_relax_dir}/inputs/POSCAR.gz")

    mock_vasp(ref_paths)

    # generate flow
    flow = MPGGADoubleRelaxMaker().make(si_struct)

    # ensure flow runs successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # validate output
    task_doc = responses[flow.jobs[-1].uuid][1].output
    assert isinstance(task_doc, TaskDoc)
    assert task_doc.output.energy == pytest.approx(-10.84145656)
