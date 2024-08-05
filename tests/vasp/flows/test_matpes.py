from __future__ import annotations

import pytest
from emmet.core.tasks import TaskDoc
from jobflow import run_locally
from pymatgen.core import Structure

from atomate2.vasp.flows.matpes import MatPesStaticFlowMaker


def test_matpes_static_flow_maker(mock_vasp, clean_dir, vasp_test_dir):
    # map from job name to directory containing reference input/output files
    pre_relax_dir = "matpes_static_flow/pbe_static"
    ref_paths = {
        "MatPES GGA static": pre_relax_dir,
        "MatPES meta-GGA static": "matpes_static_flow/r2scan_static",
    }
    si_struct = Structure.from_file(f"{vasp_test_dir}/{pre_relax_dir}/inputs/POSCAR.gz")

    mock_vasp(ref_paths)

    # generate flow
    flow = MatPesStaticFlowMaker().make(si_struct)

    assert flow.name == "MatPES static flow"
    assert len(flow) == 2
    assert [job.name for job in flow] == list(ref_paths)

    # make sure first static has LWAVE=True so we can pass its WAVECAR to second static
    # as pre-conditioned starting point
    assert flow[0].maker.input_set_generator.user_incar_settings["LWAVE"] is True

    # ensure flow runs successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # validate output
    pbe_doc = responses[flow.jobs[0].uuid][1].output
    assert isinstance(pbe_doc, TaskDoc)
    assert pbe_doc.output.energy == pytest.approx(-10.84940729)
    assert pbe_doc.output.bandgap == pytest.approx(0.6172, abs=1e-3)

    r2scan_doc = responses[flow.jobs[-1].uuid][1].output
    assert isinstance(r2scan_doc, TaskDoc)
    assert r2scan_doc.output.energy == pytest.approx(-17.53895666)
    assert r2scan_doc.output.bandgap == pytest.approx(0.8087999)

    assert isinstance(flow.output, dict)
    assert {*flow.output} == {"static1", "static2"}
