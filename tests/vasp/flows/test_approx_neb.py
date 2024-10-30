"""Test ApproxNEB workflow."""

from pathlib import Path

from jobflow import run_locally
from monty.serialization import loadfn

from atomate2.vasp.flows.approx_neb import ApproxNEBMaker


def test_approx_neb_flow(mock_vasp, clean_dir, vasp_test_dir):
    flow_input = loadfn(Path(vasp_test_dir) / "ApproxNEB/approx_neb_input.json.gz")

    flow = ApproxNEBMaker().make(
        *[
            flow_input[k]
            for k in (
                "host_structure",
                "working_ion",
                "inserted_coords_dict",
                "inserted_coords_combo",
            )
        ]
    )

    ref_paths = {
        job_name: f"ApproxNeb/{job_name.replace(' ','_')}"
        for job_name in [
            "host structure relax 1",
            "host structure relax 2",
            "ApproxNEB image relax endpoint 0",
            "ApproxNEB image relax endpoint 1",
            "ApproxNEB image relax endpoint 3",
            "ApproxNEB image relax hop 3+0 image 1",
            "ApproxNEB image relax hop 3+0 image 2",
            "ApproxNEB image relax hop 3+0 image 4",
            "ApproxNEB image relax hop 3+1 image 1",
            "ApproxNEB image relax hop 3+1 image 3",
            "ApproxNEB image relax hop 3+1 image 4",
            "ApproxNEB image relax hop 3+1 image 5",
        ]
    }

    mock_vasp(ref_paths)
    responses = run_locally(flow, create_folders=True, ensure_success=True)
