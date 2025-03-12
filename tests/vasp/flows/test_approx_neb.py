"""Test ApproxNEB workflow."""

from pathlib import Path

import pytest
from emmet.core.neb import NebPathwayResult, NebResult
from jobflow import run_locally
from monty.serialization import loadfn
from pymatgen.core import Structure

from atomate2.vasp.flows.approx_neb import ApproxNebMaker


def test_approx_neb_flow(mock_vasp, clean_dir, vasp_test_dir):
    base_dir = Path(vasp_test_dir) / "ApproxNEB"
    flow_input = loadfn(base_dir / "approx_neb_input.json.gz")

    flow = ApproxNebMaker().make(
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
        job_name: f"ApproxNeb/{job_name.replace(' ', '_')}"
        for job_name in [
            "host structure relax 1",
            "host structure relax 2",
            "ApproxNEB image relax endpoint 0",
            "ApproxNEB image relax endpoint 1",
            "ApproxNEB image relax endpoint 3",
            "ApproxNEB image relax hop 3+0 image 1",
            "ApproxNEB image relax hop 3+0 image 2",
            "ApproxNEB image relax hop 3+0 image 3",
            "ApproxNEB image relax hop 3+0 image 4",
            "ApproxNEB image relax hop 3+0 image 5",
            "ApproxNEB image relax hop 3+1 image 1",
            "ApproxNEB image relax hop 3+1 image 2",
            "ApproxNEB image relax hop 3+1 image 3",
            "ApproxNEB image relax hop 3+1 image 4",
            "ApproxNEB image relax hop 3+1 image 5",
        ]
    }

    fake_run_vasp_kwargs = {
        key: {
            "incar_exclude": ["IBRION", "ALGO", "MAGMOM"]
        }  # updated this to generate flow test data more quickly
        for key in ref_paths
    }

    # POSCAR generation for images pretty sensitive to CHGCAR
    for key in [k for k in fake_run_vasp_kwargs if "relax hop" in k]:
        fake_run_vasp_kwargs[key]["check_inputs"] = ["incar", "kpoints", "potcar"]

    mock_vasp(ref_paths, fake_run_vasp_kwargs)
    # Do not ensure success here as the following jobs failed, but the flow should not:
    # ApproxNEB image relax hop 3+0 image 3
    # ApproxNEB image relax hop 3+0 image 5
    # ApproxNEB image relax hop 3+1 image 2
    responses = run_locally(flow, create_folders=True, ensure_success=False)
    output = {
        job.name: responses[job.uuid][1].output
        for job in flow.jobs
        if job.uuid in responses
    }

    assert len(output["collate_results"].hops) == 2

    assert all(
        len(getattr(output["collate_results"].hops["3+0"], k))
        == 5  # two failed image calcs
        for k in ("energies", "images")
    )
    assert all(
        len(getattr(output["collate_results"].hops["3+1"], k))
        == 6  # one failed image calc
        for k in ("energies", "images")
    )

    assert all(
        isinstance(image, Structure)
        for hop in output["collate_results"].hops.values()
        for image in hop.images
    )

    assert isinstance(output["collate_results"], NebPathwayResult)
    assert all(
        isinstance(hop, NebResult) for hop in output["collate_results"].hops.values()
    )
    assert all(
        output["collate_results"].max_barriers[k]
        == pytest.approx(
            max(
                output["collate_results"].forward_barriers[k],
                output["collate_results"].forward_barriers[k],
            )
        )
        for k in output["collate_results"].hops
    )

    ref_results = loadfn(base_dir / "collate_results.json.gz")

    assert all(
        output["collate_results"].hops[k].energies[idx] == pytest.approx(energy)
        for k, ref_hop in ref_results.hops.items()
        for idx, energy in enumerate(ref_hop.energies)
    )

    from monty.serialization import dumpfn

    dumpfn(output["collate_results"], "/Users/aaronkaplan/Desktop/temp_aneb.json.gz")

    assert all(
        getattr(output["collate_results"], f"{direction}_barriers")[k]
        == pytest.approx(getattr(ref_results, f"{direction}_barriers")[k])
        for direction in ("forward", "reverse")
        for k in ref_results.hops
    )
