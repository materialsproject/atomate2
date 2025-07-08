"""Test VASP NEB Flows"""

from pathlib import Path

import pytest
from emmet.core.neb import (
    BarrierAnalysis,
    NebIntermediateImagesDoc,
    NebMethod,
    NebTaskDoc,
)
from jobflow import run_locally
from monty.serialization import loadfn
from pymatgen.core import Structure

from atomate2.common.jobs.neb import _get_images_from_endpoints
from atomate2.vasp.jobs.core import RelaxMaker
from atomate2.vasp.jobs.neb import NebFromEndpointsMaker, NebFromImagesMaker
from atomate2.vasp.sets.core import RelaxSetGenerator

expected_incar_tags_relax = [
    "ALGO",
    "EDIFFG",
    "ENAUG",
    "ENCUT",
    "GGA",
    "IBRION",
    "ISIF",
    "ISMEAR",
    "ISPIN",
    "LAECHG",
    "LASPH",
    "LCHARG",
    "LELF",
    "LMIXTAU",
    "LORBIT",
    "LREAL",
    "LVTOT",
    "LWAVE",
    "NELM",
    "NSW",
    "PREC",
]
expected_incar_tags_neb = [
    "IMAGES",
    "IOPT",  # VTST specific
    "LCLIMB",  # VTST specific
    "POTIM",
    "SPRING",
    *expected_incar_tags_relax,
]


def test_neb_from_endpoints_maker(mock_vasp, clean_dir, vasp_test_dir):
    """Test nearest-neighbor vacancy migration in Si supercell."""

    num_images = 5
    intermed_images = num_images - 2
    base_neb_dir = Path(vasp_test_dir) / "Si_NEB"

    ref_paths = {
        k: str(base_neb_dir / k.replace(" ", "_"))
        for k in ("relax endpoint 1", "relax endpoint 2", "NEB")
    }

    fake_run_vasp_kwargs = {
        k: {
            "incar_settings": expected_incar_tags_neb
            if k == "NEB"
            else expected_incar_tags_relax
        }
        for k in ref_paths
    }

    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    """
    # NB the endpoint structures were generated as follows:

    ```py
    structure = si_structure.to_conventional() * (2, 2, 2)

    _, n_idx, _, d = structure.get_neighbor_list(
        structure.lattice.a / 4, sites=[structure[0]]
    )
    min_idx = np.argmin(d)
    nn_idx = n_idx[min_idx]

    endpoints = [structure.copy() for _ in range(2)]
    endpoints[0].remove_sites([0])
    endpoints[1].remove_sites([nn_idx])
    ```
    """
    endpoints = [
        Structure.from_file(
            base_neb_dir / f"relax_endpoint_{idx + 1}" / "inputs" / "POSCAR.gz"
        )
        for idx in range(2)
    ]

    relax_maker = RelaxMaker(
        input_set_generator=RelaxSetGenerator(
            user_incar_settings={"ISIF": 2, "EDIFFG": -0.05}
        )
    )

    neb_job = NebFromEndpointsMaker(
        endpoint_relax_maker=relax_maker,
    ).make(
        endpoints=endpoints,
        num_images=intermed_images,
        autosort_tol=0.5,
    )

    # ensure flow runs successfully
    responses = run_locally(neb_job, create_folders=True, ensure_success=True)
    output = {job.name: responses[job.uuid][1].output for job in neb_job.jobs}

    fixed_cell_vol = endpoints[0].volume
    assert all(
        output[f"relax endpoint {1 + idx}"].output.structure.volume
        == pytest.approx(fixed_cell_vol)
        for idx in range(2)
    )

    expected_images = loadfn(str(base_neb_dir / "get_images_from_endpoints.json.gz"))
    assert len(output["get_images_from_endpoints"]) == num_images
    assert (
        output["get_images_from_endpoints"][idx] == image
        for idx, image in enumerate(expected_images)
    )

    assert isinstance(output["collect_neb_output"], NebTaskDoc)
    expected_neb_result = NebTaskDoc(
        **loadfn(str(base_neb_dir / "neb_task_doc.json.gz"))
    )
    assert all(
        output["collect_neb_output"].energies[i] == pytest.approx(energy)
        for i, energy in enumerate(expected_neb_result.energies)
    )

    # endpoints + intermediate images
    assert len(output["collect_neb_output"].images) == num_images

    # just intermediate images
    assert len(output["NEB"].images) == intermed_images

    assert all(
        getattr(output["collect_neb_output"], f"{direction}_barrier")
        == pytest.approx(getattr(expected_neb_result, f"{direction}_barrier"))
        for direction in ("forward", "reverse")
    )

    assert isinstance(output["collect_neb_output"].barrier_analysis, BarrierAnalysis)
    assert set(output["collect_neb_output"].barrier_analysis.model_dump()) == {
        "cubic_spline_pars",
        "energies",
        "forward_barrier",
        "frame_index",
        "reverse_barrier",
        "ts_energy",
        "ts_frame_index",
        "ts_in_frames",
    }


def test_neb_from_images_maker(mock_vasp, clean_dir, vasp_test_dir):
    num_images = 5
    intermed_images = num_images - 2
    base_neb_dir = Path(vasp_test_dir) / "Si_NEB"

    endpoints = [
        Structure.from_file(
            base_neb_dir / f"relax_endpoint_{1 + idx}/outputs/CONTCAR.gz"
        )
        for idx in range(2)
    ]
    images = endpoints[0].interpolate(
        endpoints[1], nimages=intermed_images + 1, autosort_tol=0.5
    )
    assert all(
        images[idx] == image
        for idx, image in enumerate(
            _get_images_from_endpoints(
                endpoints, num_images=intermed_images, autosort_tol=0.5
            )
        )
    )

    ref_paths = {"NEB": str(base_neb_dir / "NEB")}

    mock_vasp(ref_paths, {"NEB": {"incar_settings": expected_incar_tags_neb}})
    neb_job = NebFromImagesMaker().make(images)
    response = run_locally(neb_job, create_folders=True, ensure_success=True)
    output = response[neb_job.uuid][1].output

    ref_neb_task = NebIntermediateImagesDoc.from_directory(
        Path(ref_paths["NEB"]) / "outputs"
    )
    assert isinstance(output, NebIntermediateImagesDoc)
    assert len(output.images) == num_images - 2
    assert all(
        output.energies[i] == pytest.approx(energy)
        for i, energy in enumerate(ref_neb_task.energies)
    )
    assert output.neb_method == NebMethod.CLIMBING_IMAGE
