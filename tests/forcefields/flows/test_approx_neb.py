import pytest
from emmet.core.neb import NebResult
from jobflow import run_locally
from pymatgen.core import Structure

from atomate2.forcefields.flows.approx_neb import ForceFieldApproxNebFromEndpointsMaker
from atomate2.forcefields.jobs import ForceFieldStaticMaker
from atomate2.utils.testing.common import get_job_uuid_name_map


def test_approx_neb_from_endpoints(test_dir, clean_dir):
    vasp_aneb_dir = test_dir / "vasp" / "ApproxNEB"

    endpoints = [
        Structure.from_file(
            vasp_aneb_dir / f"ApproxNEB_image_relax_endpoint_{idx}/inputs/POSCAR.gz"
        )
        for idx in (0, 3)
    ]

    flow = ForceFieldApproxNebFromEndpointsMaker(
        image_relax_maker=ForceFieldStaticMaker(force_field_name="MATPES_R2SCAN")
    ).make("Zn", endpoints, vasp_aneb_dir / "host_structure_relax_2/outputs/CHGCAR.bz2")

    response = run_locally(flow)
    output = {
        job_name: response[uuid][1].output
        for uuid, job_name in get_job_uuid_name_map(flow).items()
    }

    assert isinstance(output["collate_images_single_hop"], NebResult)
    assert all(
        output["collate_images_single_hop"].energies[i] == pytest.approx(energy)
        for i, energy in enumerate(
            [
                -1559.5150146484375,
                -1554.3154296875,
                -1529.771484375,
                -1525.8846435546875,
                -1533.1453857421875,
                -1554.561767578125,
                -1559.5150146484375,
            ]
        )
    )

    assert len(output["collate_images_single_hop"].images) == 7
    assert all(
        image.volume == pytest.approx(endpoints[0].volume)
        for image in output["collate_images_single_hop"].images
    )
