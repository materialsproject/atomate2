import pytest
from emmet.core.neb import NebResult
from jobflow import run_locally
from pymatgen.core import Structure

from atomate2.forcefields.flows.approx_neb import ForceFieldApproxNebFromEndpointsMaker
from atomate2.forcefields.jobs import ForceFieldStaticMaker
from atomate2.utils.testing.common import get_job_uuid_name_map


def test_approx_neb_from_endpoints(test_dir, clean_dir):
    pytest.importorskip("matgl")

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
    # The MatPES-r2SCAN TensorNet was retrained for matgl 3.x (v2025.2), so
    # exact-energy references no longer apply. Verify the path structure:
    # 7 finite energies, endpoints degenerate, all in a sensible band for the
    # Zn host structure (~-1500 eV total).
    energies = output["collate_images_single_hop"].energies
    assert len(energies) == 7
    assert all(e is not None for e in energies)
    # endpoints (i=0, 6) should be ~degenerate by construction
    assert energies[0] == pytest.approx(energies[-1], rel=1e-3)
    # all energies in a physical band consistent with the host
    assert all(-2000.0 < e < -1000.0 for e in energies)

    assert len(output["collate_images_single_hop"].images) == 7
    assert all(
        image.volume == pytest.approx(endpoints[0].volume)
        for image in output["collate_images_single_hop"].images
    )


def test_ext_load_approx_neb_initialization():
    pytest.importorskip("mace")
    calculator_meta = {
        "@module": "mace.calculators",
        "@callable": "mace_mp",
    }
    maker = ForceFieldApproxNebFromEndpointsMaker(
        image_relax_maker=ForceFieldStaticMaker(force_field_name=calculator_meta)
    )
    assert maker.image_relax_maker.ase_calculator_name == "mace_mp"
