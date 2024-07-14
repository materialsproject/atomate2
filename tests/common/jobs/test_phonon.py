from jobflow import Flow, run_locally
from numpy.testing import assert_allclose
from pymatgen.core import Structure

from atomate2.common.jobs.phonons import get_supercell_size
from atomate2.forcefields.flows.phonons import PhononMaker
from atomate2.forcefields.jobs import ForceFieldRelaxMaker, ForceFieldStaticMaker
from atomate2.forcefields.utils import MLFF


def test_phonon_get_supercell_size(clean_dir, si_structure: Structure):
    job = get_supercell_size(si_structure, min_length=18, prefer_90_degrees=True)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    assert_allclose(responses[job.uuid][1].output, [[6, -2, 0], [0, 6, 0], [-3, -2, 5]])


def test_phonon_maker_initialization_with_all_mlff(si_structure):
    """Test PhononMaker can be initialized with all MLFF static and relax makers."""

    for mlff in MLFF:
        static_maker = ForceFieldStaticMaker(
            name=f"{mlff} static",
            force_field_name=str(mlff),
        )
        relax_maker = ForceFieldRelaxMaker(
            name=f"{mlff} relax",
            force_field_name=str(mlff),
            relax_kwargs={"fmax": 0.00001},
        )

        try:
            phonon_maker = PhononMaker(
                bulk_relax_maker=relax_maker,
                static_energy_maker=static_maker,
                phonon_displacement_maker=static_maker,
                use_symmetrized_structure="conventional",
                create_thermal_displacements=False,
                store_force_constants=False,
            )

            flow = phonon_maker.make(si_structure)
            assert isinstance(flow, Flow)
            assert len(flow) == 7, f"{len(flow)=}"
            assert flow[1].name == f"{mlff} relax", f"{flow[1].name=}"
            assert flow[3].name == f"{mlff} static", f"{flow[3].name=}"
            assert flow[4].name == "generate_phonon_displacements", f"{flow[4].name=}"
            assert flow[5].name == "run_phonon_displacements", f"{flow[5].name=}"

        except Exception as exc:
            exc.add_note(f"Failed to initialize PhononMaker with {mlff=} makers")
            raise
