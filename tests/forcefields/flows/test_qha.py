from pathlib import Path

from jobflow import run_locally
from pymatgen.core.structure import Structure

from atomate2.common.schemas.qha import PhononQHADoc
from atomate2.forcefields.flows.phonons import PhononMaker
from atomate2.forcefields.flows.qha import CHGNetQhaMaker


def test_qha_dir(clean_dir, si_structure: Structure, tmp_path: Path):
    # TODO brittle due to inability to adjust dtypes in CHGNetRelaxMaker

    flow = CHGNetQhaMaker(
        number_of_frames=5,
        ignore_imaginary_modes=True,
        phonon_maker=PhononMaker(
            min_length=10,
            store_force_constants=False,
            bulk_relax_maker=None,
            generate_frequencies_eigenvectors_kwargs={
                "tol_imaginary_modes": 5e-1,
                "tmin": 0,
                "tmax": 1000,
                "tstep": 10,
            },
        ),
    ).make(si_structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    # # validate the outputs
    ph_bs_dos_doc = responses[flow[-1].uuid][1].output
    assert isinstance(ph_bs_dos_doc, PhononQHADoc)
