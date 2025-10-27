import logging
from shutil import which

import pytest

logger = logging.getLogger("atomate2")


@pytest.mark.skipif(
    which("abinit") is None, reason="abinit must be installed to run this test."
)
# def test_run_silicon_carbide_shg(
#    mock_abinit, mock_mrgddb, mock_anaddb, abinit_test_dir, clean_dir
# ):
#    from jobflow import run_locally
#    from monty.serialization import loadfn
#    from pymatgen.core.structure import Structure
#
#    from atomate2.abinit.schemas.anaddb import AnaddbTaskDoc
#
#    # load the initial structure, the maker and the ref_paths from the test_dir
#    test_dir = (
#      abinit_test_dir / "flows" / "dfpt_core" / "ShgFlowMaker" / "silicon_carbide_shg"
#    )
#    structure = Structure.from_file(test_dir / "initial_structure.json.gz")
#    maker_info = loadfn(test_dir / "maker.json.gz")
#    maker = maker_info["maker"]
#    ref_paths = loadfn(test_dir / "ref_paths.json.gz")
#
#    mock_abinit(ref_paths)
#    mock_mrgddb(ref_paths)
#    mock_anaddb(ref_paths)
#
#    # make the flow or job, run it and ensure that it finished running successfully
#    flow_or_job = maker.make(structure)
#    responses = run_locally(flow_or_job, create_folders=True, ensure_success=True)
#
#    # validation the outputs of the flow or job
#    assert (
#        len(responses) == 14
#    )  # 1 scf + 3 ddk + 1 generate_perts + 3 dde + 4 dte + 1 mrgddb + 1 anaddb
#    for job, _ in flow_or_job.iterflow():
#        if job.name == "generate_perts":
#            assert len(responses[job.uuid]) == 2  # bc DDE and DTE
#            continue
#        assert len(responses[job.uuid]) == 1
#        if job.name == "Anaddb":
#            output1 = responses[job.uuid][1].output
#            assert isinstance(output1, AnaddbTaskDoc)

def test_run_si_phonon_band_structure(
    mock_abinit, mock_mrgddb, mock_mrgdvdb, mock_anaddb, abinit_test_dir, clean_dir
):
    from jobflow import run_locally
    from monty.serialization import loadfn
    from pymatgen.core.structure import Structure

    from atomate2.abinit.schemas.anaddb import AnaddbTaskDoc

    # load the initial structure, the maker and the ref_paths from the test_dir
    test_dir = (
        abinit_test_dir
        / "flows"
        / "dfpt_core"
        / "PhononMaker"
        / "Si_phonon_bandstructure"
    )
    structure = Structure.from_file(test_dir / "initial_structure.json.gz")
    maker_info = loadfn(test_dir / "maker.json.gz")
    maker = maker_info["maker"]
    ref_paths = loadfn(test_dir / "ref_paths.json.gz")

    mock_abinit(ref_paths)
    mock_mrgddb(ref_paths)
    mock_mrgdvdb(ref_paths)
    mock_anaddb(ref_paths)

    # make the flow or job, run it and ensure that it finished running successfully
    flow_or_job = maker.make(structure)
    responses = run_locally(flow_or_job, create_folders=True, ensure_success=True)

    # validation the outputs of the flow or job
    assert len(responses) == 15  # 1 scf + 3 ddk + 1 generate_perts + 1 dde + 2 wfq
    # + 4 phonons + 1 mrgddb + 1 mrgdv + 1 anaddb
    for job, _ in flow_or_job.iterflow():
        if job.name == "generate_perts":
            assert len(responses[job.uuid]) == 2
            # list of 2 objects: the first is the flow that replace the job
            # and the second the output associated to the flow
            continue
        assert len(responses[job.uuid]) == 1
        if job.name == "Anaddb":
            output1 = responses[job.uuid][1].output
            assert isinstance(output1, AnaddbTaskDoc)
