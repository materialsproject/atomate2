import logging
from collections.abc import Callable
from pathlib import Path
from shutil import which

import pytest

logger = logging.getLogger("atomate2")


@pytest.mark.skipif(
    which("abinit") is None, reason="abinit must be installed to run this test."
)
def test_run_silicon_carbide_shg(
    mock_abinit: Callable,
    mock_mrgddb: Callable,
    mock_anaddb: Callable,
    abinit_test_dir: Path,
    clean_dir: Path,
) -> None:
    """
    Test the ShgFlowMaker for silicon carbide second-harmonic generation (SHG).

    This test verifies that the SHG flow maker correctly executes all necessary
    DFPT calculations including SCF, DDK, DDE, DTE perturbations, DDB merging,
    and ANADDB post-processing.

    Parameters
    ----------
    mock_abinit
        Fixture to mock ABINIT runs.
    mock_mrgddb
        Fixture to mock mrgddb runs.
    mock_anaddb
        Fixture to mock anaddb runs.
    abinit_test_dir
        Path to the ABINIT test directory.
    clean_dir
        Fixture to provide a clean working directory.
    """
    from jobflow import run_locally
    from monty.serialization import loadfn
    from pymatgen.core.structure import Structure

    from atomate2.abinit.schemas.anaddb import AnaddbTaskDoc

    # Load the initial structure, maker, and reference paths from test directory
    test_dir = (
        abinit_test_dir / "flows" / "dfpt_core" / "ShgFlowMaker" / "silicon_carbide_shg"
    )
    structure = Structure.from_file(test_dir / "initial_structure.json.gz")
    maker_info = loadfn(test_dir / "maker.json.gz")
    maker = maker_info["maker"]
    ref_paths = loadfn(test_dir / "ref_paths.json.gz")

    # Setup mock fixtures with reference paths
    mock_abinit(ref_paths)
    mock_mrgddb(ref_paths)
    mock_anaddb(ref_paths)

    # Create and run the flow locally, ensuring successful completion
    flow_or_job = maker.make(structure)
    responses = run_locally(flow_or_job, create_folders=True, ensure_success=True)

    # Validate the number of jobs executed
    # Expected: 1 SCF + 3 DDK + 1 generate_perts + 3 DDE + 4 DTE + 1 mrgddb + 1 anaddb
    assert len(responses) == 14

    # Verify each job in the flow
    for job, _ in flow_or_job.iterflow():
        if job.name == "generate_perts":
            # List of 2 objects: the first is the flow that replaces the job
            # and the second is the output associated to the flow
            assert len(responses[job.uuid]) == 2
            continue
        assert len(responses[job.uuid]) == 1
        if job.name == "Anaddb":
            output1 = responses[job.uuid][1].output
            assert isinstance(output1, AnaddbTaskDoc)


def test_run_si_phonon_band_structure(
    mock_abinit: Callable,
    mock_mrgddb: Callable,
    mock_mrgdvdb: Callable,
    mock_anaddb: Callable,
    abinit_test_dir: Path,
    clean_dir: Path,
) -> None:
    """
    Test the PhononMaker for silicon phonon band structure calculation.

    This test verifies that the phonon flow maker correctly executes all necessary
    DFPT calculations including SCF, DDK, DDE, WFQ, phonon perturbations, DDB merging,
    POT file merging, and ANADDB post-processing for phonon bands and DOS.

    Parameters
    ----------
    mock_abinit
        Fixture to mock ABINIT runs.
    mock_mrgddb
        Fixture to mock mrgddb runs.
    mock_mrgdvdb
        Fixture to mock mrgdv runs.
    mock_anaddb
        Fixture to mock anaddb runs.
    abinit_test_dir
        Path to the ABINIT test directory.
    clean_dir
        Fixture to provide a clean working directory.
    """
    from jobflow import run_locally
    from monty.serialization import loadfn
    from pymatgen.core.structure import Structure

    from atomate2.abinit.schemas.anaddb import AnaddbTaskDoc

    # Load the initial structure, maker, and reference paths from test directory
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

    # Setup mock fixtures with reference paths
    mock_abinit(ref_paths)
    mock_mrgddb(ref_paths)
    mock_mrgdvdb(ref_paths)
    mock_anaddb(ref_paths)

    # Create and run the flow locally, ensuring successful completion
    flow_or_job = maker.make(structure)
    responses = run_locally(flow_or_job, create_folders=True, ensure_success=True)

    # Validate the number of jobs executed
    # Expected: 1 SCF + 3 DDK + 1 generate_perts + 1 DDE + 2 WFQ
    # + 4 phonons + 1 mrgddb + 1 mrgdv + 1 anaddb
    assert len(responses) == 15

    # Verify each job in the flow
    for job, _ in flow_or_job.iterflow():
        if job.name == "generate_perts":
            # List of 2 objects: the first is the flow that replaces the job
            # and the second is the output associated to the flow
            assert len(responses[job.uuid]) == 2
            continue
        assert len(responses[job.uuid]) == 1
        if job.name == "Anaddb":
            output1 = responses[job.uuid][1].output
            assert isinstance(output1, AnaddbTaskDoc)
