from collections.abc import Callable
from pathlib import Path


def test_band_structure_run_silicon(
    mock_abinit: Callable, abinit_test_dir: Path, clean_dir: Path
) -> None:
    """
    Test the BandStructureMaker for silicon band structure calculation.

    This test verifies that the band structure flow maker correctly executes
    all necessary calculations including SCF, NSCF, and band structure jobs.

    Parameters
    ----------
    mock_abinit
        Fixture to mock ABINIT runs.
    abinit_test_dir
        Path to the ABINIT test directory.
    clean_dir
        Fixture to provide a clean working directory.
    """
    from jobflow import run_locally
    from monty.serialization import loadfn
    from pymatgen.core.structure import Structure

    # Load the initial structure, maker, and reference paths from test directory
    test_dir = abinit_test_dir / "flows" / "core" / "BandStructureMaker" / "silicon"
    structure = Structure.from_file(test_dir / "initial_structure.json.gz")
    maker_info = loadfn(test_dir / "maker.json.gz")
    maker = maker_info["maker"]
    ref_paths = loadfn(test_dir / "ref_paths.json.gz")

    # Setup mock fixture with reference paths
    mock_abinit(ref_paths)

    # Create and run the flow locally, ensuring successful completion
    flow_or_job = maker.make(structure)
    responses = run_locally(
        flow_or_job, create_folders=True, ensure_success=True, raise_immediately=True
    )

    # Validate the number of jobs executed (expected: 3 jobs)
    assert len(responses) == 3
    for job, _parents in flow_or_job.iterflow():
        assert len(responses[job.uuid]) == 1


def test_relax_run_silicon_standard(
    mock_abinit: Callable, abinit_test_dir: Path, clean_dir: Path
) -> None:
    """
    Test the RelaxFlowMaker for standard silicon relaxation.

    This test verifies that the relaxation flow maker correctly executes
    the standard relaxation workflow for silicon.

    Parameters
    ----------
    mock_abinit
        Fixture to mock ABINIT runs.
    abinit_test_dir
        Path to the ABINIT test directory.
    clean_dir
        Fixture to provide a clean working directory.
    """
    from jobflow import run_locally
    from monty.serialization import loadfn
    from pymatgen.core.structure import Structure

    # Load the initial structure, maker, and reference paths from test directory
    test_dir = (
        abinit_test_dir / "flows" / "core" / "RelaxFlowMaker" / "silicon_standard"
    )
    structure = Structure.from_file(test_dir / "initial_structure.json.gz")
    maker_info = loadfn(test_dir / "maker.json.gz")
    maker = maker_info["maker"]
    ref_paths = loadfn(test_dir / "ref_paths.json.gz")

    # Setup mock fixture with reference paths
    mock_abinit(ref_paths)

    # Create and run the flow locally, ensuring successful completion
    flow_or_job = maker.make(structure)
    responses = run_locally(
        flow_or_job, create_folders=True, ensure_success=True, raise_immediately=True
    )

    # Validate the number of jobs executed (expected: 2 jobs)
    assert len(responses) == 2
    for job, _parents in flow_or_job.iterflow():
        assert len(responses[job.uuid]) == 1


def test_relax_ion_ioncell_relaxation() -> None:
    """
    Test the ion_ioncell_relaxation class method of RelaxFlowMaker.

    This test verifies that the ion_ioncell_relaxation factory method correctly
    creates a RelaxFlowMaker with two relaxation makers (ion and ion+cell) and
    that user ABINIT settings are properly propagated to both makers.
    """
    from atomate2.abinit.flows.core import RelaxFlowMaker

    # Define custom ABINIT settings
    settings = {"nband": 100}

    # Create RelaxFlowMaker with ion and ion+cell relaxation stages
    maker = RelaxFlowMaker.ion_ioncell_relaxation(user_abinit_settings=settings)

    # Verify that two relaxation makers were created
    assert len(maker.relaxation_makers) == 2

    # Verify that user settings were applied to the first relaxation maker (ion)
    assert (
        maker.relaxation_makers[0].input_set_generator.user_abinit_settings == settings
    )

    # Verify that user settings were applied to the second relaxation maker (ion+cell)
    assert (
        maker.relaxation_makers[1].input_set_generator.user_abinit_settings == settings
    )
