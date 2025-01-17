from emmet.core.qc_tasks import TaskDoc
from jobflow import run_locally
from pytest import approx

from atomate2.qchem.jobs.core import FreqMaker, OptMaker, SinglePointMaker


def test_single_point_maker(mock_qchem, clean_dir, h2o_molecule):
    # mapping from job name to directory containing test files
    ref_paths = {"single point": "water_single_point"}

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    # fake_run_qchem_kwargs = {"single_point": {"qin_settings": None}}
    fake_run_qchem_kwargs = {}

    # automatically use fake qchem during the test
    mock_qchem(ref_paths, fake_run_qchem_kwargs)

    # generate job
    job = SinglePointMaker().make(h2o_molecule)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validate job outputs
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, TaskDoc)
    assert output1.output.final_energy == approx(-76.4451488262)


def test_opt_maker(mock_qchem, clean_dir, h2o_molecule):
    ref_paths = {"optimization": "water_optimization"}
    fake_run_qchem_kwargs = {}
    mock_qchem(ref_paths, fake_run_qchem_kwargs)

    job = OptMaker().make(h2o_molecule)

    responses = run_locally(job, create_folders=True, ensure_success=True)
    opt_geometry = {
        "@module": "pymatgen.core.structure",
        "@class": "Molecule",
        "charge": 0,
        "spin_multiplicity": 1,
        "sites": [
            {
                "name": "O",
                "species": [{"element": "O", "occu": 1}],
                "xyz": [-0.8001136722, 2.2241304324, -0.0128020517],
                "properties": {},
                "label": "O",
            },
            {
                "name": "H",
                "species": [{"element": "H", "occu": 1}],
                "xyz": [0.1605037895, 2.195300528, 0.0211059581],
                "properties": {},
                "label": "H",
            },
            {
                "name": "H",
                "species": [{"element": "H", "occu": 1}],
                "xyz": [-1.0782701173, 1.6278690395, 0.6883760935],
                "properties": {},
                "label": "H",
            },
        ],
        "properties": {},
    }

    output1 = responses[job.uuid][1].output
    assert isinstance(output1, TaskDoc)
    assert sorted(opt_geometry.items()) == sorted(
        output1.output.optimized_molecule.as_dict().items()
    )
    assert output1.output.final_energy == approx(-76.450849061819)


def test_freq(mock_qchem, clean_dir, h2o_molecule):
    ref_paths = {"frequency": "water_frequency"}
    fake_run_qchem_kwargs = {}
    mock_qchem(ref_paths, fake_run_qchem_kwargs)

    job = FreqMaker().make(h2o_molecule)

    responses = run_locally(job, create_folders=True, ensure_success=True)
    ref_freqs = [1643.03, 3446.82, 3524.32]
    output1 = responses[job.uuid][1].output
    assert output1.calcs_reversed[0].output.frequencies == ref_freqs
    assert output1.output.final_energy == approx(-76.449405011)
