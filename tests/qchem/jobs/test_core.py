from pathlib import Path

import numpy as np
from emmet.core.qc_tasks import TaskDoc

# from emmet.core.vasp.calculation import IonicStep, VaspObject
from jobflow import run_locally

# OptMaker,
# ForceMaker,
# TransitionStateMaker,
# FreqMaker,
# PESScanMaker,
from pymatgen.core.structure import Molecule
from pytest import approx

from atomate2.qchem.jobs.core import FreqMaker, OptMaker, SinglePointMaker

# curr_dir = Path(os.path.dirname(sys.argv[0]))

# Specify the file name
file_name = "H2O.xyz"

# Construct the full path
# mol_path = curr_dir / file_name
# mol_path = Path(os.path.abspath(file_name))
mol_path = Path("tests/qchem/jobs/H2O.xyz")
H2O_structure = Molecule.from_file(mol_path)


def test_single_point_maker(mock_qchem, clean_dir, structure=H2O_structure):
    # jstore = jobflow.SETTINGS.JOB_STORE

    # mapping from job name to directory containing test files
    ref_paths = {"single_point": "water_single_point"}

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    # fake_run_qchem_kwargs = {"single_point": {"qin_settings": None}}
    fake_run_qchem_kwargs = {}

    # automatically use fake qchem during the test
    mock_qchem(ref_paths, fake_run_qchem_kwargs)

    # generate job
    job = SinglePointMaker().make(structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validate job outputs
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, TaskDoc)
    assert output1.output.final_energy == approx(-76.4451488262)


def test_opt_maker(mock_qchem, clean_dir, structure=H2O_structure):
    ref_paths = {"optimization": "water_optimization"}
    fake_run_qchem_kwargs = {}
    mock_qchem(ref_paths, fake_run_qchem_kwargs)

    job = OptMaker().make(structure)

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


def test_freq(mock_qchem, clean_dir, structure=H2O_structure):
    ref_paths = {"frequency": "water_frequency"}
    fake_run_qchem_kwargs = {}
    mock_qchem(ref_paths, fake_run_qchem_kwargs)

    job = FreqMaker().make(structure)

    responses = run_locally(job, create_folders=True, ensure_success=True)
    ref_freqs = np.array([1643.03, 3446.82, 3524.32])
    output1 = responses[job.uuid][1].output
    assert np.array_equal(
        np.array(output1.calcs_reversed[0].output.frequencies), ref_freqs
    )
    assert output1.output.final_energy == approx(-76.449405011)
