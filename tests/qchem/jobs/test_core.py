import jobflow
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

from atomate2.qchem.jobs.core import (
    SinglePointMaker,
)

H2O_structure = Molecule.from_file("H2O.xyz")


def test_single_point_maker(mock_qchem, clean_dir, structure=H2O_structure):
    jstore = jobflow.SETTINGS.JOB_STORE

    # mapping from job name to directory containing test files
    ref_paths = {"single_point": "single_point"}

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_qchem_kwargs = {"single_point": {"qin_settings": None}}

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_qchem(ref_paths, fake_run_qchem_kwargs)

    # generate job
    job = SinglePointMaker().make(structure)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job, create_folders=True, ensure_success=True)

    # validate job outputs
    output1 = responses[job.uuid][1].output
    assert isinstance(output1, TaskDoc)
    assert output1.output.energy == approx(-76.4399546592)

    with jstore.additional_stores["data"] as s:
        doc = s.query_one({"job_uuid": job.uuid})
        dd = doc["data"]
        # assert dd["@class"] == "Chgcar"
