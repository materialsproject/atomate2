"""Test MPMorph forcefield flows."""

import pytest
import re
from pymatgen.analysis.structure_matcher import StructureMatcher
from jobflow import run_locally
from atomate2.forcefields.flows.mpmorph import (
    MPMorphLJMDMaker,
    MPMorphCHGNetMDMaker,
    MPMorphMACEMDMaker,
    MPMorphSlowQuenchLJMDMaker,
    MPMorphSlowQuenchCHGNetMDMaker,
    MPMorphSlowQuenchMACEMDMaker,
    MPMorphFastQuenchLJMDMaker,
    MPMorphFastQuenchCHGNetMDMaker,
    MPMorphFastQuenchMACEMDMaker,
)
from atomate2.forcefields.md import (
    ForceFieldMDMaker,
    LJMDMaker,
    MACEMDMaker,
    CHGNetMDMaker,
)

name_to_maker = {
    "LJ": MPMorphLJMDMaker,
    "LJ Slow Quench": MPMorphSlowQuenchLJMDMaker,
    "LJ Fast Quench": MPMorphFastQuenchLJMDMaker,
    "MACE": MPMorphMACEMDMaker,
    "MACE Slow Quench": MPMorphSlowQuenchMACEMDMaker,
    "MACE Fast Quench": MPMorphFastQuenchMACEMDMaker,
    "CHGNet": MPMorphCHGNetMDMaker,
    "CHGNet Slow Quench": MPMorphSlowQuenchCHGNetMDMaker,
    "CHGNet Fast Quench": MPMorphFastQuenchCHGNetMDMaker,
}

name_to_md_maker = {
    "LJ": LJMDMaker,
    "MACE": MACEMDMaker,
    "CHGNet": CHGNetMDMaker,
}


def _get_uuid_from_job(job, dct):
    if hasattr(job, "jobs"):
        for j in job.jobs:
            _get_uuid_from_job(j, dct)
    else:
        dct[job.uuid] = job.name


@pytest.mark.parametrize(
    "ff_name",
    [
        "MACE",
    ],
)

#       "CHGNet",
#       "CHGNet Slow Quench",
#       "CHGNet Fast Quench",


def test_mpmorph_mlff_maker(ff_name, si_structure, test_dir, clean_dir):
    temp = 300
    n_steps_convergence = 10
    n_steps_production = 20

    ref_energies_per_atom = {
        "CHGNet": -5.280157089233398,
        "MACE": -5.311369895935059,
        "LJ": 0,
    }

    # ASE can slightly change tolerances on structure positions
    matcher = StructureMatcher()

    unit_cell_structure = si_structure.copy()

    structure = unit_cell_structure.to_conventional() * (2, 2, 2)

    for mlff_name in name_to_md_maker:
        if mlff_name in ff_name:
            md_maker = name_to_md_maker[mlff_name]
            break
    flow = name_to_maker[ff_name](
        temperature=temp,
        steps_convergence=n_steps_convergence,
        steps_total_production=n_steps_production,
        md_maker=md_maker(name=f"{mlff_name} MD Maker", mb_velocity_seed=1234),
        production_md_maker=md_maker(
            name=f"{mlff_name} Production MD Maker", mb_velocity_seed=1234
        ),
    ).make(structure)

    uuids = {}
    _get_uuid_from_job(flow, uuids)

    response = run_locally(
        flow,
        ensure_success=True,
    )
    for resp in response.values():
        if hasattr(resp[1], "replace") and resp[1].replace is not None:
            for job in resp[1].replace:
                uuids[job.uuid] = job.name

    # check number of jobs spawned
    assert len(uuids) == 7

    main_mp_morph_job_names = [
        "MD Maker 1",
        "MD Maker 2",
        "MD Maker 3",
        "MD Maker 4",
        "MD Maker production run",
    ]

    task_docs = {}

    for uuid, job_name in uuids.items():
        for i, mp_job_name in enumerate(main_mp_morph_job_names):
            if mp_job_name in job_name:
                task_docs[mp_job_name] = response[uuid][1].output
                break

    print("_______________DEBUG_______________")
    print(task_docs["MD Maker 1"].input.__dir__())
    print(task_docs["MD Maker 1"].output.__dir__())
    print(task_docs["MD Maker 1"].output.forcefield_objects)

    print("_______________DEBUG_______________")

    # check number of steps of each MD equilibrate run and production run
    assert all(
        doc.output.n_steps == n_steps_convergence + 1
        for name, doc in task_docs.items()
        if "production run" not in name
    )
    assert task_docs["MD Maker production run"].output.n_steps == n_steps_production + 1

    # check temperature of each MD equilibrate run and production run

    # check that MD Maker volume are constants
    assert task_docs["MD Maker 1"].output.energy == pytest.approx(-130, abs=1)
    assert task_docs["MD Maker 2"].output.energy == pytest.approx(-325, abs=1)
    assert task_docs["MD Maker 3"].output.energy == pytest.approx(-329, abs=1)
    assert task_docs["MD Maker 4"].output.energy == pytest.approx(-270, abs=1)

    assert task_docs["MD Maker 1"].input.structure.volume == pytest.approx(669.914)
    assert task_docs["MD Maker 2"].input.structure.volume == pytest.approx(1063.798)
    assert task_docs["MD Maker 3"].input.structure.volume == pytest.approx(1587.94379)
    assert task_docs["MD Maker 4"].input.structure.volume == pytest.approx(2260.959)

    assert False
