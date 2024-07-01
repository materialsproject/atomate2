"""Test MPMorph forcefield flows."""

import pytest
from jobflow import run_locally

from atomate2.forcefields.flows.mpmorph import (
    MPMorphCHGNetMDMaker,
    MPMorphFastQuenchCHGNetMDMaker,
    MPMorphFastQuenchLJMDMaker,
    MPMorphFastQuenchMACEMDMaker,
    MPMorphLJMDMaker,
    MPMorphMACEMDMaker,
    MPMorphSlowQuenchCHGNetMDMaker,
    MPMorphSlowQuenchLJMDMaker,
    MPMorphSlowQuenchMACEMDMaker,
)
from atomate2.forcefields.md import CHGNetMDMaker, LJMDMaker, MACEMDMaker

_velocity_seed = 1234

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
        "MACE Fast Quench",
        "MACE Slow Quench",
    ],
)
#       "CHGNet",
#       "CHGNet Slow Quench",
#       "CHGNet Fast Quench",


def test_mpmorph_mlff_maker(ff_name, si_structure, test_dir, clean_dir):
    temp = 300
    n_steps_convergence = 10
    n_steps_production = 20

    n_steps_quench = 15
    quench_temp_steps = 100
    quench_end_temp = 500
    quench_start_temp = 900

    unit_cell_structure = si_structure.copy()

    structure = unit_cell_structure.to_conventional() * (2, 2, 2)

    for mlff_name in name_to_md_maker:
        if mlff_name in ff_name:
            md_maker = name_to_md_maker[mlff_name]
            break

    quench_kwargs = (
        {
            "quench_n_steps": n_steps_quench,
            "quench_temperature_step": quench_temp_steps,
            "quench_end_temperature": quench_end_temp,
            "quench_start_temperature": quench_start_temp,
            "md_maker": md_maker(
                name=f"{mlff_name} Quench MD Maker", mb_velocity_seed=_velocity_seed
            ),
        }
        if "Slow Quench" in ff_name
        else {}
    )

    flow = name_to_maker[ff_name](
        temperature=temp,
        steps_convergence=n_steps_convergence,
        steps_total_production=n_steps_production,
        md_maker=md_maker(
            name=f"{mlff_name} MD Maker", mb_velocity_seed=_velocity_seed
        ),
        production_md_maker=md_maker(
            name=f"{mlff_name} Production MD Maker",
            mb_velocity_seed=_velocity_seed,
        ),
        quench_maker_kwargs=quench_kwargs,
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
    if "Fast Quench" in ff_name:
        assert len(uuids) == 10
    elif "Slow Quench" in ff_name:
        assert len(uuids) == 11
    else:  # "Main MPMorph MLFF Maker"
        assert len(uuids) == 7

    main_mp_morph_job_names = [
        "MD Maker 1",
        "MD Maker 2",
        "MD Maker 3",
        "MD Maker 4",
        "production run",
    ]

    if "Fast Quench" in ff_name:
        main_mp_morph_job_names.extend(["static", "relax"])
    if "Slow Quench" in ff_name:
        main_mp_morph_job_names.extend(
            [
                f"{T}K"
                for T in range(quench_start_temp, quench_end_temp, -quench_temp_steps)
            ]
        )
    task_docs = {}
    for uuid, job_name in uuids.items():
        for i, mp_job_name in enumerate(main_mp_morph_job_names):
            if mp_job_name in job_name:
                task_docs[mp_job_name] = response[uuid][1].output
                break

    # check number of steps of each MD equilibrate run and production run
    assert all(
        doc.output.n_steps == n_steps_convergence + 1
        for name, doc in task_docs.items()
        if "MD Maker" in name
    )
    assert task_docs["production run"].output.n_steps == n_steps_production + 1

    # check initial structure is scaled correctly
    ref_volumes = [
        669.9137883357736,
        1063.7982842554181,
        1587.9437945736847,
        2260.959035633235,
    ]

    assert all(
        any(
            doc.output.structure.volume == pytest.approx(ref_volume, abs=1e-2)
            for name, doc in task_docs.items()
            if "MD Maker" in name
        )
        for ref_volume in ref_volumes
    )

    # check temperature of each MD equilibrate run and production run #NOTE: Temperature flucations are a lot in most MDs, this is designed to check if MD is injected with approximately the right temperature, and that's why tolerance is so high
    assert all(
        doc.forcefield_objects["trajectory"].frame_properties[0]["temperature"]
        == pytest.approx(temp, abs=50)
        for name, doc in task_docs.items()
        if "MD Maker" in name
    )
    assert task_docs["production run"].forcefield_objects[
        "trajectory"
    ].frame_properties[0]["temperature"] == pytest.approx(temp, abs=50)

    # check that MD Maker Energies are close # TODO: This may be unnecessary because it changes from model to model
    assert task_docs["MD Maker 1"].output.energy == pytest.approx(-130, abs=5)
    assert task_docs["MD Maker 2"].output.energy == pytest.approx(-325, abs=5)
    assert task_docs["MD Maker 3"].output.energy == pytest.approx(-329, abs=5)
    assert task_docs["MD Maker 4"].output.energy == pytest.approx(-270, abs=5)

    if "Fast Quench" in ff_name:
        assert (
            task_docs["static"].input.structure.volume
            == task_docs["relax"].output.structure.volume
        )
        assert (
            task_docs["relax"].output.structure.volume
            <= task_docs["production run"].output.structure.volume
        )  # Ensures that the unit cell relaxes when fast quenched at 0K

    if "Slow Quench" in ff_name:
        # check volume doesn't change from production run
        assert all(
            doc.output.structure.volume
            == pytest.approx(
                task_docs["production run"].output.structure.volume, abs=1e-1
            )
            for name, doc in task_docs.items()
            if "K" in name
        )
        # check that the number of steps is correct
        assert all(
            doc.output.n_steps == n_steps_quench + 1
            for name, doc in task_docs.items()
            if "K" in name
        )
        # check that the temperature is correct

        ref_tempature = [
            T for T in range(quench_start_temp, quench_end_temp, -quench_temp_steps)
        ]
        assert all(
            any(
                doc.forcefield_objects["trajectory"].frame_properties[0]["temperature"]
                == pytest.approx(T, abs=100)
                for name, doc in task_docs.items()
                if "K" in name
            )
            for T in ref_tempature
        )
