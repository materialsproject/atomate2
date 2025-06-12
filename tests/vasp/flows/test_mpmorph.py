"""Test MPMorph VASP flows."""

import pytest
from jobflow import run_locally

from atomate2.vasp.flows.mpmorph import (
    MPMorphFastQuenchVaspMDMaker,
    MPMorphSlowQuenchVaspMDMaker,
    MPMorphVaspMDMaker,
)
from atomate2.vasp.jobs.mpmorph import (
    BaseMPMorphMDMaker,
    FastQuenchVaspMaker,
    SlowQuenchVaspMaker,
)

name_to_maker = {
    "MPMorph Vasp": MPMorphVaspMDMaker,
    "MPMorph Vasp Slow Quench": MPMorphSlowQuenchVaspMDMaker,
    "MPMorph Vasp Fast Quench": MPMorphFastQuenchVaspMDMaker,
}


@pytest.fixture
def initial_structure(si_structure):
    initial_structure = si_structure.copy()
    initial_structure.scale_lattice(1.1 * initial_structure.volume)
    return initial_structure * (2, 2, 2)


def _get_uuid_from_job(job, dct):
    if hasattr(job, "jobs"):
        for j in job.jobs:
            _get_uuid_from_job(j, dct)
    else:
        dct[job.uuid] = job.name


@pytest.mark.parametrize(
    "quench_type",
    [
        "fast",
        "slow",
    ],
)
def test_vasp_mpmorph(
    initial_structure, mock_vasp, clean_dir, vasp_test_dir, quench_type
):
    job_names = [
        "MP Morph VASP Equilibrium Volume Maker Convergence MPMorph VASP MD Maker 1",
        "MP Morph VASP Equilibrium Volume Maker Convergence MPMorph VASP MD Maker 2",
        "MP Morph VASP Equilibrium Volume Maker Convergence MPMorph VASP MD Maker 3",
        "MP Morph VASP MD Maker production run",
    ]

    temperature = 500
    quench_temps = {"start": temperature, "end": 300, "step": 50}
    steps = {"convergence": 100, "production": 100, "quench": 100}

    quench_maker = None
    if quench_type == "fast":
        quench_maker = FastQuenchVaspMaker()
        job_names += ["MP GGA relax 1", "MP GGA relax 2", "MP GGA static"]
    elif quench_type == "slow":
        quench_maker = SlowQuenchVaspMaker(
            BaseMPMorphMDMaker(name="Slow Quench VASP Maker"),
            quench_n_steps=steps["quench"],
            quench_temperature_step=quench_temps["step"],
            quench_end_temperature=quench_temps["end"],
            quench_start_temperature=quench_temps["start"],
            descent_method="stepwise",
        )
        job_names += [
            "Vasp Slow Quench MD Maker 350K",
            "Vasp Slow Quench MD Maker 400K",
            "Vasp Slow Quench MD Maker 450K",
            "Vasp Slow Quench MD Maker 500K",
        ]

    ref_paths = {
        job_name: f"Si_mp_morph/{'_'.join(job_name.split(' '))}"
        for job_name in job_names
    }

    mock_vasp(ref_paths)

    flow = MPMorphVaspMDMaker.from_temperature_and_steps(
        temperature=temperature,
        n_steps_convergence=steps["convergence"],
        n_steps_production=steps["production"],
        end_temp=None,
        quench_maker=quench_maker,
    ).make(initial_structure)

    uuids = {}
    _get_uuid_from_job(flow, uuids)

    responses = run_locally(
        flow,
        create_folders=True,
        ensure_success=True,
    )

    for resp in responses.values():
        if hasattr(resp[1], "replace") and resp[1].replace is not None:
            for job in resp[1].replace:
                uuids[job.uuid] = job.name

    task_docs = {}
    for uuid, job_name in uuids.items():
        if (resp := responses.get(uuid)) is not None:
            task_docs[job_name] = resp[1].output

    if quench_type == "fast":
        assert len(uuids) == 9
    elif quench_type == "slow":
        assert len(uuids) == 10
    else:
        assert len(uuids) == 6

    # check number of steps of each MD equilibrate run and production run

    for mode, num_steps in steps.items():
        assert all(
            doc.input.parameters["NSW"] == num_steps
            for name, doc in task_docs.items()
            if mode in name.lower()
        )

    # check temperature of each MD equilibrate run and production run
    assert all(
        (
            doc.input.parameters["TEBEG"] == temperature
            and doc.input.parameters["TEEND"] == temperature
        )
        for name, doc in task_docs.items()
        if any(name_str in name.lower() for name_str in ("convergence", "production"))
    )

    # check that MD Maker Energies are close

    ref_eos = {
        "energy": [-47.31250505, -65.66356248, -66.3485729],
        "volume": [184.2262917923377, 359.8169761569094, 621.7637347991395],
    }

    assert all(
        task_docs[
            "MP Morph VASP Equilibrium Volume Maker "
            f"Convergence MPMorph VASP MD Maker {1 + idx}"
        ].output.energy
        == pytest.approx(ref_eos["energy"][idx])
        for idx in range(3)
    )

    assert all(
        task_docs[
            "MP Morph VASP Equilibrium Volume Maker "
            f"Convergence MPMorph VASP MD Maker {1 + idx}"
        ].output.structure.volume
        == pytest.approx(ref_eos["volume"][idx])
        for idx in range(3)
    )

    if quench_type == "fast":
        assert task_docs["MP GGA static"].input.structure.volume == pytest.approx(
            task_docs["MP GGA relax 2"].output.structure.volume,
        )
        assert task_docs["MP GGA relax 2"].input.structure.volume == pytest.approx(
            task_docs["MP GGA relax 1"].output.structure.volume,
        )

        """
        # @BryantLI-BLI: does this block make sense?
        #    The structure won't always shrink in volume
        assert (
            task_docs["MP GGA relax 1"].output.structure.volume
            <= task_docs[
                "MP Morph VASP MD Maker production run"
            ].output.structure.volume
        )  # Ensures that the unit cell relaxes when fast quenched at 0K
        """

    if quench_type == "slow":
        # check volume doesn't change from production run
        assert all(
            doc.output.structure.volume
            == pytest.approx(
                task_docs[
                    "MP Morph VASP MD Maker production run"
                ].output.structure.volume,
                abs=1e-1,
            )
            for name, doc in task_docs.items()
            if "K" in name
        )
        # check that the number of steps is correct
        assert all(
            doc.input.parameters["NSW"] == steps["quench"]
            for name, doc in task_docs.items()
            if "quench" in name.lower()
        )
        # check that the temperature is correct

        ref_tempature = list(
            range(quench_temps["start"], quench_temps["end"], -quench_temps["step"])
        )

        assert all(
            any(
                doc.input.parameters["TEBEG"] == pytest.approx(T)
                for name, doc in task_docs.items()
                if "K" in name
            )
            for T in ref_tempature
        )
        assert all(
            any(
                doc.input.parameters["TEEND"] == pytest.approx(T)
                for name, doc in task_docs.items()
                if "K" in name
            )
            for T in ref_tempature
        )
