"""Test MPMorph VASP flows."""

import pytest
from jobflow import run_locally
from pymatgen.core import Structure
from pymatgen.io.vasp import Kpoints

from atomate2.common.flows.mpmorph import EquilibriumVolumeMaker, MPMorphMDMaker
from atomate2.vasp.flows.mpmorph import (
    MPMorphVaspMDFastQuenchMaker,
    MPMorphVaspMDMaker,
    MPMorphVaspMDSlowQuenchMaker,
)
from atomate2.vasp.jobs.md import MDMaker
from atomate2.vasp.run import DEFAULT_HANDLERS
from atomate2.vasp.sets.core import MDSetGenerator

name_to_maker = {
    "MPMorph Vasp": MPMorphVaspMDMaker,
    "MPMorph Vasp Slow Quench": MPMorphVaspMDSlowQuenchMaker,
    "MPMorph Vasp Fast Quench": MPMorphVaspMDFastQuenchMaker,
}


@pytest.mark.parametrize(
    "maker_name",
    [
        "MPMorph Vasp",
        "MPMorph Vasp Slow Quench",
        "MPMorph Vasp Fast Quench",
    ],
)
def test_base_mpmorph_makers(mock_vasp, clean_dir, vasp_test_dir, maker_name):
    job_names = [
        "MP Morph VASP Equilibrium Volume Maker MPMorph MD Maker 1",
        "MP Morph VASP Equilibrium Volume Maker MPMorph MD Maker 2",
        "MP Morph VASP Equilibrium Volume Maker MPMorph MD Maker 3",
        "MP Morph VASP MD Maker production run",
        "MP Morph VASP MD Maker Slow Quench production run",
        "MP Morph VASP MD Maker Fast Quench production run",
        "Vasp Slow Quench MD Maker 900K",
        "Vasp Slow Quench MD Maker 800K",
        "Vasp Slow Quench MD Maker 700K",
        "Vasp Slow Quench MD Maker 600K",
        "MP pre-relax",
        "MP meta-GGA relax",
        "MP meta-GGA static",
    ]
    job_paths = [
        "Si_mp_morph/BaseVaspMPMorph/Si_0.8",
        "Si_mp_morph/BaseVaspMPMorph/Si_1.0",
        "Si_mp_morph/BaseVaspMPMorph/Si_1.2",
        "Si_mp_morph/BaseVaspMPMorph/Si_prod",
        "Si_mp_morph/BaseVaspMPMorph/Si_prod",
        "Si_mp_morph/BaseVaspMPMorph/Si_prod",
        "Si_mp_morph/BaseVaspMPMorph/Si_900K",
        "Si_mp_morph/BaseVaspMPMorph/Si_800K",
        "Si_mp_morph/BaseVaspMPMorph/Si_700K",
        "Si_mp_morph/BaseVaspMPMorph/Si_600K",
        "Si_mp_morph/BaseVaspMPMorph/pre_relax",
        "Si_mp_morph/BaseVaspMPMorph/relax",
        "Si_mp_morph/BaseVaspMPMorph/static",
    ]

    ref_paths = dict(zip(job_names, job_paths))

    mock_vasp(ref_paths)

    intial_structure = Structure.from_file(
        f"{vasp_test_dir}/Si_mp_morph/BaseVaspMPMorph/Si_1.0/inputs/POSCAR"
    )
    temperature: int = 300
    end_temp: int = 300
    steps_convergence: int = 10
    steps_production: int = 20

    n_steps_quench = 15
    quench_temp_steps = 100
    quench_end_temp = 500
    quench_start_temp = 900

    quench_kwargs = (
        {
            "quench_n_steps": n_steps_quench,
            "quench_temperature_step": quench_temp_steps,
            "quench_end_temperature": quench_end_temp,
            "quench_start_temperature": quench_start_temp,
        }
        if "Slow Quench" in maker_name
        else {}
    )

    flow = name_to_maker[maker_name](
        temperature=temperature,
        end_temp=end_temp,
        steps_convergence=steps_convergence,
        steps_total_production=steps_production,
        quench_maker_kwargs=quench_kwargs,
    ).make(intial_structure)

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

    main_mp_morph_job_names = [
        "MD Maker 1",
        "MD Maker 2",
        "MD Maker 3",
        "production run",
    ]

    if "Fast Quench" in maker_name:
        main_mp_morph_job_names.extend(["static", "relax", "pre relax"])
    if "Slow Quench" in maker_name:
        main_mp_morph_job_names.extend(
            [
                f"{T}K"
                for T in range(quench_start_temp, quench_end_temp, -quench_temp_steps)
            ]
        )

    task_docs = {}
    for uuid, job_name in uuids.items():
        for _i, mp_job_name in enumerate(main_mp_morph_job_names):
            if mp_job_name in job_name:
                task_docs[mp_job_name] = responses[uuid][1].output
                break

    ref_md_energies = {
        "energy": [-3.94691047, -35.4165625, -32.22834385],
        "volume": [82.59487098351644, 161.31810738968053, 278.7576895693679],
    }

    # Asserts right number of jobs spawned
    # check number of jobs spawned
    if "Fast Quench" in maker_name:
        assert len(uuids) == 9
    elif "Slow Quench" in maker_name:
        assert len(uuids) == 10
    else:  # "Main MPMorph MLFF Maker"
        assert len(uuids) == 6

    # check number of steps of each MD equilibrate run and production run
    assert all(
        doc.input.parameters["NSW"] == steps_convergence
        for name, doc in task_docs.items()
        if "MD Maker" in name
    )
    assert task_docs["production run"].input.parameters["NSW"] == steps_production

    # check initial structure is scaled correctly
    assert all(
        any(
            doc.output.structure.volume == pytest.approx(ref_volume, abs=1e-2)
            for name, doc in task_docs.items()
            if "MD Maker" in name
        )
        for ref_volume in ref_md_energies["volume"]
    )

    # check temperature of each MD equilibrate run and production run
    assert all(
        doc.input.parameters["TEBEG"] == temperature
        for name, doc in task_docs.items()
        if "MD Maker" in name
    )
    assert task_docs["production run"].input.parameters["TEBEG"] == temperature

    assert all(
        doc.input.parameters["TEEND"] == end_temp
        for name, doc in task_docs.items()
        if "MD Maker" in name
    )
    assert task_docs["production run"].input.parameters["TEEND"] == end_temp

    # check that MD Maker Energies are close

    assert all(
        any(
            doc.output.energy == pytest.approx(ref_energy, abs=1)
            for name, doc in task_docs.items()
            if "MD Maker" in name
        )
        for ref_energy in ref_md_energies["energy"]
    )

    if "Fast Quench" in maker_name:
        assert task_docs["static"].input.structure.volume == pytest.approx(
            task_docs["relax"].output.structure.volume, 1e-5
        )
        assert (
            task_docs["relax"].output.structure.volume
            <= task_docs["production run"].output.structure.volume
        )  # Ensures that the unit cell relaxes when fast quenched at 0K

    if "Slow Quench" in maker_name:
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
            doc.input.parameters["NSW"] == n_steps_quench
            for name, doc in task_docs.items()
            if "K" in name
        )
        # check that the temperature is correct

        ref_tempature = list(
            range(quench_start_temp, quench_end_temp, -quench_temp_steps)
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


base_mpmorph_incar_settings = {
    "ISPIN": 1,
    # Do not consider magnetism in AIMD simulations
    "LREAL": "Auto",
    # Perform calculation in real space for AIMD due to large unit cell size
    "LAECHG": False,
    # Don't need AECCAR for AIMD
    "EDIFFG": None,
    # Does not apply to MD simulations, see: https://www.vasp.at/wiki/index.php/EDIFFG
    "GGA": "PS",
    # Just let VASP decide based on POTCAR - the default, PS yields the error below
    "LPLANE": False,
    # LPLANE is recommended to be False on Cray machines (https://www.vasp.at/wiki/index.php/LPLANE)
    "LDAUPRINT": 0,
    "SIGMA": 0.05,
}


def _get_uuid_from_job(job, dct):
    if hasattr(job, "jobs"):
        for j in job.jobs:
            _get_uuid_from_job(j, dct)
    else:
        dct[job.uuid] = job.name


def test_equilibrium_volume_maker(mock_vasp, clean_dir, vasp_test_dir):
    job_names = [
        "Equilibrium Volume Maker molecular dynamics 1",
        "Equilibrium Volume Maker molecular dynamics 2",
        "Equilibrium Volume Maker molecular dynamics 3",
    ]
    job_paths = [
        "Si_mp_morph/Si_0.8",
        "Si_mp_morph/Si_1.0",
        "Si_mp_morph/Si_1.2",
    ]

    ref_paths = dict(zip(job_names, job_paths))

    mock_vasp(ref_paths)

    intial_structure = Structure.from_file(
        f"{vasp_test_dir}/Si_mp_morph/Si_1.0/inputs/POSCAR.gz"
    )
    temperature: int = 300
    end_temp: int = 300
    steps_convergence: int = 20

    gamma_point = Kpoints()
    incar_settings = base_mpmorph_incar_settings

    aimd_equil_maker = MDMaker(
        input_set_generator=MDSetGenerator(
            ensemble="nvt",
            start_temp=temperature,
            end_temp=end_temp,
            nsteps=steps_convergence,
            time_step=2,
            # adapted from MPMorph settings
            user_incar_settings=incar_settings,
            user_kpoints_settings=gamma_point,
        )
    )

    flow = EquilibriumVolumeMaker(
        md_maker=aimd_equil_maker,
    ).make(structure=intial_structure)

    responses = run_locally(flow, create_folders=True, ensure_success=True)

    ref_md_energies = {
        "energy": [-13.44200043, -35.97470303, -32.48531985],
        "volume": [82.59487098351644, 161.31810738968053, 278.7576895693679],
    }
    uuids = list(responses)

    assert len(uuids) == 5
    for i in range(len(ref_md_energies["energy"])):
        assert responses[uuids[1 + i]][
            1
        ].output.output.structure.volume == pytest.approx(ref_md_energies["volume"][i])
        assert responses[uuids[1 + i]][1].output.output.energy == pytest.approx(
            ref_md_energies["energy"][i]
        )

    assert isinstance(responses[uuids[4]][1].output, Structure)
    assert responses[uuids[4]][1].output.volume == pytest.approx(171.51227)


def test_recursion_equilibrium_volume_maker(mock_vasp, clean_dir, vasp_test_dir):

    job_names = [
        "Equilibrium Volume Maker molecular dynamics 1",
        "Equilibrium Volume Maker molecular dynamics 2",
        "Equilibrium Volume Maker molecular dynamics 3",
        "Equilibrium Volume Maker molecular dynamics 4",
    ]

    job_paths = [
        "Si_mp_morph/recursion/Si_3.48",
        "Si_mp_morph/recursion/Si_4.35",
        "Si_mp_morph/recursion/Si_5.22",
        "Si_mp_morph/recursion/Si_6.80",
    ]

    ref_paths = dict(zip(job_names, job_paths))

    mock_vasp(ref_paths)

    intial_structure = Structure.from_file(
        f"{vasp_test_dir}/Si_mp_morph/recursion/Si_4.35/inputs/POSCAR.gz"
    )
    temperature: int = 300
    end_temp: int = 300
    steps_convergence: int = 20

    gamma_point = Kpoints()
    incar_settings = base_mpmorph_incar_settings.copy()
    incar_settings.update({"SIGMA": 0.2})

    # For close separations, positive energy is reasonable and expected
    _vasp_handlers = [
        handler for handler in DEFAULT_HANDLERS if "PositiveEnergy" not in str(handler)
    ]

    aimd_equil_maker = MDMaker(
        input_set_generator=MDSetGenerator(
            ensemble="nvt",
            start_temp=temperature,
            end_temp=end_temp,
            nsteps=steps_convergence,
            time_step=2,
            # adapted from MPMorph settings
            user_incar_settings=incar_settings,
            user_kpoints_settings=gamma_point,
        ),
        run_vasp_kwargs={"handlers": _vasp_handlers},
    )

    flow = EquilibriumVolumeMaker(
        md_maker=aimd_equil_maker,
    ).make(structure=intial_structure)

    responses = run_locally(flow, create_folders=True, ensure_success=True)

    pre_recursion_ref_md_energies = {
        "energy": [71.53406556, -14.64974676, -35.39207478],
        "volume": [42.28857394356042, 82.59487098351644, 142.7239370595164],
        "V0": 170.97868797782795,
        "Vmax": 142.7239370595164,
        "Vmin": 42.28857394356042,
    }

    post_recursion_ref_md_energies = {
        "energy": [71.53406556, -14.64974676, -35.39207478, -31.18239194],
        "volume": [
            42.28857394356042,
            82.59487098351644,
            142.7239370595164,
            314.80734557888934,
        ],
        "V0": 170.97868797782795,
        "Vmax": 142.7239370595164,
        "Vmin": 42.28857394356042,
    }

    uuids = list(responses)

    assert len(uuids) == 7

    for i in range(len(pre_recursion_ref_md_energies["energy"])):
        assert responses[uuids[1 + i]][
            1
        ].output.output.structure.volume == pytest.approx(
            pre_recursion_ref_md_energies["volume"][i]
        )
        assert responses[uuids[1 + i]][1].output.output.energy == pytest.approx(
            pre_recursion_ref_md_energies["energy"][i]
        )

    assert responses[uuids[5]][1].output.output.structure.volume == pytest.approx(
        post_recursion_ref_md_energies["volume"][3]
    )
    assert responses[uuids[5]][1].output.output.energy == pytest.approx(
        post_recursion_ref_md_energies["energy"][3]
    )

    assert isinstance(responses[uuids[-1]][1].output, Structure)
    assert responses[uuids[-1]][1].output.volume == pytest.approx(177.244197)


def test_mp_morph_maker(mock_vasp, clean_dir, vasp_test_dir):
    ref_paths = {
        "Equilibrium Volume Maker molecular dynamics 1": "Si_mp_morph/Si_0.8",
        "Equilibrium Volume Maker molecular dynamics 2": "Si_mp_morph/Si_1.0",
        "Equilibrium Volume Maker molecular dynamics 3": "Si_mp_morph/Si_1.2",
        "MP Morph md production run": "Si_mp_morph/Si_prod",
    }

    mock_vasp(ref_paths)

    intial_structure = Structure.from_file(
        f"{vasp_test_dir}/Si_mp_morph/Si_1.0/inputs/POSCAR.gz"
    )
    temperature: int = 300
    end_temp: int = 300
    steps_convergence: int = 20
    steps_production: int = 50

    gamma_point = Kpoints()
    incar_settings = base_mpmorph_incar_settings

    aimd_equil_maker = MDMaker(
        input_set_generator=MDSetGenerator(
            ensemble="nvt",
            start_temp=temperature,
            end_temp=end_temp,
            nsteps=steps_convergence,
            time_step=2,
            user_incar_settings=incar_settings,
            user_kpoints_settings=gamma_point,
        )
    )

    aimd_prod_maker = MDMaker(
        input_set_generator=MDSetGenerator(
            ensemble="nvt",
            start_temp=temperature,
            end_temp=end_temp,
            nsteps=steps_production,
            time_step=2,
            user_incar_settings=incar_settings,
            user_kpoints_settings=gamma_point,
        )
    )

    flow = MPMorphMDMaker(
        convergence_md_maker=EquilibriumVolumeMaker(md_maker=aimd_equil_maker),
        production_md_maker=aimd_prod_maker,
    ).make(structure=intial_structure)

    responses = run_locally(
        flow,
        create_folders=True,
        ensure_success=True,
    )

    uuids = list(responses)

    ref_md_energies = {
        "energy": [-13.44200043, -35.97470303, -32.48531985],
        "volume": [82.59487098351644, 161.31810738968053, 278.7576895693679],
    }

    assert len(uuids) == 6
    for i in range(len(ref_md_energies["energy"])):
        assert responses[uuids[1 + i]][
            1
        ].output.output.structure.volume == pytest.approx(ref_md_energies["volume"][i])
        assert responses[uuids[1 + i]][1].output.output.energy == pytest.approx(
            ref_md_energies["energy"][i]
        )

    assert isinstance(responses[uuids[4]][1].output, Structure)
    assert responses[uuids[4]][1].output.volume == pytest.approx(171.51227)

    assert responses[uuids[5]][1].output.output.structure.volume == pytest.approx(
        172.7682
    )
    assert responses[uuids[5]][1].output.output.energy == pytest.approx(-38.1286)


def test_mpmorph_vasp_maker(mock_vasp, clean_dir, vasp_test_dir):
    job_names = [
        "MP Morph VASP Equilibrium Volume Maker MPMorph MD Maker 1",
        "MP Morph VASP Equilibrium Volume Maker MPMorph MD Maker 2",
        "MP Morph VASP Equilibrium Volume Maker MPMorph MD Maker 3",
        "MP Morph VASP MD Maker production run",
    ]

    job_paths = [
        "Si_mp_morph/Si_0.8",
        "Si_mp_morph/Si_1.0",
        "Si_mp_morph/Si_1.2",
        "Si_mp_morph/Si_prod",
    ]

    ref_paths = dict(zip(job_names, job_paths))

    mock_vasp(ref_paths)

    intial_structure = Structure.from_file(
        f"{vasp_test_dir}/Si_mp_morph/Si_1.0/inputs/POSCAR.gz"
    )
    temperature: int = 300
    end_temp: int = 300
    steps_convergence: int = 20
    steps_production: int = 50

    flow = MPMorphVaspMDMaker(
        temperature=temperature,
        end_temp=end_temp,
        steps_convergence=steps_convergence,
        steps_total_production=steps_production,
    ).make(structure=intial_structure)

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

    main_mp_morph_job_names = [
        "MD Maker 1",
        "MD Maker 2",
        "MD Maker 3",
        "MD Maker 4",
        "production run",
    ]

    task_docs = {}
    for uuid, job_name in uuids.items():
        for _i, mp_job_name in enumerate(main_mp_morph_job_names):
            if mp_job_name in job_name:
                task_docs[mp_job_name] = responses[uuid][1].output
                break

    ref_md_energies = {
        "energy": [-13.44200043, -35.97470303, -32.48531985],
        "volume": [82.59487098351644, 161.31810738968053, 278.7576895693679],
    }
    # Asserts right number of jobs spawned
    assert len(uuids) == 6

    # check number of steps of each MD equilibrate run and production run
    assert all(
        doc.input.parameters["NSW"] == steps_convergence
        for name, doc in task_docs.items()
        if "MD Maker" in name
    )
    assert task_docs["production run"].input.parameters["NSW"] == steps_production

    # check initial structure is scaled correctly
    assert all(
        any(
            doc.output.structure.volume == pytest.approx(ref_volume, abs=1e-2)
            for name, doc in task_docs.items()
            if "MD Maker" in name
        )
        for ref_volume in ref_md_energies["volume"]
    )

    # check temperature of each MD equilibrate run and production run
    assert all(
        doc.input.parameters["TEBEG"] == temperature
        for name, doc in task_docs.items()
        if "MD Maker" in name
    )
    assert task_docs["production run"].input.parameters["TEBEG"] == temperature

    assert all(
        doc.input.parameters["TEEND"] == end_temp
        for name, doc in task_docs.items()
        if "MD Maker" in name
    )
    assert task_docs["production run"].input.parameters["TEEND"] == end_temp

    # check that MD Maker Energies are close

    assert all(
        any(
            doc.output.energy == pytest.approx(ref_volume, abs=1e-2)
            for name, doc in task_docs.items()
            if "MD Maker" in name
        )
        for ref_volume in ref_md_energies["energy"]
    )
