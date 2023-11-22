import pytest
from emmet.core.tasks import TaskDoc
from jobflow import Flow, run_locally
from pymatgen.core import Structure

from atomate2.vasp.flows.eos.mp import (
    MPGGAEosDoubleRelaxMaker,
    MPGGAEosMaker,
)
from atomate2.vasp.jobs.eos.mp import (
    MPMetaGGAEosStaticMaker,
)

expected_incar_relax = {
    "ISIF": 3,
    "IBRION": 2,
    "EDIFF": 1.0e-6,
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "LMAXMIX": 6,
    "KSPACING": 0.22,
}

expected_incar_relax_1 = {
    **expected_incar_relax,
    "EDIFFG": -0.05,
}

expected_incar_deform = {**expected_incar_relax, "ISIF": 2}

expected_incar_static = {**expected_incar_relax, "NSW": 0, "IBRION": -1, "ISMEAR": -5}
expected_incar_static.pop("ISIF")


def structure_equality(struct1: Structure, struct2: Structure):
    structs = [struct1, struct2]
    for struct in structs:
        for site_prop in struct.site_properties:
            struct.remove_site_property(site_prop)
    return structs[0] == structs[1]


def test_mp_eos_double_relax_maker(mock_vasp, clean_dir, vasp_test_dir):
    ref_paths = {
        "EOS MP GGA relax 1": "Si_EOS_MP_GGA/mp-149-PBE-EOS_MP_GGA_relax_1",
        "EOS MP GGA relax 2": "Si_EOS_MP_GGA/mp-149-PBE-EOS_MP_GGA_relax_2",
    }

    structure = Structure.from_file(
        f"{vasp_test_dir}/{ref_paths['EOS MP GGA relax 1']}/inputs/POSCAR"
    )

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        key: {"incar_settings": list(expected_incar_relax)} for key in ref_paths
    }
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    flow = MPGGAEosDoubleRelaxMaker().make(structure)
    # ensure flow runs successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    for uuid in responses:
        assert isinstance(responses[uuid][1].output, TaskDoc)

    taskdocs = [responses[uuid][1].output for uuid in responses]

    # ensure that output structure of first relaxation is fed to second
    assert structure_equality(taskdocs[1].input.structure, taskdocs[0].output.structure)

    assert len(responses) == len(ref_paths)

    expected_incar_pars = [expected_incar_relax_1, expected_incar_relax]
    for ijob in range(2):
        for key, expected in expected_incar_pars[ijob].items():
            actual = taskdocs[ijob].input.parameters.get(key, None)
            assert actual == expected, f"{key=}, {actual=}, {expected=}"


@pytest.mark.parametrize("do_statics", [False, True])
def test_mp_eos_maker(
    do_statics: bool,
    mock_vasp,
    clean_dir,
    vasp_test_dir,
    nframes: int = 2,
    linear_strain: tuple = (-0.05, 0.05),
):
    base_ref_path = "Si_EOS_MP_GGA/"
    ref_paths = {}
    expected_incars = {
        "EOS MP GGA relax 1": expected_incar_relax_1,
        "EOS MP GGA relax 2": expected_incar_relax,
    }

    for i in range(2):
        ref_paths[f"EOS MP GGA relax {1+i}"] = f"mp-149-PBE-EOS_MP_GGA_relax_{1+i}"

    for i in range(nframes):
        ref_paths[
            f"EOS Deformation Relax {i}"
        ] = f"mp-149-PBE-EOS_Deformation_Relax_{i}"
        expected_incars[f"EOS Deformation Relax {i}"] = expected_incar_deform

        if do_statics:
            ref_paths[f"EOS Static {i}"] = f"mp-149-PBE-EOS_Static_{i}"
            expected_incars[f"EOS Static {i}"] = expected_incar_static

    if do_statics:
        ref_paths["EOS equilibrium static"] = "mp-149-PBE-EOS_equilibrium_static"
        expected_incars["EOS equilibrium static"] = expected_incar_static

    ref_paths = {job: base_ref_path + ref_paths[job] for job in ref_paths}

    fake_run_vasp_kwargs = {
        key: {"incar_settings": list(expected_incars[key])} for key in ref_paths
    }

    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    static_maker = None
    if do_statics:
        static_maker = MPMetaGGAEosStaticMaker(
            copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR",)}
        )

    structure = Structure.from_file(
        f"{vasp_test_dir}/{ref_paths['EOS MP GGA relax 1']}/inputs/POSCAR"
    )

    # cannot perform least-squares fit for four parameters with only 3 data points
    flow = MPGGAEosMaker(
        static_maker=static_maker,
        number_of_frames=nframes,
        linear_strain=linear_strain,
        postprocessor=None,
    ).make(structure)

    # ensure flow runs successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    taskdocs = {}
    jobs = []
    for job in flow:
        if isinstance(job, Flow):
            jobs.extend([t.name for t in job.jobs])
        else:
            jobs.append(job.name)

    for ijob, uuid in enumerate(flow.job_uuids):
        taskdocs[jobs[ijob]] = responses[uuid][1].output

    assert len(responses) == len(ref_paths)

    if do_statics:
        # Check that deformation jobs correctly feed structures
        # into statics

        assert structure_equality(
            taskdocs["EOS equilibrium static"].input.structure,
            taskdocs["EOS MP GGA relax 2"].output.structure,
        ), "Equilibrium static input structure is wrong!"

        for i in range(nframes):
            assert structure_equality(
                taskdocs[f"EOS Static {i}"].input.structure,
                taskdocs[f"EOS Deformation Relax {i}"].output.structure,
            ), f"Static {i} has incorrect input structure!"

    relaxed_totens = [
        taskdocs[f"EOS Deformation Relax {i}"].calcs_reversed[0].output.energy
        for i in range(nframes)
    ]
    ref_relaxed_totens = [-10.547764, -10.632079]

    assert all(
        abs(relaxed_totens[i] - ref_relaxed_totens[i]) < 1.0e-6 for i in range(nframes)
    )

    if do_statics:
        static_totens = [
            taskdocs[f"EOS Static {i}"].calcs_reversed[0].output.energy
            for i in range(nframes)
        ]
        ref_static_totens = [-10.547764, -10.63208]
        assert all(
            abs(static_totens[i] - ref_static_totens[i]) < 1.0e-6
            for i in range(nframes)
        )
