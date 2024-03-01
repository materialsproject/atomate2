import pytest
from emmet.core.tasks import TaskDoc
from jobflow import Flow, run_locally
from monty.serialization import loadfn
from pymatgen.core import Structure
from pytest import approx

from atomate2.common.jobs.eos import PostProcessEosPressure
from atomate2.vasp.flows.eos import MPGGAEosDoubleRelaxMaker, MPGGAEosMaker
from atomate2.vasp.jobs.eos import MPGGAEosStaticMaker

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
        ref_paths[f"EOS MP GGA relax deformation {i}"] = (
            f"mp-149-PBE-EOS_Deformation_Relax_{i}"
        )
        expected_incars[f"EOS MP GGA relax deformation {i}"] = expected_incar_deform

        if do_statics:
            ref_paths[f"EOS MP GGA static {i}"] = f"mp-149-PBE-EOS_Static_{i}"
            expected_incars[f"EOS MP GGA static {i}"] = expected_incar_static

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
        static_maker = MPGGAEosStaticMaker(
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
        postprocessor=PostProcessEosPressure(),
        _store_transformation_information=False,
    ).make(structure)

    # ensure flow runs successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    jobs = []
    for job in flow:
        if isinstance(job, Flow):
            jobs.extend(list(job.jobs))
        else:
            jobs.append(job)

    postprocess_uuid = next(
        job.uuid for job in jobs if job.name == "MP GGA EOS Maker postprocessing"
    )
    flow_output = responses[postprocess_uuid][1].output

    jobs = {job.name: job.uuid for job in jobs if job.name in ref_paths}
    job_output = {
        job_name: responses[uuid][1].output for job_name, uuid in jobs.items()
    }

    # deformation jobs not included in this
    assert len(job_output) == len(ref_paths)

    ref_energies = {"EOS MP GGA relax 1": -10.849349, "EOS MP GGA relax 2": -10.849357}
    if do_statics:
        ref_energies["EOS equilibrium static"] = -10.849357

    # check that TaskDoc energies agree
    assert all(
        approx(ref_energies[key]) == job_output[key].calcs_reversed[0].output.energy
        for key in ref_energies
    )

    ref_eos_fit = loadfn(f"{vasp_test_dir}/{base_ref_path}/Si_pressure_EOS_fit.json.gz")
    job_types_to_check = ("relax", "static") if do_statics else ("relax",)
    for job_type in job_types_to_check:
        for key in ("energy", "volume", "EOS"):
            data = flow_output[job_type][key]
            if isinstance(data, list):
                assert all(
                    approx(ref_eos_fit[job_type][key][i]) == data[i]
                    for i in range(len(data))
                )
            elif isinstance(data, dict):
                assert all(
                    approx(v) == data[k] for k, v in ref_eos_fit[job_type][key].items()
                )
            elif isinstance(data, (float, int)):
                assert approx(ref_eos_fit[job_type][key]) == data
