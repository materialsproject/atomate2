from jobflow import run_locally
from pymatgen.core import Molecule, Structure

from atomate2.vasp.flows.adsorption import AdsorptionMaker
from atomate2.vasp.jobs.adsorption import AdsorptionDocument


def test_adsorption(mock_vasp, clean_dir, test_dir):
    # mapping from job name to directory containing test files
    ref_paths = {
        "bulk_relax_maker__bulk_relax_job": "Au_adsorption/bulk_relax",
        "mol_relax_maker__mol_relax_job": "Au_adsorption/mol_relax",
        "mol_static_maker__mol_static_job": "Au_adsorption/mol_static",
        "slab_relax_maker__slab_relax_job": "Au_adsorption/slab_relax",
        "slab_static_maker__slab_static_job": "Au_adsorption/slab_static",
        "slab_relax_maker__adsconfig_0": ("Au_adsorption/ads_relax_1_3"),
        "slab_relax_maker__adsconfig_1": ("Au_adsorption/ads_relax_2_3"),
        "slab_relax_maker__adsconfig_2": ("Au_adsorption/ads_relax_3_3"),
        "slab_static_maker__static_adsconfig_0": ("Au_adsorption/ads_static_1_3"),
        "slab_static_maker__static_adsconfig_1": ("Au_adsorption/ads_static_2_3"),
        "slab_static_maker__static_adsconfig_2": ("Au_adsorption/ads_static_3_3"),
    }

    fake_run_vasp_kwargs = {
        path: {"incar_settings": ["ISIF", "NSW"]} for path in ref_paths
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    molcule_str = Structure.from_file(
        test_dir / "vasp/Au_adsorption/mol_relax/inputs/POSCAR"
    )

    molecule_indices = [i for i, site in enumerate(molcule_str)]
    molecule_coords = [molcule_str[i].coords for i in molecule_indices]
    molecule_species = [molcule_str[i].species_string for i in molecule_indices]

    molecule = Molecule(molecule_species, molecule_coords)
    bulk_structure = Structure.from_file(
        test_dir / "vasp/Au_adsorption/bulk_relax/inputs/POSCAR"
    )

    flow = AdsorptionMaker(min_lw=5.0, min_slab_size=4.0).make(
        molecule=molecule,
        structure=bulk_structure,
    )

    # Run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    job_names = [
        response.output.task_label
        for job_responses in responses.values()
        for response in job_responses.values()
        if hasattr(response.output, "task_label")
    ]

    expected_job_names = [
        "bulk_relax_maker__bulk_relax_job",
        "mol_relax_maker__mol_relax_job",
        "mol_static_maker__mol_static_job",
        "slab_relax_maker__slab_relax_job",
        "slab_static_maker__slab_static_job",
        "slab_relax_maker__adsconfig_0",
        "slab_relax_maker__adsconfig_1",
        "slab_relax_maker__adsconfig_2",
        "slab_static_maker__static_adsconfig_0",
        "slab_static_maker__static_adsconfig_1",
        "slab_static_maker__static_adsconfig_2",
    ]

    for actual_name in expected_job_names:
        assert actual_name in job_names, f"Job '{actual_name}' not found."

    assert flow[-1].uuid in responses, "Adsorption calculation job not found."

    adsorption_document = next(iter(responses[flow[-1].uuid].values())).output

    assert isinstance(
        adsorption_document, AdsorptionDocument
    ), "Output is not an AdsorptionDocument instance."

    # Extract the adsorption energies
    adsorption_energies = adsorption_document.adsorption_energies

    # The energies should already be sorted based on your job function
    expected_energies = [
        -3.0666021499999943,
        -2.9407460899999904,
        -2.0976731399999906,
    ]

    assert (
        adsorption_energies == expected_energies
    ), "Adsorption energies do not match expected values."
