from jobflow import run_locally
from pymatgen.core import Molecule, Structure

from atomate2.vasp.flows.adsorption import AdsorptionMaker


def test_adsorption(mock_vasp, clean_dir, test_dir):
    # mapping from job name to directory containing test files
    ref_paths = {
        "bulk relaxation maker - bulk relaxation job": "Au_adsorption/bulk_relax",
        "molecule relaxation maker - molecule relaxation job": (
            "Au_adsorption/mol_relax"
        ),
        "adsorption relaxation maker - slab relaxation job": (
            "Au_adsorption/slab_relax"
        ),
        "molecule static maker - molecule static job": "Au_adsorption/mol_static",
        "adsorption static maker - slab static job": "Au_adsorption/slab_static",
        "adsorption relaxation maker - configuration 0": (
            "Au_adsorption/ads_relax_1_3"
        ),
        "adsorption relaxation maker - configuration 1": (
            "Au_adsorption/ads_relax_2_3"
        ),
        "adsorption relaxation maker - configuration 2": (
            "Au_adsorption/ads_relax_3_3"
        ),
        "adsorption static maker - static configuration 0": (
            "Au_adsorption/ads_static_1_3"
        ),
        "adsorption static maker - static configuration 1": (
            "Au_adsorption/ads_static_2_3"
        ),
        "adsorption static maker - static configuration 2": (
            "Au_adsorption/ads_static_3_3"
        ),
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

    flow = AdsorptionMaker().make(
        molecule=molecule, structure=bulk_structure, min_lw=5.0, min_slab_size=4.0
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
        "bulk relaxation maker - bulk relaxation job",
        "molecule relaxation maker - molecule relaxation job",
        "adsorption relaxation maker - slab relaxation job",
        "molecule static maker - molecule static job",
        "adsorption static maker - slab static job",
        "adsorption relaxation maker - configuration 0",
        "adsorption relaxation maker - configuration 1",
        "adsorption relaxation maker - configuration 2",
        "adsorption static maker - static configuration 0",
        "adsorption static maker - static configuration 1",
        "adsorption static maker - static configuration 2",
    ]
    for actual_name in expected_job_names:
        assert actual_name in job_names, f"Job '{actual_name}' not found."

    assert flow[-1].uuid in responses, "ads calculation job not found"

    adsorption_calculation_job = responses.get(flow[-1].uuid)
    adsorption_energy = [
        energy
        for response in adsorption_calculation_job.values()
        for energy in response.output.get("adsorption_energy", [])
    ]

    assert isinstance(adsorption_energy, list)

    adsorption_energy.sort()

    assert adsorption_energy == [
        -3.0084328299999967,
        -2.9288308699999916,
        -2.092973299999997,
    ], "adsorption energy is inaccurate or not found in response"
