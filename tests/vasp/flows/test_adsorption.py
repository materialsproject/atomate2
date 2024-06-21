from jobflow import run_locally
from pymatgen.core import Molecule, Structure

from atomate2.vasp.flows.adsorption import AdsorptionMaker


def test_adsorption(mock_vasp, clean_dir, test_dir):
    # mapping from job name to directory containing test files
    ref_paths = {
        "bulk relaxation maker - bulk relaxation job": "Au_adsorption/bulk_relax",
        "molecule relaxation maker - molecule relaxation job": "Au_adsorption/mol_relax",
        "adsorption relaxation maker - slab relaxation job": "Au_adsorption/slab_relax",
        "molecule static maker - molecule static job": "Au_adsorption/mol_static",
        "adsorption static maker - slab static job": "Au_adsorption/slab_static",
        "adsorption relaxation maker - configuration 0": "Au_adsorption/ads_relax_1_3",
        "adsorption relaxation maker - configuration 1": "Au_adsorption/ads_relax_2_3",
        "adsorption relaxation maker - configuration 2": "Au_adsorption/ads_relax_3_3",
        "adsorption static maker - static configuration 0": "Au_adsorption/ads_static_1_3",
        "adsorption static maker - static configuration 1": "Au_adsorption/ads_static_2_3",
        "adsorption static maker - static configuration 2": "Au_adsorption/ads_rstatic_3_3",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "bulk relaxation maker - bulk relaxation job": {
            "incar_settings": ["NSW", "ISIF"]
        },
        "molecule relaxation maker - molecule relaxation job": {
            "incar_settings": ["NSW", "ISIF"]
        },
        "adsorption relaxation maker - slab relaxation job": {
            "incar_settings": ["NSW", "ISIF"]
        },
        "molecule static maker - molecule static job": {
            "incar_settings": ["NSW", "ISIF"]
        },
        "adsorption static maker - slab static job": {
            "incar_settings": ["NSW", "ISIF"]
        },
        "adsorption relaxation maker - configuration 0": {
            "incar_settings": ["NSW", "ISIF"]
        },
        "adsorption relaxation maker - configuration 1": {
            "incar_settings": ["NSW", "ISIF"]
        },
        "adsorption relaxation maker - configuration 2": {
            "incar_settings": ["NSW", "ISIF"]
        },
        "adsorption static maker - static configuration 0": {
            "incar_settings": ["NSW", "ISIF"]
        },
        "adsorption static maker - static configuration 1": {
            "incar_settings": ["NSW", "ISIF"]
        },
        "adsorption static maker - static configuration 2": {
            "incar_settings": ["NSW", "ISIF"]
        },
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

    # Check that the correct number of jobs are created
    assert (
        len(responses) == 16 or len(responses) == 9
    ), "Unexpected number of jobs in the flow."

    # Verify job names and order
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
    for response, expected_name in zip(responses, expected_job_names):
        assert (
            response.name == expected_name
        ), f"Job '{response.name}' does not match expected '{expected_name}'."

    # assert flow[-2].uuid in responses
    # assert flow[-1].uuid in responses
