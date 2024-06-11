import pytest
from jobflow import run_locally
from pymatgen.core import Molecule, Structure

from atomate2.vasp.flows.adsorption import AdsorptionMaker


@pytest.fixture()
def test_adsorption(mock_vasp, clean_dir, test_dir):
    # mapping from job name to directory containing test files
    ref_paths = {
        "ads relax bulk": "Au_adsorption/bulk_relax",
        "ads relax mol": "Au_adsorption/mol",
        "ads relax slab": "Au_adsorption/slab",
        "ads static mol": "Au_adsorption/mol_static",
        "ads static slab": "Au_adsorption/slab_static",
        "ads relax 1/3": "Au_adsorption/ads_relax_1_3",
        "ads relax 2/3": "Au_adsorption/ads_relax_2_3",
        "ads relax 3/3": "Au_adsorption/ads_relax_3_3",
        "ads static 1/3": "Au_adsorption/ads_static_1_3",
        "ads static 2/3": "Au_adsorption/ads_static_2_3",
        "ads static 3/3": "Au_adsorption/ads_rstatic_3_3",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "ads relax bulk": {"incar_settings": ["NSW", "ISMEAR", "ISIF"]},
        "ads relax mol": {"incar_settings": ["NSW", "ISMEAR", "ISIF"]},
        "ads relax slab": {"incar_settings": ["NSW", "ISMEAR", "ISIF"]},
        "ads static mol": {"incar_settings": ["NSW", "ISMEAR", "ISIF"]},
        "ads static slab": {"incar_settings": ["NSW", "ISMEAR", "ISIF"]},
        "ads relax 1/3": {"incar_settings": ["NSW", "ISMEAR", "ISIF"]},
        "ads relax 2/3": {"incar_settings": ["NSW", "ISMEAR", "ISIF"]},
        "ads relax 3/3": {"incar_settings": ["NSW", "ISMEAR", "ISIF"]},
        "ads static 1/3": {"incar_settings": ["NSW", "ISMEAR", "ISIF"]},
        "ads static 2/3": {"incar_settings": ["NSW", "ISMEAR", "ISIF"]},
        "ads static 3/3": {"incar_settings": ["NSW", "ISMEAR", "ISIF"]},
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
    assert len(responses) == 9, "Unexpected number of jobs in the flow."

    # Verify job names and order
    expected_job_names = [
        "molecule relaxation maker - molecule relaxation job",
        "molecule static maker - molecule static job",
        "bulk relaxation maker - bulk relaxation job",
        "generate_slab",
        "generate_adslabs",
        "adsorption relaxation maker - slab relaxation job",
        "adsorption static maker - slab static job",
        "run_adslabs_job",
        "adsorption relaxation maker - configuration 0",
        "adsorption relaxation maker - configuration 1",
        "adsorption relaxation maker - configuration 2",
        "adsorption static maker - static configuration 0",
        "adsorption static maker - static configuration 1",
        "adsorption static maker - static configuration 2",
        "adsorption_calculations",
        "store_inputs",
    ]
    for response, expected_name in zip(responses, expected_job_names):
        assert (
            response.name == expected_name
        ), f"Job '{response.name}' does not match expected '{expected_name}'."

    assert flow[-2].uuid in responses
    assert flow[-1].uuid in responses
