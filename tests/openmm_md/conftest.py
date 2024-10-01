import pytest
from emmet.core.openmm import OpenMMInterchange
from jobflow import run_locally


@pytest.fixture
def run_job(tmp_path):
    def run_job(job):
        response_dict = run_locally(job, ensure_success=True, root_dir=tmp_path)
        return list(response_dict.values())[-1][1].output

    return run_job


@pytest.fixture(scope="package")
def openmm_data(test_dir):
    return test_dir / "openmm"


@pytest.fixture(scope="package")
def interchange(openmm_data):
    # we use openff to generate the interchange object that we test on
    # but we don't want to create a logical dependency on openff, in
    # case the user has another way of generating the interchange object
    regenerate_test_data = False
    if regenerate_test_data:
        import openff.toolkit as tk
        from openff.interchange import Interchange
        from openff.interchange.components._packmol import pack_box
        from openff.toolkit import ForceField
        from openff.units import unit

        from atomate2.openff.utils import (
            create_mol_spec,
            merge_specs_by_name_and_smiles,
        )
        from atomate2.openmm.utils import openff_to_openmm_interchange

        o = create_mol_spec("O", 300, charge_method="mmff94")
        cco = create_mol_spec("CCO", 10, charge_method="mmff94")
        cco2 = create_mol_spec("CCO", 20, name="cco2", charge_method="mmff94")
        mol_specs = [o, cco, cco2]
        mol_specs.sort(
            key=lambda x: tk.Molecule.from_json(x.openff_mol).to_smiles() + x.name
        )

        topology = pack_box(
            molecules=[tk.Molecule.from_json(spec.openff_mol) for spec in mol_specs],
            number_of_copies=[spec.count for spec in mol_specs],
            mass_density=0.8 * unit.grams / unit.milliliter,
        )

        mol_specs = merge_specs_by_name_and_smiles(mol_specs)

        openff_interchange = Interchange.from_smirnoff(
            force_field=ForceField("openff_unconstrained-2.1.1.offxml"),
            topology=topology,
            charge_from_molecules=[
                tk.Molecule.from_json(spec.openff_mol) for spec in mol_specs
            ],
            allow_nonintegral_charges=True,
        )

        openmm_interchange = openff_to_openmm_interchange(openff_interchange)

        with open(openmm_data / "interchange.json", "w") as file:
            file.write(openmm_interchange.model_dump_json())

    with open(openmm_data / "interchange.json") as file:
        return OpenMMInterchange.model_validate_json(file.read())


@pytest.fixture
def output_dir(test_dir):
    return test_dir / "classical_md" / "output_dir"
