import tempfile
from pathlib import Path

import openff.toolkit as tk
import pytest
from jobflow import run_locally
from openff.interchange import Interchange
from openff.interchange.components._packmol import pack_box
from openff.toolkit import ForceField
from openff.units import unit

from atomate2.openff.utils import create_mol_spec, merge_specs_by_name_and_smiles


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def run_job(temp_dir):
    def run_job(job):
        response_dict = run_locally(job, ensure_success=True, root_dir=temp_dir)
        return list(response_dict.values())[-1][1].output

    return run_job


@pytest.fixture
def openmm_data(test_dir):
    return test_dir / "openmm"


@pytest.fixture(scope="package")
def interchange():
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

    return Interchange.from_smirnoff(
        force_field=ForceField("openff_unconstrained-2.1.1.offxml"),
        topology=topology,
        charge_from_molecules=[
            tk.Molecule.from_json(spec.openff_mol) for spec in mol_specs
        ],
        allow_nonintegral_charges=True,
    )


@pytest.fixture
def output_dir(test_dir):
    return test_dir / "classical_md" / "output_dir"
