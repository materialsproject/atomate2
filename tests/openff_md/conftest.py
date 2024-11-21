import numpy as np
import openff.toolkit as tk
import pytest
from jobflow import run_locally
from openff.interchange import Interchange
from openff.interchange.components._packmol import pack_box
from openff.toolkit import ForceField
from openff.units import unit

from atomate2.openff.utils import create_mol_spec, merge_specs_by_name_and_smiles


@pytest.fixture
def run_job(tmp_path):
    def run_job(job):
        response_dict = run_locally(job, ensure_success=True, root_dir=tmp_path)
        return list(response_dict.values())[-1][1].output

    return run_job


@pytest.fixture
def mol_specs_small():
    return [
        create_mol_spec("CCO", 10, name="ethanol", charge_method="mmff94"),
        create_mol_spec("O", 20, name="water", charge_method="mmff94"),
    ]


@pytest.fixture
def openff_data(test_dir):
    return test_dir / "openff"


@pytest.fixture
def mol_files(openff_data):
    geo_dir = openff_data / "molecule_charge_files"
    return {
        "CCO_xyz": str(geo_dir / "CCO.xyz"),
        "CCO_charges": str(geo_dir / "CCO.npy"),
        "FEC_r_xyz": str(geo_dir / "FEC-r.xyz"),
        "FEC_s_xyz": str(geo_dir / "FEC-s.xyz"),
        "FEC_charges": str(geo_dir / "FEC.npy"),
        "PF6_xyz": str(geo_dir / "PF6.xyz"),
        "PF6_charges": str(geo_dir / "PF6.npy"),
        "Li_charges": str(geo_dir / "Li.npy"),
        "Li_xyz": str(geo_dir / "Li.xyz"),
    }


@pytest.fixture
def mol_specs_salt(mol_files):
    charges = np.load(mol_files["PF6_charges"])
    return [
        create_mol_spec("CCO", 10, name="ethanol", charge_method="mmff94"),
        create_mol_spec("O", 20, name="water", charge_method="mmff94"),
        create_mol_spec("[Li+]", 5, name="li", charge_method="mmff94"),
        create_mol_spec(
            "F[P-](F)(F)(F)(F)F",
            5,
            name="pf6",
            partial_charges=charges,
            geometry=mol_files["PF6_xyz"],
        ),
    ]


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
