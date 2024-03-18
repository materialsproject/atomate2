from monty.serialization import MontyEncoder, MontyDecoder
import openff.toolkit as tk
import json

import atomate2.classical_md


def test_monty_serialization():
    # Create the molecule using openff.toolkit
    original_mol = tk.Molecule.from_smiles("O")
    mol_dict = original_mol.to_dict()
    mol_dict["@module"] = "openff.toolkit.topology"
    mol_dict["@class"] = "Molecule"

    original_mol.as_dict()

    # Serialize the molecule using MontyEncoder
    serialized_mol = json.dumps(original_mol, cls=MontyEncoder)

    # Deserialize the molecule using MontyDecoder
    deserialized_mol = json.loads(serialized_mol, cls=MontyDecoder)

    # Convert both molecules to their canonical SMILES for comparison
    original_smiles = original_mol.to_smiles(
        isomeric=True, explicit_hydrogens=False, mapped=False
    )
    deserialized_smiles = deserialized_mol.to_smiles(
        isomeric=True, explicit_hydrogens=False, mapped=False
    )
