from pymatgen.core import Molecule
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element

from atomate2.common.schemas.molecule import MoleculeMetadata

molecule = Molecule(
    species=["Mg", "O"],
    coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
)


def test_from_molecule(test_dir):
    metadata = MoleculeMetadata.from_molecule(molecule).dict()
    assert metadata["nsites"] == 2
    assert metadata["elements"] == [Element("Mg"), Element("O")]
    assert metadata["nelements"] == 2
    assert metadata["composition"] == Composition("MgO")
    assert metadata["composition_reduced"] == Composition("MgO").reduced_composition
    assert metadata["formula_pretty"] == "MgO"
    assert metadata["formula_anonymous"] == "AB"
    assert metadata["chemsys"] == "Mg-O"
    assert metadata["point_group"] == "C*v"
    assert metadata["charge"] == 0
    assert metadata["spin_multiplicity"] == 1
    assert metadata["nelectrons"] == 20


def test_from_comp(test_dir):
    metadata = MoleculeMetadata.from_composition(molecule.composition).dict()
    assert metadata["elements"] == [Element("Mg"), Element("O")]
    assert metadata["nelements"] == 2
    assert metadata["composition"] == Composition("MgO")
    assert metadata["composition_reduced"] == Composition("MgO").reduced_composition
    assert metadata["formula_pretty"] == "MgO"
    assert metadata["formula_anonymous"] == "AB"
    assert metadata["chemsys"] == "Mg-O"
