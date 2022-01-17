from atomate2.common.schemas.structure import StructureMetadata
from pymatgen.core import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element
from pytest import approx

structure = Structure(
    lattice=[[0, 2.13, 2.13], [2.13, 0, 2.13], [2.13, 2.13, 0]],
    species=["Mg", "O"],
    coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
)


def test_from_structure(test_dir):
    metadata = StructureMetadata.from_structure(structure).dict()
    assert metadata["nsites"] == 2
    assert metadata["elements"] == [Element("Mg"), Element("O")]
    assert metadata["nelements"] == 2
    assert metadata["composition"] == Composition("MgO")
    assert metadata["composition_reduced"] == Composition("MgO").reduced_composition
    assert metadata["formula_pretty"] == "MgO"
    assert metadata["formula_anonymous"] == "AB"
    assert metadata["chemsys"] == "Mg-O"
    assert metadata["volume"] == approx(19.327194)
    assert metadata["density"] == approx(3.4628426017699754)
    assert metadata["density_atomic"] == approx(9.663597)
    assert metadata["symmetry"]["symbol"] == "Fm-3m"
    assert metadata["symmetry"]["number"] == 225
    assert metadata["symmetry"]["point_group"] == "m-3m"
    assert metadata["symmetry"]["symprec"] == 0.1
    assert type(metadata["symmetry"]["version"]) is str


def test_from_composition(test_dir):
    metadata = StructureMetadata.from_composition(structure.composition).dict()
    assert metadata["elements"] == [Element("Mg"), Element("O")]
    assert metadata["nelements"] == 2
    assert metadata["composition"] == Composition("MgO")
    assert metadata["composition_reduced"] == Composition("MgO").reduced_composition
    assert metadata["formula_pretty"] == "MgO"
    assert metadata["formula_anonymous"] == "AB"
    assert metadata["chemsys"] == "Mg-O"
