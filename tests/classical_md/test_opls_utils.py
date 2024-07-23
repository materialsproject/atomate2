import openff.toolkit as tk
import pytest
from openff.interchange.components._packmol import pack_box
from openff.units import unit

from atomate2.classical_md.opls_utils import create_system_from_xml, download_opls_xml

# skip with pytest


@pytest.mark.skip("annoying test")
def test_download_xml(temp_dir):
    pytest.importorskip("selenium")

    download_opls_xml("CCO", temp_dir / "CCO.xml")

    assert (temp_dir / "CCO.xml").exists()


def test_create_system_from_xml(classical_md_data):
    opls_xmls = classical_md_data / "opls_xml_files"

    # uncomment to regenerate data
    # download_opls_xml("CCO", opls_xmls / "CCO.xml")
    # download_opls_xml("CO", opls_xmls / "CO.xml")

    mol_specs = [
        {"smile": "CCO", "count": 10},
        {"smile": "CO", "count": 20},
    ]

    topology = pack_box(
        molecules=[tk.Molecule.from_smiles(spec["smile"]) for spec in mol_specs],
        number_of_copies=[spec["count"] for spec in mol_specs],
        mass_density=0.8 * unit.grams / unit.milliliter,
    )

    create_system_from_xml(
        topology,
        [
            opls_xmls / "CCO.xml",
            opls_xmls / "CO.xml",
        ],
    )
