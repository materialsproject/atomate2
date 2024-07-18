import pytest

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

    mol_specs_dicts = [
        {"smile": "CCO", "count": 10, "name": "ethanol", "charge_method": "mmff94"},
        {"smile": "CO", "count": 20, "name": "water", "charge_method": "mmff94"},
    ]

    create_system_from_xml(
        mol_specs_dicts,
        1.0,
        [
            opls_xmls / "CCO.xml",
            opls_xmls / "CO.xml",
        ],
    )
