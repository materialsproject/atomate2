import openff.toolkit as tk
import pytest
from openff.interchange.components._packmol import pack_box
from openff.units import unit

from atomate2.classical_md.openmm.jobs.opls import (
    create_system_from_xml,
    download_opls_xml,
    generate_faux_interchange,
)
from atomate2.classical_md.utils import create_mol_spec

# skip with pytest


@pytest.mark.skip("annoying test")
def test_download_xml(temp_dir):
    pytest.importorskip("selenium")

    download_opls_xml("CCO", temp_dir / "CCO.xml")

    assert (temp_dir / "CCO.xml").exists()


# xml_path = Path(f"{tmpdir}/{i}.xml")
# with open(xml_path, "w") as f:
#     f.write(xml_string)
#
# incremented_file = Path(f"{tmpdir}/{i}_incremented.xml")
# increment_atom_types_and_classes(xml_path, incremented_file, int(i * 1000))
# incremented_xml_files.append(incremented_file)


def test_create_system_from_xml(classical_md_data):
    opls_xmls = classical_md_data / "opls_xml_files"

    # load strings of xml files into dict
    ff_xmls = [
        (opls_xmls / "CCO.xml").read_text(),
        (opls_xmls / "CO.xml").read_text(),
    ]

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

    create_system_from_xml(topology, ff_xmls)


def test_generate_faux_interchange(classical_md_data, run_job):
    mol_specs = [
        create_mol_spec("CCO", 10, name="ethanol", charge_method="mmff94"),
        create_mol_spec("CO", 300, name="water", charge_method="mmff94"),
    ]

    ff_xmls = [
        (classical_md_data / "opls_xml_files" / "CCO.xml").read_text(),
        (classical_md_data / "opls_xml_files" / "CO.xml").read_text(),
    ]

    job = generate_faux_interchange(mol_specs, 1.0, ff_xmls)
    task_doc = run_job(job)
    assert len(task_doc.molecule_specs) == 2
    assert task_doc.molecule_specs[0].name == "ethanol"
    assert task_doc.molecule_specs[0].count == 10
    assert task_doc.molecule_specs[1].name == "water"
    assert task_doc.molecule_specs[1].count == 300
