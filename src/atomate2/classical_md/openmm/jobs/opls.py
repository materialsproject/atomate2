"""Utilities for working with the OPLS forcefield in OpenMM."""

from __future__ import annotations

import io
import re
import tempfile
import time
import warnings
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import openff.toolkit as tk
from emmet.core.classical_md import ClassicalMDTaskDocument, MoleculeSpec
from emmet.core.classical_md.openmm import OpenMMInterchange
from emmet.core.vasp.task_valid import TaskState
from jobflow import Response
from openff.interchange.components._packmol import pack_box
from openff.units import unit
from openmm import Context, LangevinMiddleIntegrator, System, XmlSerializer
from openmm.app import ForceField
from openmm.app.forcefield import PME
from openmm.app.pdbxfile import PDBxFile
from openmm.unit import kelvin, picoseconds

from atomate2.classical_md.core import openff_job
from atomate2.classical_md.utils import (
    create_mol_spec_list,
    merge_specs_by_name_and_smiles,
)

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.common.by import By
    from webdriver_manager.chrome import ChromeDriverManager

except ImportError:
    warnings.warn(
        "The `selenium` package is not installed. "
        "It's required to run the opls web scraper.",
        stacklevel=1,
    )


def increment_atom_types_and_classes(
    input_path: Path, output_path: Path, increment: int = 1000
) -> None:
    """
    Increment atom types in an OpenMM XML file.

    This will write out a new file with all atom types incremented.
    This is necessary to avoid clashing atom types when combining
    multiple XML files downloaded from LigParGen.
    """

    # tbh this is GPT garbage
    def increment_name(name: str) -> str:
        match = re.match(r"(\D+)(\d+)", name)
        if match:
            prefix, num = match.groups()
            return f"{prefix}{int(num) + increment}"
        return name

    tree = ElementTree.parse(input_path)  # noqa: S314
    root = tree.getroot()

    atom_type_map = {}
    class_map = {}

    for atom_type in root.findall(".//AtomTypes/Type"):
        old_name = atom_type.attrib["name"]
        new_name = increment_name(old_name)
        atom_type_map[old_name] = new_name
        atom_type.attrib["name"] = new_name

        old_class = atom_type.attrib["class"]
        new_class = increment_name(old_class)
        class_map[old_class] = new_class
        atom_type.attrib["class"] = new_class

    for atom in root.findall(".//Residues/Residue/Atom"):
        old_type = atom.attrib["type"]
        atom.attrib["type"] = atom_type_map[old_type]

    for bond in root.findall(".//HarmonicBondForce/Bond"):
        bond.attrib["class1"] = class_map[bond.attrib["class1"]]
        bond.attrib["class2"] = class_map[bond.attrib["class2"]]

    for angle in root.findall(".//HarmonicAngleForce/Angle"):
        angle.attrib["class1"] = class_map[angle.attrib["class1"]]
        angle.attrib["class2"] = class_map[angle.attrib["class2"]]
        angle.attrib["class3"] = class_map[angle.attrib["class3"]]

    for torsion in root.findall(".//PeriodicTorsionForce/Proper"):
        torsion.attrib["class1"] = class_map[torsion.attrib["class1"]]
        torsion.attrib["class2"] = class_map[torsion.attrib["class2"]]
        torsion.attrib["class3"] = class_map[torsion.attrib["class3"]]
        torsion.attrib["class4"] = class_map[torsion.attrib["class4"]]

    for torsion in root.findall(".//PeriodicTorsionForce/Improper"):
        torsion.attrib["class1"] = class_map[torsion.attrib["class1"]]
        torsion.attrib["class2"] = class_map[torsion.attrib["class2"]]
        torsion.attrib["class3"] = class_map[torsion.attrib["class3"]]
        torsion.attrib["class4"] = class_map[torsion.attrib["class4"]]

    for atom in root.findall(".//NonbondedForce/Atom"):
        old_type = atom.attrib["type"]
        atom.attrib["type"] = atom_type_map[old_type]

    tree.write(output_path)


def create_system_from_xml(
    topology: tk.Topology,
    ff_xmls: list[str],
) -> System:
    """Create an OpenMM system from a list of molecule specifications and XML files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # write the XML files to the temporary directory
        incremented_xml_files = []
        for i, xml_string in enumerate(ff_xmls):
            xml_path = Path(f"{tmpdir}/{i}.xml")
            with open(xml_path, "w") as f:
                f.write(xml_string)

            incremented_file = Path(f"{tmpdir}/{i}_incremented.xml")
            increment_atom_types_and_classes(xml_path, incremented_file, int(i * 1000))
            incremented_xml_files.append(incremented_file)

        ff = ForceField(incremented_xml_files[0])
        for i, xml_path in enumerate(incremented_xml_files[1:]):
            ff.loadFile(xml_path, resname_prefix=f"{i + 1}")

        return ff.createSystem(topology.to_openmm(), nonbondedMethod=PME)


def download_opls_xml(
    names_smiles: dict[str, str], output_dir: str | Path, overwrite_files: bool = False
) -> None:
    """Download an OPLS-AA/M XML file from the LigParGen website using Selenium."""
    # Initialize the Chrome driver
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

    for name, smiles in names_smiles.items():
        final_file = Path(output_dir) / f"{name}.xml"
        if final_file.exists() and not overwrite_files:
            continue
        try:
            # Specify the directory where you want to download files
            with tempfile.TemporaryDirectory() as tmpdir:
                download_dir = tmpdir

                # Set up Chrome options
                chrome_options = webdriver.ChromeOptions()
                prefs = {"download.default_directory": download_dir}
                chrome_options.add_experimental_option("prefs", prefs)

                # Initialize Chrome with the options
                driver = webdriver.Chrome(options=chrome_options)

                # Open the first webpage
                driver.get("https://zarbi.chem.yale.edu/ligpargen/")

                # Find the SMILES input box and enter the SMILES code
                smiles_input = driver.find_element(By.ID, "smiles")
                smiles_input.send_keys(smiles)

                # Find and click the "Submit Molecule" button
                submit_button = driver.find_element(
                    By.XPATH,
                    '//button[@type="submit" and contains(text(), "Submit Molecule")]',
                )
                submit_button.click()

                # Wait for the second page to load
                # time.sleep(2)  # Adjust this delay as needed based on the loading time

                # Find and click the "XML" button under Downloads and OpenMM
                xml_button = driver.find_element(
                    By.XPATH, '//input[@type="submit" and @value="XML"]'
                )
                xml_button.click()

                # Wait for the file to download
                time.sleep(0.3)  # Adjust as needed based on the download time

                file = next(Path(tmpdir).iterdir())
                # copy downloaded file to output_file using os
                Path(file).rename(final_file)

        except Exception as e:  # noqa: BLE001
            warnings.warn(
                f"{name} ({smiles}) failed to download because an error occurred: {e}",
                stacklevel=1,
            )

    driver.quit()


@openff_job
def generate_openmm_interchange(
    input_mol_specs: list[MoleculeSpec | dict],
    mass_density: float,
    ff_xmls: list[str],
    pack_box_kwargs: dict = None,
    tags: list[str] = None,
) -> Response:
    """Generate a OpenMMInterchange with arbitrary XML files."""
    # TODO: warn if using unsupported properties
    mol_specs = create_mol_spec_list(input_mol_specs)

    mol_specs = merge_specs_by_name_and_smiles(mol_specs)

    pack_box_kwargs = pack_box_kwargs or {}
    topology = pack_box(
        molecules=[tk.Molecule.from_json(spec.openff_mol) for spec in mol_specs],
        number_of_copies=[spec.count for spec in mol_specs],
        mass_density=mass_density * unit.grams / unit.milliliter,
        **pack_box_kwargs,
    )

    system = create_system_from_xml(topology, ff_xmls)

    # these values don't actually matter because integrator is only
    # used to generate the state
    integrator = LangevinMiddleIntegrator(
        298 * kelvin, 1 / picoseconds, 1 * picoseconds
    )
    context = Context(system, integrator)
    context.setPositions(topology.get_positions().magnitude / 10)
    state = context.getState(getPositions=True)

    with io.StringIO() as s:
        PDBxFile.writeFile(
            topology.to_openmm(), np.zeros(shape=(topology.n_atoms, 3)), file=s
        )
        s.seek(0)
        pdb = s.read()

    interchange = OpenMMInterchange(
        system=XmlSerializer.serialize(system),
        state=XmlSerializer.serialize(state),
        topology=pdb,
    )

    interchange_json = interchange.json()
    interchange_bytes = interchange_json.encode("utf-8")

    task_doc = ClassicalMDTaskDocument(
        state=TaskState.SUCCESS,
        interchange=interchange_bytes,
        molecule_specs=mol_specs,
        force_field="opls",  # TODO: change to flexible value
        tags=tags,
    )

    return Response(output=task_doc)
