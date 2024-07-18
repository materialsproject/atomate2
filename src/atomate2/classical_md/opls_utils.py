"""Utilities for working with the OPLS forcefield in OpenMM."""

from __future__ import annotations

import os
import re
import tempfile
import time
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import openff.toolkit as tk
from openff.interchange.components._packmol import pack_box
from openff.units import unit
from openmm import System  # noqa: TCH002
from openmm.app import ForceField
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

from atomate2.classical_md.utils import create_mol_spec_list


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
    input_mol_specs: list[dict], mass_density: float, xml_files: list[Path | str]
) -> System:
    """Create an OpenMM system from a list of molecule specifications and XML files."""
    xml_paths = [Path(file) for file in xml_files]

    with tempfile.TemporaryDirectory() as tmpdir:
        mol_specs = create_mol_spec_list(input_mol_specs)

        pack_box_kwargs: dict = {}
        topology = pack_box(
            molecules=[tk.Molecule.from_json(spec.openff_mol) for spec in mol_specs],
            number_of_copies=[spec.count for spec in mol_specs],
            mass_density=mass_density * unit.grams / unit.milliliter,
            **pack_box_kwargs,
        )

        openmm_topology = topology.to_openmm()

        incremented_xml_files = []
        for i, xml_path in enumerate(xml_paths):
            incremented_file = Path(f"{tmpdir}/{xml_path.stem}_incremented.xml")
            increment_atom_types_and_classes(xml_path, incremented_file, int(i * 1000))
            incremented_xml_files.append(incremented_file)

        ff = ForceField(incremented_xml_files[0])
        for i, xml_path in enumerate(incremented_xml_files[1:]):
            ff.loadFile(xml_path, resname_prefix=f"{i + 1}")

        return ff.createSystem(openmm_topology)


def download_opls_xml(smiles: str, output_file: str | Path) -> None:
    """Download an OPLS-AA/M XML file from the LigParGen website using Selenium."""
    # Initialize the Chrome driver
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

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
            time.sleep(1)  # Adjust this delay as needed based on the download time

            file = next(Path(tmpdir).iterdir())
            # copy downloaded file to output_file using os
            os.rename(file, output_file)

    finally:
        driver.quit()
