#!/usr/bin/env python
"""
CLI of siesta-pseudos inspired by abips.py Script to download and install pseudopotential tables from the web or use local files.
"""

import os
import click
import requests
import tarfile
import shutil
from rich.console import Console
from rich.table import Table
from rich.text import Text

# Import plotting functions from same package
from atomate2.siesta.cli.pseudo.plot_pseudopotential import (
    parse_psml,
    plot_wavefunctions,
    plot_potentials,
    plot_3d_potential,
    plot_occupation_map,
    plot_density,
)

# Initialize rich console
console = Console()

# Define the URL to the pseudos directory (for fallback downloading)
BASE_URL = (
    "https://raw.githubusercontent.com/arsalan-akhtar/atomate2siesta/main/pseudos/"
)

# Optional: Define your GitHub Personal Access Token
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# List of available pseudos with local and remote file information
PSEUDOS = {
    "ONCVPSP-PBE-FR-PDv0.4-Standard": {
        "filename": "nc-fr-04_pbe_standard_psml.tgz",
        "local_path": "nc-fr-04_pbe_standard_psml.tgz",
        "xc_name": "PBE",
        "relativity_type": "FR",
        "version": "0.4",
        "elements": {
            "Ag",
            "Al",
            "Ar",
            "As",
            "Au",
            "B",
            "Ba",
            "Be",
            "Bi",
            "Br",
            "C",
            "Ca",
            "Cd",
            "Cl",
            "Co",
            "Cr",
            "Cs",
            "Cu",
            "F",
            "Fe",
            "Ga",
            "Ge",
            "H",
            "He",
            "Hf",
            "Hg",
            "I",
            "In",
            "Ir",
            "K",
            "Kr",
            "Li",
            "Mg",
            "Mn",
            "Mo",
            "N",
            "Na",
            "Nb",
            "Ne",
            "Ni",
            "O",
            "Os",
            "P",
            "Pb",
            "Pd",
            "Po",
            "Pt",
            "Rb",
            "Re",
            "Rh",
            "Rn",
            "Ru",
            "S",
            "Sb",
            "Sc",
            "Se",
            "Si",
            "Sn",
            "Sr",
            "Ta",
            "Tc",
            "Te",
            "Ti",
            "Tl",
            "V",
            "W",
            "Xe",
            "Y",
            "Zn",
            "Zr",
        },
        "url": "http://www.pseudo-dojo.org/pseudos/nc-fr-04_pbe_standard_psml.tgz",
    },
    "ONCVPSP-PBE-FR-PDv0.4-Stringent": {
        "filename": "nc-fr-04_pbe_stringent_psml.tgz",
        "local_path": "nc-fr-04_pbe_stringent_psml.tgz",
        "xc_name": "PBE",
        "relativity_type": "FR",
        "version": "0.4",
        "elements": {
            "Ag",
            "Al",
            "Ar",
            "As",
            "Au",
            "B",
            "Ba",
            "Be",
            "Bi",
            "Br",
            "C",
            "Ca",
            "Cd",
            "Cl",
            "Co",
            "Cr",
            "Cs",
            "Cu",
            "F",
            "Fe",
            "Ga",
            "Ge",
            "H",
            "He",
            "Hf",
            "Hg",
            "I",
            "In",
            "Ir",
            "K",
            "Kr",
            "La",
            "Li",
            "Lu",
            "Mg",
            "Mn",
            "Mo",
            "N",
            "Na",
            "Nb",
            "Ne",
            "Ni",
            "O",
            "Os",
            "P",
            "Pb",
            "Pd",
            "Po",
            "Pt",
            "Rb",
            "Re",
            "Rh",
            "Rn",
            "Ru",
            "S",
            "Sb",
            "Sc",
            "Se",
            "Si",
            "Sn",
            "Sr",
            "Ta",
            "Tc",
            "Te",
            "Ti",
            "Tl",
            "V",
            "W",
            "Xe",
            "Y",
            "Zn",
            "Zr",
        },
        "url": "http://www.pseudo-dojo.org/pseudos/nc-fr-04_pbe_stringent_psml.tgz",
    },
    "ONCVPSP-PBE-SR-PDv0.4.3plus-Standard": {
        "filename": "nc-sr-04-3plus_pbe_standard_psml.tgz",
        "local_path": "nc-sr-04-3plus_pbe_standard_psml.tgz",
        "xc_name": "PBE",
        "relativity_type": "SR",
        "version": "0.4.3",
        "elements": {
            "Ce",
            "Dy",
            "Er",
            "Eu",
            "Gd",
            "Ho",
            "Lu",
            "Nd",
            "Pm",
            "Pr",
            "Sm",
            "Tb",
            "Tm",
            "Yb",
        },
        "url": "http://www.pseudo-dojo.org/pseudos/nc-sr-04-3plus_pbe_standard_psml.tgz",
    },
    "ONCVPSP-PBE-SR-PDv0.4-Standard": {
        "filename": "nc-sr-04_pbe_standard_psml.tgz",
        "local_path": "nc-sr-04_pbe_standard_psml.tgz",
        "xc_name": "PBE",
        "relativity_type": "SR",
        "version": "0.4",
        "elements": {
            "Ag",
            "Al",
            "Ar",
            "As",
            "Au",
            "B",
            "Ba",
            "Be",
            "Bi",
            "Br",
            "C",
            "Ca",
            "Cd",
            "Cl",
            "Co",
            "Cr",
            "Cs",
            "Cu",
            "F",
            "Fe",
            "Ga",
            "Ge",
            "H",
            "He",
            "Hf",
            "Hg",
            "I",
            "In",
            "Ir",
            "K",
            "Kr",
            "La",
            "Li",
            "Lu",
            "Mg",
            "Mn",
            "Mo",
            "N",
            "Na",
            "Nb",
            "Ne",
            "Ni",
            "O",
            "Os",
            "P",
            "Pb",
            "Pd",
            "Po",
            "Pt",
            "Rb",
            "Re",
            "Rh",
            "Rn",
            "Ru",
            "S",
            "Sb",
            "Sc",
            "Se",
            "Si",
            "Sn",
            "Sr",
            "Ta",
            "Tc",
            "Te",
            "Ti",
            "Tl",
            "V",
            "W",
            "Xe",
            "Y",
            "Zn",
            "Zr",
        },
        "url": "http://www.pseudo-dojo.org/pseudos/nc-sr-04_pbe_standard_psml.tgz",
    },
    "ONCVPSP-PBE-SR-PDv0.4-Stringent": {
        "filename": "nc-sr-04_pbe_stringent_psml.tgz",
        "local_path": "nc-sr-04_pbe_stringent_psml.tgz",
        "xc_name": "PBE",
        "relativity_type": "SR",
        "version": "0.4",
        "elements": {
            "Ag",
            "Al",
            "Ar",
            "As",
            "Au",
            "B",
            "Ba",
            "Be",
            "Bi",
            "Br",
            "C",
            "Ca",
            "Cd",
            "Cl",
            "Co",
            "Cr",
            "Cs",
            "Cu",
            "F",
            "Fe",
            "Ga",
            "Ge",
            "H",
            "He",
            "Hf",
            "Hg",
            "I",
            "In",
            "Ir",
            "K",
            "Kr",
            "La",
            "Li",
            "Lu",
            "Mg",
            "Mn",
            "Mo",
            "N",
            "Na",
            "Nb",
            "Ne",
            "Ni",
            "O",
            "Os",
            "P",
            "Pb",
            "Pd",
            "Po",
            "Pt",
            "Rb",
            "Re",
            "Rh",
            "Rn",
            "Ru",
            "S",
            "Sb",
            "Sc",
            "Se",
            "Si",
            "Sn",
            "Sr",
            "Ta",
            "Tc",
            "Te",
            "Ti",
            "Tl",
            "V",
            "W",
            "Xe",
            "Y",
            "Zn",
            "Zr",
        },
        "url": "http://www.pseudo-dojo.org/pseudos/nc-sr-04_pbe_stringent_psml.tgz",
    },
    "ONCVPSP-PBEsol-FR-PDv0.4-Standard": {
        "filename": "nc-fr-04_pbesol_standard_psml.tgz",
        "local_path": "nc-fr-04_pbesol_standard_psml.tgz",
        "xc_name": "PBEsol",
        "relativity_type": "FR",
        "version": "0.4",
        "elements": {
            "Ag",
            "Al",
            "Ar",
            "As",
            "Au",
            "B",
            "Ba",
            "Be",
            "Bi",
            "Br",
            "C",
            "Ca",
            "Cd",
            "Cl",
            "Co",
            "Cr",
            "Cs",
            "Cu",
            "F",
            "Fe",
            "Ga",
            "Ge",
            "H",
            "He",
            "Hf",
            "Hg",
            "I",
            "In",
            "Ir",
            "K",
            "Kr",
            "La",
            "Li",
            "Mg",
            "Mn",
            "Mo",
            "N",
            "Na",
            "Nb",
            "Ne",
            "Ni",
            "O",
            "Os",
            "P",
            "Pb",
            "Pd",
            "Po",
            "Pt",
            "Rb",
            "Re",
            "Rh",
            "Rn",
            "Ru",
            "S",
            "Sb",
            "Sc",
            "Se",
            "Si",
            "Sn",
            "Sr",
            "Ta",
            "Tc",
            "Te",
            "Ti",
            "Tl",
            "V",
            "W",
            "Xe",
            "Y",
            "Zn",
            "Zr",
        },
        "url": "http://www.pseudo-dojo.org/pseudos/nc-fr-04_pbesol_standard_psml.tgz",
    },
    "ONCVPSP-PBEsol-FR-PDv0.4-Stringent": {
        "filename": "nc-fr-04_pbesol_stringent_psml.tgz",
        "local_path": "nc-fr-04_pbesol_stringent_psml.tgz",
        "xc_name": "PBEsol",
        "relativity_type": "FR",
        "version": "0.4",
        "elements": {
            "Ag",
            "Al",
            "Ar",
            "As",
            "Au",
            "B",
            "Ba",
            "Be",
            "Bi",
            "Br",
            "C",
            "Ca",
            "Cd",
            "Cl",
            "Co",
            "Cr",
            "Cs",
            "Cu",
            "F",
            "Fe",
            "Ga",
            "Ge",
            "H",
            "He",
            "Hf",
            "Hg",
            "I",
            "In",
            "Ir",
            "K",
            "Kr",
            "La",
            "Li",
            "Lu",
            "Mg",
            "Mn",
            "Mo",
            "N",
            "Na",
            "Nb",
            "Ne",
            "Ni",
            "O",
            "Os",
            "P",
            "Pb",
            "Pd",
            "Po",
            "Pt",
            "Rb",
            "Re",
            "Rh",
            "Rn",
            "Ru",
            "S",
            "Sb",
            "Sc",
            "Se",
            "Si",
            "Sn",
            "Sr",
            "Ta",
            "Tc",
            "Te",
            "Ti",
            "Tl",
            "V",
            "W",
            "Xe",
            "Y",
            "Zn",
            "Zr",
        },
        "url": "http://www.pseudo-dojo.org/pseudos/nc-fr-04_pbesol_stringent_psml.tgz",
    },
    "ONCVPSP-PBEsol-SR-PDv0.4-Standard": {
        "filename": "nc-sr-04_pbesol_standard_psml.tgz",
        "local_path": "nc-sr-04_pbesol_standard_psml.tgz",
        "xc_name": "PBEsol",
        "relativity_type": "SR",
        "version": "0.4",
        "elements": {
            "Ag",
            "Al",
            "Ar",
            "As",
            "Au",
            "B",
            "Ba",
            "Be",
            "Bi",
            "Br",
            "C",
            "Ca",
            "Cd",
            "Cl",
            "Co",
            "Cr",
            "Cs",
            "Cu",
            "F",
            "Fe",
            "Ga",
            "Ge",
            "H",
            "He",
            "Hf",
            "Hg",
            "I",
            "In",
            "Ir",
            "K",
            "Kr",
            "La",
            "Li",
            "Mg",
            "Mn",
            "Mo",
            "N",
            "Na",
            "Nb",
            "Ne",
            "Ni",
            "O",
            "Os",
            "P",
            "Pb",
            "Pd",
            "Po",
            "Pt",
            "Rb",
            "Re",
            "Rh",
            "Rn",
            "Ru",
            "S",
            "Sb",
            "Sc",
            "Se",
            "Si",
            "Sn",
            "Sr",
            "Ta",
            "Tc",
            "Te",
            "Ti",
            "Tl",
            "V",
            "W",
            "Xe",
            "Y",
            "Zn",
            "Zr",
        },
        "url": "http://www.pseudo-dojo.org/pseudos/nc-sr-04_pbesol_standard_psml.tgz",
    },
    "ONCVPSP-PBEsol-SR-PDv0.4-Stringent": {
        "filename": "nc-sr-04_pbesol_stringent_psml.tgz",
        "local_path": "nc-sr-04_pbesol_stringent_psml.tgz",
        "xc_name": "PBEsol",
        "relativity_type": "SR",
        "version": "0.4",
        "elements": {
            "Ag",
            "Al",
            "Ar",
            "As",
            "Au",
            "B",
            "Ba",
            "Be",
            "Bi",
            "Br",
            "C",
            "Ca",
            "Cd",
            "Cl",
            "Co",
            "Cr",
            "Cs",
            "Cu",
            "F",
            "Fe",
            "Ga",
            "Ge",
            "H",
            "He",
            "Hf",
            "Hg",
            "I",
            "In",
            "Ir",
            "K",
            "Kr",
            "La",
            "Li",
            "Lu",
            "Mg",
            "Mn",
            "Mo",
            "N",
            "Na",
            "Nb",
            "Ne",
            "Ni",
            "O",
            "Os",
            "P",
            "Pb",
            "Pd",
            "Po",
            "Pt",
            "Rb",
            "Re",
            "Rh",
            "Rn",
            "Ru",
            "S",
            "Sb",
            "Sc",
            "Se",
            "Si",
            "Sn",
            "Sr",
            "Ta",
            "Tc",
            "Te",
            "Ti",
            "Tl",
            "V",
            "W",
            "Xe",
            "Y",
            "Zn",
            "Zr",
        },
        "url": "http://www.pseudo-dojo.org/pseudos/nc-sr-04_pbesol_stringent_psml.tgz",
    },
    "ONCVPSP-PBE-SR-PDv0.5-Standard": {
        "filename": "nc-sr-05_pbe_standard_psml.tgz",
        "local_path": "nc-sr-05_pbe_standard_psml.tgz",
        "xc_name": "PBE",
        "relativity_type": "SR",
        "version": "0.5",
        "elements": {
            "Ag",
            "Al",
            "Ar",
            "As",
            "Au",
            "B",
            "Ba",
            "Be",
            "Bi",
            "Br",
            "C",
            "Ca",
            "Cd",
            "Cl",
            "Co",
            "Cr",
            "Cs",
            "Cu",
            "F",
            "Fe",
            "Ga",
            "Ge",
            "H",
            "He",
            "Hf",
            "Hg",
            "I",
            "In",
            "Ir",
            "K",
            "Kr",
            "La",
            "Li",
            "Lu",
            "Mg",
            "Mn",
            "Mo",
            "N",
            "Na",
            "Nb",
            "Ne",
            "Ni",
            "O",
            "Os",
            "P",
            "Pb",
            "Pd",
            "Po",
            "Pt",
            "Rb",
            "Re",
            "Rh",
            "Rn",
            "Ru",
            "S",
            "Sb",
            "Sc",
            "Se",
            "Si",
            "Sn",
            "Sr",
            "Ta",
            "Tc",
            "Te",
            "Ti",
            "Tl",
            "V",
            "W",
            "Xe",
            "Y",
            "Zn",
            "Zr",
        },
        "url": "http://www.pseudo-dojo.org/pseudos/nc-sr-05_pbe_standard_psml.tgz",
    },
    "ONCVPSP-PBE-SR-PDv0.5-Stringent": {
        "filename": "nc-sr-05_pbe_stringent_psml.tgz",
        "local_path": "nc-sr-05_pbe_stringent_psml.tgz",
        "xc_name": "PBE",
        "relativity_type": "SR",
        "version": "0.5",
        "elements": {
            "Ag",
            "Al",
            "Ar",
            "As",
            "Au",
            "B",
            "Ba",
            "Be",
            "Bi",
            "Br",
            "C",
            "Ca",
            "Cd",
            "Cl",
            "Co",
            "Cr",
            "Cs",
            "Cu",
            "F",
            "Fe",
            "Ga",
            "Ge",
            "H",
            "He",
            "Hf",
            "Hg",
            "I",
            "In",
            "Ir",
            "K",
            "Kr",
            "La",
            "Li",
            "Lu",
            "Mg",
            "Mn",
            "Mo",
            "N",
            "Na",
            "Nb",
            "Ne",
            "Ni",
            "O",
            "Os",
            "P",
            "Pb",
            "Pd",
            "Po",
            "Pt",
            "Rb",
            "Re",
            "Rh",
            "Rn",
            "Ru",
            "S",
            "Sb",
            "Sc",
            "Se",
            "Si",
            "Sn",
            "Sr",
            "Ta",
            "Tc",
            "Te",
            "Ti",
            "Tl",
            "V",
            "W",
            "Xe",
            "Y",
            "Zn",
            "Zr",
        },
        "url": "http://www.pseudo-dojo.org/pseudos/nc-sr-05_pbe_stringent_psml.tgz",
    },
}

# Define the directory to store pseudos
PSEUDO_DIR = os.path.expanduser("~/.siesta/pseudos")


def get_local_pseudo_path(pseudo_name):
    """Get the path to a local pseudopotential file in the project directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
    pseudos_dir = os.path.join(project_dir, "pseudos")
    local_file = os.path.join(pseudos_dir, PSEUDOS[pseudo_name]["local_path"])

    # console.print(f"[cyan]DEBUG: Script directory: {script_dir}[/cyan]")
    # console.print(f"[cyan]DEBUG: Project directory: {project_dir}[/cyan]")

    if os.path.exists(pseudos_dir):
        # console.print(f"[cyan]DEBUG: pseudos directory found at {pseudos_dir}[/cyan]")
        # console.print(f"[cyan]DEBUG: Contents of pseudos directory: {os.listdir(pseudos_dir)}[/cyan]")
        pass
    else:
        # console.print(f"[red]DEBUG: pseudos directory not found at {pseudos_dir}[/red]")
        pass

    if os.path.exists(local_file):
        # console.print(f"[cyan]DEBUG: Local pseudos file found at {local_file}[/cyan]")
        return local_file
    # console.print(f"[red]DEBUG: Local pseudos file not found at {local_file}[/red]")
    return None


def download_and_extract_pseudo(pseudo_file_name, pseudo_name, local_only=False):
    """Download or use local pseudo and extract it."""
    pseudo_folder = os.path.join(PSEUDO_DIR, pseudo_name)
    os.makedirs(pseudo_folder, exist_ok=True)
    console.print(f"[green]Created folder: {pseudo_folder}[/green]")
    local_path = os.path.join(pseudo_folder, pseudo_file_name)

    local_pseudo_path = get_local_pseudo_path(pseudo_name)
    if local_pseudo_path:
        console.print(
            f"[green]Found local file for {pseudo_name} at {local_pseudo_path}[/green]"
        )
        shutil.copy(local_pseudo_path, local_path)
    elif local_only:
        console.print(
            f"[red]Local file for {pseudo_name} not found and --local-only specified.[/red]"
        )
        return
    else:
        console.print(
            f"[yellow]Local file for {pseudo_name} not found. Downloading...[/yellow]"
        )
        headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
        url = PSEUDOS[pseudo_name].get("url", f"{BASE_URL}{pseudo_file_name}")
        console.print(f"[blue]Downloading {pseudo_file_name} from {url}...[/blue]")
        response = requests.get(url, headers=headers, stream=True)

        if response.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(response.content)
        else:
            console.print(
                f"[red]Failed to download {pseudo_file_name}. Status code: {response.status_code}[/red]"
            )
            return

    console.print(f"[blue]Extracting {pseudo_file_name} into {pseudo_folder}...[/blue]")
    with tarfile.open(local_path, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            member.name = os.path.basename(member.name)
            tar.extract(member, path=pseudo_folder)

    if os.path.exists(local_path):
        os.remove(local_path)
        console.print(f"[green]Deleted archive {local_path}[/green]")


@click.group()
@click.version_option("0.1.0")
def cli():
    """Command-line interface for Siesta pseudopotential management."""
    pass


@cli.command()
def available():
    """Show available pseudopotential repositories with installation status."""
    console.print(
        "[bold magenta]List of available pseudopotential repositories:[/bold magenta]"
    )
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("ps_generator", style="blue")
    table.add_column("ps_type", style="blue")
    table.add_column("xc_name", style="blue")
    table.add_column("relativity_type", style="blue")
    table.add_column("project_name", style="blue")
    table.add_column("version", style="blue")
    table.add_column("installed", style="blue")
    table.add_column("name", style="blue")

    for name in PSEUDOS:
        pseudo_folder_path = os.path.join(PSEUDO_DIR, name)
        is_installed = "True" if os.path.exists(pseudo_folder_path) else "False"
        installed_text = Text(
            is_installed, style="green" if is_installed == "True" else "red"
        )
        table.add_row(
            "ONCVPSP",
            "NC",
            PSEUDOS[name]["xc_name"],
            PSEUDOS[name]["relativity_type"],
            "PD",
            PSEUDOS[name]["version"],
            installed_text,
            name,
        )

    console.print(table)


@cli.command()
def list():
    """List pseudopotential repos with installation status."""
    if not os.path.exists(PSEUDO_DIR):
        console.print(
            f"[red]Could not find any pseudopotential repository installed in: {PSEUDO_DIR}[/red]"
        )
        return

    installed_pseudos = os.listdir(PSEUDO_DIR)
    console.print("[bold magenta]List of pseudopotential repositories:[/bold magenta]")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("ps_generator", style="blue")
    table.add_column("ps_type", style="blue")
    table.add_column("xc_name", style="blue")
    table.add_column("relativity_type", style="blue")
    table.add_column("project_name", style="blue")
    table.add_column("version", style="blue")
    table.add_column("installed", style="blue")
    table.add_column("name", style="blue")

    for pseudo_name in PSEUDOS:
        pseudo_folder_path = os.path.join(PSEUDO_DIR, pseudo_name)
        is_installed = "True" if os.path.exists(pseudo_folder_path) else "False"
        installed_text = Text(
            is_installed, style="green" if is_installed == "True" else "red"
        )
        # console.print(f"[cyan]DEBUG: Checking {pseudo_folder_path}: {'exists' if is_installed == 'True' else 'does not exist'}[/cyan]")
        table.add_row(
            "ONCVPSP",
            "NC",
            PSEUDOS[pseudo_name]["xc_name"],
            PSEUDOS[pseudo_name]["relativity_type"],
            "PD",
            PSEUDOS[pseudo_name]["version"],
            installed_text,
            pseudo_name,
        )

    console.print(table)
    if not installed_pseudos:
        console.print(f"[red]No pseudopotentials found in: {PSEUDO_DIR}[/red]")


@cli.command()
@click.argument("pseudo_name", required=False)
@click.option(
    "--local-only", is_flag=True, help="Use only local files, fail if not found."
)
@click.option(
    "--all", "install_all", is_flag=True, help="Install all available pseudopotentials."
)
def install(pseudo_name, local_only, install_all):
    """Install pseudopotential repositories by name(s).

    Examples:
        atomate2siesta-pseudos install ONCVPSP-PBEsol-FR-PDv0.4-Standard
        atomate2siesta-pseudos install --all
    """
    if install_all:
        # Install all available pseudopotentials
        console.print(
            f"[cyan]Installing all {len(PSEUDOS)} available pseudopotentials...[/cyan]\n"
        )

        success_count = 0
        failed_count = 0

        for name in PSEUDOS:
            console.print(f"[blue]Installing {name}...[/blue]")
            try:
                download_and_extract_pseudo(
                    pseudo_file_name=PSEUDOS[name]["filename"],
                    pseudo_name=name,
                    local_only=local_only,
                )
                success_count += 1
                console.print(f"[green]✓ Successfully installed {name}[/green]\n")
            except Exception as e:
                failed_count += 1
                console.print(f"[red]✗ Failed to install {name}: {e}[/red]\n")

        # Summary
        console.print("[bold cyan]Installation Summary:[/bold cyan]")
        console.print(f"[green]✓ Successful: {success_count}/{len(PSEUDOS)}[/green]")
        if failed_count > 0:
            console.print(f"[red]✗ Failed: {failed_count}/{len(PSEUDOS)}[/red]")

    elif pseudo_name:
        # Install specific pseudopotential
        if pseudo_name in PSEUDOS:
            download_and_extract_pseudo(
                pseudo_file_name=PSEUDOS[pseudo_name]["filename"],
                pseudo_name=pseudo_name,
                local_only=local_only,
            )
        else:
            console.print(
                f"[red]Pseudo '{pseudo_name}' not found. Use [bold]avail[/bold] command to get repo names.[/red]"
            )
    else:
        console.print(
            "[red]Error: Please specify a pseudo name or use --all flag.[/red]\n"
            "Examples:\n"
            "  atomate2siesta-pseudos install ONCVPSP-PBEsol-FR-PDv0.4-Standard\n"
            "  atomate2siesta-pseudos install --all"
        )


@cli.command()
@click.argument("pseudo_name", required=False)
@click.option("--force", is_flag=True, help="Force uninstall without confirmation.")
@click.option(
    "--all", "uninstall_all", is_flag=True, help="Uninstall all pseudopotentials."
)
def uninstall(pseudo_name, force, uninstall_all):
    """Uninstall pseudopotential repositories by name(s).

    \b
    Examples:
        atomate2siesta-pseudos uninstall ONCVPSP-PBE-SR-PDv0.4-Standard
        atomate2siesta-pseudos uninstall --all
        atomate2siesta-pseudos uninstall --all --force
    """
    if uninstall_all:
        # Get list of installed pseudos
        installed = []
        if os.path.exists(PSEUDO_DIR):
            for pseudo_name_dir in os.listdir(PSEUDO_DIR):
                pseudo_path = os.path.join(PSEUDO_DIR, pseudo_name_dir)
                if os.path.isdir(pseudo_path) and pseudo_name_dir in PSEUDOS:
                    installed.append(pseudo_name_dir)

        if not installed:
            console.print(
                f"[yellow]No pseudopotentials installed at {PSEUDO_DIR}.[/yellow]"
            )
            return

        console.print(
            f"[cyan]Found {len(installed)} installed pseudopotentials:[/cyan]"
        )
        for name in installed:
            console.print(f"  • {name}")

        if not force:
            click.confirm(
                f"\nAre you sure you want to uninstall ALL {len(installed)} pseudopotentials?",
                abort=True,
            )

        # Uninstall all
        success_count = 0
        for name in installed:
            pseudo_folder = os.path.join(PSEUDO_DIR, name)
            try:
                shutil.rmtree(pseudo_folder)
                console.print(f"[green]✓ Uninstalled {name}[/green]")
                success_count += 1
            except Exception as e:
                console.print(f"[red]✗ Failed to uninstall {name}: {e}[/red]")

        console.print(
            f"\n[bold green]Uninstalled {success_count}/{len(installed)} pseudopotentials.[/bold green]"
        )
        return

    # Single pseudo uninstall
    if not pseudo_name:
        console.print(
            "[red]Error: Please provide a pseudo name or use --all flag.[/red]"
        )
        console.print("Examples:")
        console.print(
            "  atomate2siesta-pseudos uninstall ONCVPSP-PBE-SR-PDv0.4-Standard"
        )
        console.print("  atomate2siesta-pseudos uninstall --all")
        return

    if pseudo_name not in PSEUDOS:
        console.print(
            f"[red]Pseudo '{pseudo_name}' not found. Use [bold]available[/bold] command to get repo names.[/red]"
        )
        return

    pseudo_folder = os.path.join(PSEUDO_DIR, pseudo_name)
    if not os.path.exists(pseudo_folder):
        console.print(
            f"[yellow]Pseudopotential '{pseudo_name}' is not installed at {pseudo_folder}.[/yellow]"
        )
        return

    if not force:
        click.confirm(
            f"Are you sure you want to uninstall '{pseudo_name}' from {pseudo_folder}?",
            abort=True,
        )

    try:
        shutil.rmtree(pseudo_folder)
        console.print(
            f"[green]Successfully uninstalled '{pseudo_name}' from {pseudo_folder}.[/green]"
        )
    except Exception as e:
        console.print(f"[red]Failed to uninstall '{pseudo_name}': {e}[/red]")


@cli.command()
@click.argument("pseudo_name")
@click.argument("element", required=False)
def show(pseudo_name, element):
    """Show info on pseudopotential table(s).

    If ELEMENT is provided, show detailed shell information from PSML file.

    Examples:
        atomate2siesta-pseudos show ONCVPSP-PBE-SR-PDv0.4-Standard
        atomate2siesta-pseudos show ONCVPSP-PBE-SR-PDv0.4-Standard Zr
    """
    if pseudo_name not in PSEUDOS:
        console.print(f"[red]Pseudo '{pseudo_name}' not found.[/red]")
        return

    pseudo_folder_path = os.path.join(PSEUDO_DIR, pseudo_name)
    is_installed = "True" if os.path.exists(pseudo_folder_path) else "False"

    console.print(
        f"[bold yellow]Information about:[/bold yellow] [green]{pseudo_name}[/green]\n"
    )
    console.print(
        f"[yellow]XC Functional:[/yellow] [green]{PSEUDOS[pseudo_name]['xc_name']}[/green]"
    )
    console.print(
        f"[yellow]Relativity Type:[/yellow] [green]{PSEUDOS[pseudo_name]['relativity_type']}[/green]"
    )

    if is_installed == "True":
        console.print(
            f"[yellow]The Pseudo Installed Path:[/yellow] [green]{PSEUDO_DIR}/{pseudo_name}[/green]"
        )
    console.print(f"[yellow]Installed:[/yellow] {is_installed}")

    # If element is provided, show detailed shell information
    if element:
        if is_installed != "True":
            console.print(
                f"\n[red]Cannot show shell info: Pseudo not installed. Run [bold]install {pseudo_name}[/bold] first.[/red]"
            )
            return

        psml_file = os.path.join(pseudo_folder_path, f"{element}.psml")
        if not os.path.exists(psml_file):
            console.print(
                f"\n[red]PSML file for element '{element}' not found at {psml_file}.[/red]"
            )
            return

        try:
            # Parse PSML file to extract valence configuration
            _, valence_config, _, _, element_name = parse_psml(psml_file)

            if not valence_config:
                console.print(
                    "\n[red]No valence configuration found in PSML file.[/red]"
                )
                return

            # Display shell information
            console.print(
                f"\n[bold cyan]Valence Configuration for {element_name}:[/bold cyan]"
            )

            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Shell", style="cyan")
            table.add_column("n", style="green")
            table.add_column("l", style="green")
            table.add_column("Occupation", style="yellow")

            l_names = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g"}
            for conf in valence_config:
                shell_name = f"{conf['n']}{l_names.get(conf['l'], '?')}"
                table.add_row(
                    shell_name,
                    str(conf["n"]),
                    conf["l_str"],
                    f"{conf['occupation']:.2f}",
                )

            console.print(table)

        except Exception as e:
            console.print(f"\n[red]Error reading PSML file: {e}[/red]")


@cli.command()
@click.argument("element")
def element(element):
    """Find all pseudos in the installed tables for the given element (symbol or znucl)."""
    # pseudos_found = [name for name in PSEUDOS if element.lower() in name.lower()]
    pseudos_found = [
        name for name, data in PSEUDOS.items() if element in data["elements"]
    ]

    # for pseudo in pseudos_found:
    if pseudos_found:
        console.print(
            f"[bold magenta]Pseudos found for element '{element}':[/bold magenta]"
        )
        # console.print(f"[blue]- {pseudo}[/blue]")
        console.print(f"[blue]- {pseudos_found}[/blue]")
    else:
        console.print(f"[red]No pseudos found for element '{element}'.[/red]")


# @cli.command()
# @click.argument("pseudos", nargs=-1)
# def mkff(pseudos):
#     """Compute form factors for pseudos and show them."""
#     if pseudos:
#         console.print(f"[blue]Computing form factors for: {', '.join(pseudos)}[/blue]")
#         # Call a function or external tool to process the pseudos
#     else:
#         console.print("[red]Please provide at least one pseudo name.[/red]")


@cli.command()
@click.argument("pseudo_name")
@click.argument("element")
@click.option(
    "--basis-size",
    type=click.Choice(["SZ", "DZ", "DZP", "TZ", "TZP", "TZDP"]),
    default="DZP",
    help="PAO.BasisSize to generate (default: DZP)",
)
@click.option(
    "--n-shells",
    type=int,
    default=1,
    help="Number of n-shells per l (1=valence only, 2=valence+excited, etc.)",
)
@click.option(
    "--rc-method",
    type=click.Choice(["psml", "scaled", "hydrogenic", "fixed"]),
    default="psml",
    help="Method to determine cutoff radii: psml (from wavefunction decay), scaled (by atomic radius), hydrogenic (n²/Z scaling), fixed (standard values)",
)
@click.option(
    "--rc-threshold",
    type=float,
    default=0.05,
    help="Threshold for PSML wavefunction decay (default: 0.05 = 5%%)",
)
@click.option(
    "--output-file",
    type=click.Path(),
    help="Output file for basis block (default: stdout)",
)
def basis(
    pseudo_name, element, basis_size, n_shells, rc_method, rc_threshold, output_file
):
    """Generate PAO.Basis block from PSML pseudopotential file.

    Examples:
        atomate2siesta-pseudos basis ONCVPSP-PBEsol-FR-PDv0.4-Standard Si
        atomate2siesta-pseudos basis ONCVPSP-PBE-SR-PDv0.4-Standard Fe --basis-size TZP
        atomate2siesta-pseudos basis ONCVPSP-PBE-SR-PDv0.4-Standard O --output-file O.basis
    """
    if pseudo_name not in PSEUDOS:
        console.print(
            f"[red]Pseudo '{pseudo_name}' not found. Use [bold]avail[/bold] command to get repo names.[/red]"
        )
        return

    pseudo_folder = os.path.join(PSEUDO_DIR, pseudo_name)
    if not os.path.exists(pseudo_folder):
        console.print(
            f"[red]Pseudopotential '{pseudo_name}' is not installed at {pseudo_folder}. Run [bold]install {pseudo_name}[/bold] first.[/red]"
        )
        return

    psml_file = os.path.join(pseudo_folder, f"{element}.psml")
    if not os.path.exists(psml_file):
        console.print(
            f"[red]PSML file for element '{element}' not found at {psml_file}.[/red]"
        )
        return

    try:
        # Parse PSML file to extract valence configuration and wavefunctions
        radial_grid, valence_config, wavefunctions, _, element_name = parse_psml(
            psml_file
        )

        if not valence_config:
            console.print("[red]No valence configuration found in PSML file.[/red]")
            return

        # Generate PAO.Basis block
        basis_block = generate_pao_basis_block(
            element_name,
            valence_config,
            basis_size,
            n_shells,
            rc_method=rc_method,
            rc_threshold=rc_threshold,
            radial_grid=radial_grid,
            wavefunctions=wavefunctions,
        )

        if output_file:
            with open(output_file, "w") as f:
                f.write(basis_block)
            console.print(f"[green]PAO.Basis block written to: {output_file}[/green]")
        else:
            console.print("\n[cyan]PAO.Basis block:[/cyan]\n")
            console.print(basis_block)

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise click.Abort()


def extract_rc_from_psml(radial_grid, wavefunctions, n, ang_mom, threshold=0.05):
    """Extract cutoff radius from PSML wavefunction decay.

    Args:
        radial_grid: Radial grid points (bohr)
        wavefunctions: List of wavefunction dicts with 'n', 'l', 'data' keys
        n: Principal quantum number
        ang_mom: Angular momentum (l)
        threshold: Threshold for wavefunction decay (default: 0.05 = 5%)

    Returns:
        Cutoff radius in bohr where |psi| drops below threshold*max(|psi|)
    """
    import numpy as np

    # Find matching wavefunction
    for wf in wavefunctions:
        if wf["n"] == n and wf["l"] == ang_mom:
            psi = wf["data"]
            psi_abs = np.abs(psi)
            max_psi = np.max(psi_abs)

            # Find where |psi| > threshold * max
            cutoff_idx = np.where(psi_abs > threshold * max_psi)[0]

            if len(cutoff_idx) > 0:
                rc = radial_grid[cutoff_idx[-1]]
                return rc

    # Fallback if no wavefunction found
    return None


def get_rc_scaled_by_element(element, ang_mom):
    """Scale rc by element's atomic radius (periodic trends).

    Args:
        element: Element symbol
        ang_mom: Angular momentum (l)

    Returns:
        Scaled cutoff radius in bohr
    """
    from pymatgen.core import Element

    try:
        el = Element(element)

        # Reference: Si (Z=14) with atomic radius 1.10 Angstrom
        # has rc_s=5.0, rc_p=6.0, rc_d=5.5 bohr
        ref_radius = 1.10  # Angstrom
        el_radius = el.atomic_radius if el.atomic_radius else ref_radius

        # Scale factor based on atomic size
        scale = el_radius / ref_radius

        # Base values for Si (in bohr)
        rc_ref = {
            0: 5.00,  # s orbital
            1: 6.00,  # p orbital
            2: 5.50,  # d orbital
            3: 5.00,  # f orbital
        }

        return rc_ref.get(ang_mom, 5.00) * scale

    except Exception:
        # Fallback to fixed values
        return {0: 5.00, 1: 6.00, 2: 5.50, 3: 5.00}.get(ang_mom, 5.00)


def get_rc_hydrogenic(element, n, ang_mom):
    """Calculate rc using hydrogenic (n²/Z_eff) scaling.

    Args:
        element: Element symbol
        n: Principal quantum number
        ang_mom: Angular momentum (l)

    Returns:
        Cutoff radius in bohr based on quantum numbers
    """
    from pymatgen.core import Element

    try:
        el = Element(element)
        Z = el.Z

        # Slater's effective nuclear charge approximation
        # Simple estimate: Z_eff ≈ Z - (number of inner electrons)
        if Z <= 2:
            Z_eff = Z
        elif Z <= 10:
            Z_eff = Z - 2  # Shield 1s electrons
        elif Z <= 18:
            Z_eff = Z - 10  # Shield 1s + 2s2p electrons
        else:
            Z_eff = Z - 18  # Shield up to 3s3p

        # Ensure Z_eff is reasonable
        Z_eff = max(Z_eff, 2.0)

        # Hydrogenic radius: r ~ n²/Z_eff (in atomic units)
        # Base scaling for n=3, Z_eff=6 → rc ~ 4.5 bohr
        rc_base = 4.5 * (n**2 / 9) * (6.0 / Z_eff)

        # l-dependent factor (higher l orbitals slightly more diffuse for same n)
        l_factor = {0: 1.1, 1: 1.3, 2: 1.2, 3: 1.1}

        return rc_base * l_factor.get(ang_mom, 1.0)

    except Exception:
        # Fallback to fixed values
        return {0: 5.00, 1: 6.00, 2: 5.50, 3: 5.00}.get(ang_mom, 5.00)


def generate_pao_basis_block(
    element,
    valence_config,
    basis_size,
    num_n_shells=1,
    rc_method="psml",
    rc_threshold=0.05,
    radial_grid=None,
    wavefunctions=None,
):
    """Generate PAO.Basis block for SIESTA from valence configuration.

    Args:
        element: Element symbol
        valence_config: List of dicts with 'n', 'l', 'l_str', 'occupation' keys
        basis_size: PAO.BasisSize (SZ, DZ, DZP, TZ, TZP, TZDP)
        num_n_shells: Number of n-shells per l (1=valence, 2=valence+excited, etc.)
        rc_method: Method to determine rc ('psml', 'scaled', 'hydrogenic', 'fixed')
        rc_threshold: Threshold for PSML wavefunction decay (default: 0.05)
        radial_grid: Radial grid from PSML (for psml method)
        wavefunctions: Wavefunctions from PSML (for psml method)

    Returns:
        String containing the PAO.Basis block in SIESTA format
    """
    # Determine number of zeta functions per orbital based on basis_size
    zeta_map = {
        "SZ": {"base": 1, "polarization": 0},  # Single-zeta, no polarization
        "DZ": {"base": 2, "polarization": 0},  # Double-zeta, no polarization
        "DZP": {"base": 2, "polarization": 1},  # Double-zeta + polarization
        "TZ": {"base": 3, "polarization": 0},  # Triple-zeta, no polarization
        "TZP": {"base": 3, "polarization": 1},  # Triple-zeta + polarization
        "TZDP": {"base": 3, "polarization": 2},  # Triple-zeta + double polarization
    }

    n_zeta_base = zeta_map[basis_size]["base"]
    n_zeta_pol = zeta_map[basis_size]["polarization"]

    # Group valence orbitals by l (angular momentum)
    orbitals_by_l = {}
    for conf in valence_config:
        ang_mom = conf["l"]
        if ang_mom not in orbitals_by_l:
            orbitals_by_l[ang_mom] = []
        orbitals_by_l[ang_mom].append(conf)

    # Count number of l shells (s, p, d, f, ...)
    n_shells = len(orbitals_by_l)
    if n_zeta_pol > 0:
        n_shells += 1  # Add polarization shell

    # Find maximum l for polarization
    max_l = max(conf["l"] for conf in valence_config)
    pol_l = max_l + 1  # Polarization orbital l value
    pol_l_marker = {0: "S", 1: "P", 2: "D", 3: "F", 4: "G"}.get(pol_l, "X")

    # Estimate cutoff radii (rc) - standard values from SIESTA defaults
    # These are reasonable starting points that users can adjust
    # First zeta has largest rc, subsequent zetas decrease by ~15-20%
    rc_base = {
        0: 5.00,  # s orbital base cutoff (bohr)
        1: 6.00,  # p orbital base cutoff (bohr)
        2: 5.50,  # d orbital base cutoff (bohr)
        3: 5.00,  # f orbital base cutoff (bohr)
    }

    # Scale factors for multiple zetas (descending order)
    # Standard SIESTA practice: each zeta ~0.85x of previous
    zeta_scale = 0.85

    # Build PAO.Basis block
    lines = ["%block PAO.Basis"]
    lines.append(f"{element}  {n_shells}  # Label, l_shells")

    # Add valence orbitals grouped by l
    for ang_mom in sorted(orbitals_by_l.keys()):
        # Get the highest n for this l (principal quantum number)
        max_n_for_l = max(conf["n"] for conf in orbitals_by_l[ang_mom])

        # Generate multiple n-shells if requested
        for shell_idx in range(num_n_shells):
            # n value for this shell (max_n, max_n+1, max_n+2, ...)
            n_val = max_n_for_l + shell_idx

            # Determine base cutoff radius using selected method
            rc_largest = None

            # Method 1: Extract from PSML wavefunction decay (BEST - element-specific)
            if rc_method == "psml" and radial_grid is not None and wavefunctions:
                rc_largest = extract_rc_from_psml(
                    radial_grid, wavefunctions, n_val, ang_mom, rc_threshold
                )
                if rc_largest is None and rc_method == "psml":
                    # Fallback to next best method if PSML data not available
                    rc_largest = get_rc_scaled_by_element(element, ang_mom)

            # Method 2: Scale by atomic radius (periodic trends)
            elif rc_method == "scaled":
                rc_largest = get_rc_scaled_by_element(element, ang_mom)

            # Method 3: Hydrogenic scaling (n²/Z_eff)
            elif rc_method == "hydrogenic":
                rc_largest = get_rc_hydrogenic(element, n_val, ang_mom)

            # Method 4: Fixed values (original method)
            else:  # rc_method == "fixed"
                rc_largest = rc_base.get(ang_mom, 5.00)

            # For excited shells (shell_idx > 0), use larger rc
            if shell_idx > 0:
                rc_largest *= 1.0 + (0.2 * shell_idx)

            # Generate rc values in descending order (largest to smallest)
            rc_list = []
            for i in range(n_zeta_base):
                rc_val = rc_largest * (zeta_scale**i)
                rc_list.append(f"{rc_val:.2f}")
            rc_values = " ".join(rc_list)

            # Add this orbital line with shell name (e.g., 3s, 4p, 3d)
            l_names = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g"}
            shell_name = f"{n_val}{l_names.get(ang_mom, '?')}"
            shell_type = " (excited)" if shell_idx > 0 else ""
            lines.append(
                f"  n={n_val} {ang_mom} {n_zeta_base}  # {shell_name} orbital{shell_type}"
            )
            lines.append(f"    {rc_values}  # rc(izeta=1..Nzeta) (Bohr)")

    # Add polarization orbital if requested
    if n_zeta_pol > 0:
        # Determine polarization cutoff radius using selected method
        pol_rc_base = None

        if rc_method == "psml" and radial_grid is not None and wavefunctions:
            # Try to extract from PSML (may not exist for polarization)
            pol_rc_base = extract_rc_from_psml(
                radial_grid, wavefunctions, max_n_for_l, pol_l, rc_threshold
            )
            if pol_rc_base is None:
                # Fallback: use scaled method
                pol_rc_base = get_rc_scaled_by_element(element, pol_l)
                # Polarization orbitals are typically smaller
                pol_rc_base *= 0.9

        elif rc_method == "scaled":
            pol_rc_base = get_rc_scaled_by_element(element, pol_l)
            pol_rc_base *= 0.9  # Polarization typically more compact

        elif rc_method == "hydrogenic":
            pol_rc_base = get_rc_hydrogenic(element, max_n_for_l, pol_l)
            pol_rc_base *= 0.9  # Polarization typically more compact

        else:  # fixed
            pol_rc_base = rc_base.get(pol_l, 4.50)

        # Generate descending rc values for polarization orbitals
        rc_list_pol = []
        for i in range(n_zeta_pol):
            rc_val = pol_rc_base * (zeta_scale**i)
            rc_list_pol.append(f"{rc_val:.2f}")
        rc_values_pol = " ".join(rc_list_pol)

        # Polarization orbital name
        pol_l_name = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g"}.get(pol_l, "?")
        pol_comment = f"{pol_l_name}-polarization"

        lines.append(f"  {pol_l} {n_zeta_pol} {pol_l_marker}  # {pol_comment} orbital")
        lines.append(f"    {rc_values_pol}  # rc(izeta=1..Nzeta) (Bohr)")

    lines.append("%endblock PAO.Basis")

    return "\n".join(lines)


@cli.command()
@click.argument("pseudo_name")
@click.argument("element")
@click.option(
    "--plot-type",
    type=click.Choice(
        ["wavefunctions", "potentials", "3d-potential", "occupation", "density", "all"]
    ),
    default="all",
    help="Type of plot to generate",
)
@click.option(
    "--output-dir", default=".", type=click.Path(), help="Output directory for plots"
)
@click.option(
    "--r-plot",
    type=float,
    help="Maximum radial distance (bohr) for wavefunctions, potentials, and density plots",
)
def plot(pseudo_name, element, plot_type, output_dir, r_plot):
    """Generate plots for a specified element in a pseudopotential repository."""
    if pseudo_name not in PSEUDOS:
        console.print(
            f"[red]Pseudo '{pseudo_name}' not found. Use [bold]avail[/bold] command to get repo names.[/red]"
        )
        return

    pseudo_folder = os.path.join(PSEUDO_DIR, pseudo_name)
    if not os.path.exists(pseudo_folder):
        console.print(
            f"[red]Pseudopotential '{pseudo_name}' is not installed at {pseudo_folder}. Run [bold]install {pseudo_name}[/bold] first.[/red]"
        )
        return

    psml_file = os.path.join(pseudo_folder, f"{element}.psml")
    if not os.path.exists(psml_file):
        console.print(
            f"[red]PSML file for element '{element}' not found at {psml_file}.[/red]"
        )
        return

    try:
        (
            radial_grid,
            valence_config,
            wavefunctions,
            potentials,
            element_name,
        ) = parse_psml(psml_file)

        if not radial_grid.size:
            console.print("[red]No radial grid data found in the file.[/red]")
            return
        if not potentials:
            console.print(
                "[yellow]Warning: No potentials found, some plots may be skipped.[/yellow]"
            )

        os.makedirs(output_dir, exist_ok=True)

        if plot_type in ["wavefunctions", "all"]:
            if wavefunctions:
                plot_wavefunctions(
                    radial_grid,
                    wavefunctions,
                    os.path.join(output_dir, f"{element_name}_wavefunctions.png"),
                    element_name,
                    r_max=r_plot,
                )
                console.print(
                    f"[green]Generated wavefunctions plot: {os.path.join(output_dir, f'{element_name}_wavefunctions.png')}[/green]"
                )
            else:
                console.print(
                    "[yellow]Warning: No wavefunctions (projectors) found, skipping wavefunctions plot.[/yellow]"
                )

        if plot_type in ["potentials", "all"]:
            if potentials:
                plot_potentials(
                    radial_grid,
                    potentials,
                    os.path.join(output_dir, f"{element_name}_potentials.png"),
                    element_name,
                    r_max=r_plot,
                )
                console.print(
                    f"[green]Generated potentials plot: {os.path.join(output_dir, f'{element_name}_potentials.png')}[/green]"
                )
            else:
                console.print(
                    "[yellow]Warning: No potentials found, skipping potentials plot.[/yellow]"
                )

        if plot_type in ["3d-potential", "all"]:
            if potentials:
                plot_3d_potential(
                    radial_grid,
                    potentials,
                    os.path.join(output_dir, f"{element_name}_3d_potential.png"),
                    element_name,
                    r_max=r_plot,
                )
                console.print(
                    f"[green]Generated 3D potential plot: {os.path.join(output_dir, f'{element_name}_3d_potential.png')}[/green]"
                )
            else:
                console.print(
                    "[yellow]Warning: No potentials found, skipping 3D potential plot.[/yellow]"
                )

        if plot_type in ["occupation", "all"]:
            if valence_config:
                plot_occupation_map(
                    valence_config,
                    os.path.join(output_dir, f"{element_name}_occupation_map.png"),
                    element_name,
                )
                console.print(
                    f"[green]Generated occupation map plot: {os.path.join(output_dir, f'{element_name}_occupation_map.png')}[/green]"
                )
            else:
                console.print(
                    "[yellow]Warning: No valence configuration found, skipping occupation plot.[/yellow]"
                )

        if plot_type in ["density", "all"]:
            if wavefunctions:
                plot_density(
                    radial_grid,
                    wavefunctions,
                    os.path.join(output_dir, f"{element_name}_density.png"),
                    element_name,
                    r_max=r_plot,
                )
                console.print(
                    f"[green]Generated density plot: {os.path.join(output_dir, f'{element_name}_density.png')}[/green]"
                )
            else:
                console.print(
                    "[yellow]Warning: No wavefunctions (projectors) found, skipping density plot.[/yellow]"
                )

        console.print(f"[green]Plots generated in {output_dir}[/green]")

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise click.Abort()


if __name__ == "__main__":
    cli()
