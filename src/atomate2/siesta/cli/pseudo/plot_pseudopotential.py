#!/usr/bin/env python
import click
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xml.etree import ElementTree as ET
from scipy.interpolate import interp1d
import os


def parse_psml(file_path):
    """Parse the XML-based PSML file with namespace handling."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        ns = {"psml": "http://esl.cecam.org/PSML/ns/1.1"}
        click.echo("Successfully parsed XML-based PSML file")
        click.echo(f"Root element: {root.tag}")
        click.echo(f"Root attributes: {root.attrib}")

        # Extract element name
        pseudo_atom = root.find(".//psml:pseudo-atom-spec", namespaces=ns)
        element_name = (
            pseudo_atom.get("atomic-label", "Unknown")
            if pseudo_atom is not None
            else "Unknown"
        )
        click.echo(f"Extracted element name: {element_name}")

        # Extract radial grid from <grid><grid-data>
        grid = root.find(".//psml:grid", namespaces=ns)
        if grid is None:
            click.echo("No <grid> element found in PSML file")
            raise ValueError("No <grid> element found in PSML file")
        grid_data = grid.find("psml:grid-data", namespaces=ns)
        if grid_data is None:
            click.echo("No <grid-data> element found in <grid>")
            raise ValueError("No <grid-data> element found in <grid>")
        radial_grid = np.array([float(x) for x in grid_data.text.split() if x.strip()])
        click.echo(f"Extracted radial grid with {len(radial_grid)} points")

        # Extract valence configuration
        valence_config = []
        valence = root.find(".//psml:valence-configuration", namespaces=ns)
        if valence is None:
            click.echo("Warning: No <valence-configuration> element found")
        else:
            for shell in valence.findall("psml:shell", namespaces=ns):
                try:
                    n = int(shell.get("n"))
                    l_str = shell.get("l")
                    l = {"s": 0, "p": 1, "d": 2, "f": 3}.get(l_str, -1)
                    if l == -1:
                        raise ValueError(f"Invalid l value: {l_str}")
                    occ = float(shell.get("occupation"))
                    valence_config.append(
                        {"n": n, "l": l, "occupation": occ, "l_str": l_str}
                    )
                except (ValueError, TypeError) as e:
                    click.echo(f"Skipping invalid valence config: {shell.attrib} ({e})")
            click.echo(f"Extracted {len(valence_config)} valence configurations")

        # Extract wavefunctions (using nonlocal projectors as proxies)
        wavefunctions = []
        projectors = root.find(".//psml:nonlocal-projectors", namespaces=ns)
        if projectors is None:
            click.echo("Warning: No <nonlocal-projectors> element found")
        else:
            # Map projectors to n values based on l and sequence
            l_to_n = {0: [], 1: [], 2: [], 3: []}
            for conf in valence_config:
                l_to_n[conf["l"]].append(conf["n"])
            for proj in projectors.findall("psml:proj", namespaces=ns):
                try:
                    l_str = proj.get("l")
                    l = {"s": 0, "p": 1, "d": 2, "f": 3}.get(l_str, -1)
                    if l == -1:
                        raise ValueError(f"Invalid l value: {l_str}")
                    seq = int(proj.get("seq", 1)) - 1  # seq starts at 1
                    n = (
                        l_to_n[l][seq]
                        if seq < len(l_to_n[l])
                        else l_to_n[l][-1]
                        if l_to_n[l]
                        else 3
                    )
                    radfunc = proj.find("psml:radfunc/psml:data", namespaces=ns)
                    if radfunc is None:
                        click.echo(
                            f"Skipping projector l={l_str} due to missing <radfunc><data>"
                        )
                        continue
                    data = np.array(
                        [float(x) for x in radfunc.text.split() if x.strip()]
                    )
                    if len(data) != len(radial_grid):
                        x_old = np.linspace(0, len(data) - 1, len(data))
                        x_new = np.linspace(0, len(data) - 1, len(radial_grid))
                        interpolator = interp1d(
                            x_old, data, kind="cubic", fill_value="extrapolate"
                        )
                        data = interpolator(x_new)
                        click.echo(
                            f"Interpolated projector l={l_str} from {len(data)} to {len(radial_grid)} points"
                        )
                    wavefunctions.append({"n": n, "l": l, "l_str": l_str, "data": data})
                except (ValueError, TypeError, IndexError) as e:
                    click.echo(f"Skipping invalid projector: {proj.attrib} ({e})")
            click.echo(f"Extracted {len(wavefunctions)} wavefunctions (projectors)")

        # Extract potentials
        potentials = []
        local_pot = root.find(".//psml:local-potential", namespaces=ns)
        if local_pot is not None:
            radfunc = local_pot.find("psml:radfunc/psml:data", namespaces=ns)
            if radfunc is not None:
                try:
                    data = np.array(
                        [float(x) for x in radfunc.text.split() if x.strip()]
                    )
                    if len(data) != len(radial_grid):
                        x_old = np.linspace(0, len(data) - 1, len(data))
                        x_new = np.linspace(0, len(data) - 1, len(radial_grid))
                        interpolator = interp1d(
                            x_old, data, kind="cubic", fill_value="extrapolate"
                        )
                        data = interpolator(x_new)
                        click.echo(
                            f"Interpolated local potential from {len(data)} to {len(radial_grid)} points"
                        )
                    potentials.append({"l": None, "data": data})
                except ValueError as e:
                    click.echo(f"Skipping invalid local potential: {e}")
        semilocal = root.find(".//psml:semilocal-potentials", namespaces=ns)
        if semilocal is not None:
            for slps in semilocal.findall("psml:slps", namespaces=ns):
                try:
                    l_str = slps.get("l")
                    l = {"s": 0, "p": 1, "d": 2, "f": 3}.get(l_str, -1)
                    if l == -1:
                        raise ValueError(f"Invalid l value: {l_str}")
                    n = slps.get("n")
                    if n is not None:
                        n = int(n)
                    else:
                        n = next(
                            (conf["n"] for conf in valence_config if conf["l"] == l), 3
                        )
                    radfunc = slps.find("psml:radfunc/psml:data", namespaces=ns)
                    if radfunc is None:
                        click.echo(
                            f"Skipping semilocal potential l={l_str} due to missing <radfunc><data>"
                        )
                        continue
                    data = np.array(
                        [float(x) for x in radfunc.text.split() if x.strip()]
                    )
                    if len(data) != len(radial_grid):
                        x_old = np.linspace(0, len(data) - 1, len(data))
                        x_new = np.linspace(0, len(data) - 1, len(radial_grid))
                        interpolator = interp1d(
                            x_old, data, kind="cubic", fill_value="extrapolate"
                        )
                        data = interpolator(x_new)
                        click.echo(
                            f"Interpolated semilocal potential l={l_str} from {len(data)} to {len(radial_grid)} points"
                        )
                    potentials.append({"n": n, "l": l, "l_str": l_str, "data": data})
                except (ValueError, TypeError) as e:
                    click.echo(
                        f"Skipping invalid semilocal potential: {slps.attrib} ({e})"
                    )
        click.echo(f"Extracted {len(potentials)} potentials")

        return radial_grid, valence_config, wavefunctions, potentials, element_name

    except ET.ParseError as e:
        raise ValueError(f"Failed to parse PSML file as XML: {e}")
    except Exception as e:
        raise ValueError(f"Error processing PSML file: {e}")


def plot_wavefunctions(
    radial_grid, wavefunctions, output_file, element_name, r_max=None
):
    """Plot radial wavefunctions (projectors) for different n, l values."""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    colors = sns.color_palette("husl", len(wavefunctions))
    for wf, color in zip(wavefunctions, colors):
        label = f"{wf['n']}{['s', 'p', 'd', 'f'][wf['l']]}"
        plt.plot(radial_grid, wf["data"], label=label, linewidth=2, color=color)

    plt.title(
        f"{element_name} Radial Projector Functions (Wavefunction Proxy)", fontsize=14
    )
    plt.xlabel("r (bohr)", fontsize=12)
    plt.ylabel("Projector Value", fontsize=12)
    if r_max is not None:
        plt.xlim(0, r_max)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_potentials(radial_grid, potentials, output_file, element_name, r_max=None):
    """Plot local and semilocal potentials in a polar plot."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="polar")
    colors = sns.color_palette("husl", len(potentials))

    for pot, color in zip(potentials, colors):
        label = (
            f"{pot['n']}{['s', 'p', 'd', 'f'][pot['l']]}"
            if pot["l"] is not None
            else "Local Potential"
        )
        theta = np.linspace(0, 2 * np.pi, 100)
        r = np.abs(pot["data"])
        r_plot = np.interp(
            np.linspace(0, len(radial_grid), len(theta)), np.arange(len(radial_grid)), r
        )
        ax.plot(theta, r_plot, label=label, linewidth=2, color=color)

    ax.set_title(f"{element_name} Pseudopotentials (Polar Representation)", fontsize=14)
    ax.set_xlabel("Radial Distance (bohr)", fontsize=12)
    if r_max is not None:
        ax.set_rlim(0, r_max)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_3d_potential(radial_grid, potentials, output_file, element_name, r_max=None):
    """Create a 3D surface plot of the local potential."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    theta = np.linspace(0, 2 * np.pi, 100)
    r_grid = radial_grid
    if r_max is not None:
        mask = radial_grid <= r_max
        r_grid = radial_grid[mask]
    r, theta = np.meshgrid(r_grid, theta)

    local_potential = next(
        (pot["data"] for pot in potentials if pot["l"] is None), potentials[0]["data"]
    )
    if r_max is not None:
        local_potential = local_potential[mask]
    z = np.tile(local_potential, (len(theta), 1))

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    surf = ax.plot_surface(x, y, z, cmap="viridis", edgecolor="none")
    fig.colorbar(surf, ax=ax, label="Potential (Hartree)")
    ax.set_title(f"{element_name} 3D Local Pseudopotential Surface", fontsize=14)
    ax.set_xlabel("x (bohr)", fontsize=12)
    ax.set_ylabel("y (bohr)", fontsize=12)
    ax.set_zlabel("Potential (Hartree)", fontsize=12)
    if r_max is not None:
        ax.set_xlim(-r_max, r_max)
        ax.set_ylim(-r_max, r_max)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_occupation_map(valence_config, output_file, element_name):
    """Create a heatmap of valence electron occupations."""
    n_values = sorted(set(conf["n"] for conf in valence_config))
    l_values = sorted(set(conf["l"] for conf in valence_config))

    occ_matrix = np.zeros((len(n_values), len(l_values)))
    for conf in valence_config:
        n_idx = n_values.index(conf["n"])
        l_idx = l_values.index(conf["l"])
        occ_matrix[n_idx, l_idx] = conf["occupation"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        occ_matrix,
        xticklabels=["s", "p", "d", "f"][: len(l_values)],
        yticklabels=n_values,
        annot=True,
        cmap="Blues",
    )
    plt.title(f"{element_name} Valence Electron Occupation", fontsize=14)
    plt.xlabel("Angular Momentum (l)", fontsize=12)
    plt.ylabel("Principal Quantum Number (n)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_density(radial_grid, wavefunctions, output_file, element_name, r_max=None):
    """Create a density plot of wavefunction (projector) magnitudes."""
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("husl", len(wavefunctions))
    for wf, color in zip(wavefunctions, colors):
        density = wf["data"] ** 2
        label = f"{wf['n']}{['s', 'p', 'd', 'f'][wf['l']]} Density"
        plt.fill_between(radial_grid, density, alpha=0.5, label=label, color=color)

    plt.title(
        f"{element_name} Radial Density (Projector Magnitude Squared)", fontsize=14
    )
    plt.xlabel("r (bohr)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    if r_max is not None:
        plt.xlim(0, r_max)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


@click.command()
@click.argument("psml_file", type=click.Path(exists=True))
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
def plot_pseudopotential(psml_file, plot_type, output_dir, r_plot):
    """Generate plots from an XML-based PSML pseudopotential file."""
    try:
        (
            radial_grid,
            valence_config,
            wavefunctions,
            potentials,
            element_name,
        ) = parse_psml(psml_file)

        if not radial_grid.size:
            raise ValueError("No radial grid data found in the file")
        if not potentials:
            click.echo("Warning: No potentials found, some plots may be skipped")

        if plot_type in ["wavefunctions", "all"]:
            if wavefunctions:
                plot_wavefunctions(
                    radial_grid,
                    wavefunctions,
                    os.path.join(output_dir, f"{element_name}_wavefunctions.png"),
                    element_name,
                    r_max=r_plot,
                )
            else:
                click.echo(
                    "Warning: No wavefunctions (projectors) found, skipping wavefunctions plot"
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
            else:
                click.echo("Warning: No potentials found, skipping potentials plot")
        if plot_type in ["3d-potential", "all"]:
            if potentials:
                plot_3d_potential(
                    radial_grid,
                    potentials,
                    os.path.join(output_dir, f"{element_name}_3d_potential.png"),
                    element_name,
                    r_max=r_plot,
                )
            else:
                click.echo("Warning: No potentials found, skipping 3D potential plot")
        if plot_type in ["occupation", "all"]:
            if valence_config:
                plot_occupation_map(
                    valence_config,
                    os.path.join(output_dir, f"{element_name}_occupation_map.png"),
                    element_name,
                )
            else:
                click.echo(
                    "Warning: No valence configuration found, skipping occupation plot"
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
            else:
                click.echo(
                    "Warning: No wavefunctions (projectors) found, skipping density plot"
                )

        click.echo(f"Plots generated in {output_dir}")

    except ValueError as e:
        click.echo(f"Error: {e}")
        raise click.Abort()
    except Exception as e:
        click.echo(f"Unexpected error: {e}")
        raise click.Abort()


if __name__ == "__main__":
    plot_pseudopotential()
