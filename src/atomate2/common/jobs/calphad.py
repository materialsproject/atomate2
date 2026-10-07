"""Jobs for CALPHAD free energy models from special quasirandom structures."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import tempfile
import warnings
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from jobflow import Flow, Response, job
from pymatgen.io.atat import Mcsqs

from atomate2 import SETTINGS
from atomate2.common.schemas.calphad import CalphadDoc, SqsCalculation

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from jobflow import Maker
    from pymatgen.core import Structure


def _copy_sqs(
    elements: Sequence[str], lattices: Sequence[str], level: int, cwd: str = "."
) -> None:
    """Copy the SQS of each lattice from the ATAT database and add bump files."""
    cmd = shlex.split(SETTINGS.SQS2TDB_CMD)
    for lattice in lattices:
        # the first call only writes species.in, the second copies the structures
        for _ in range(2):
            subprocess.run(
                [
                    *cmd,
                    "-cp",
                    f"-sp={','.join(elements)}",
                    f"-l={lattice}",
                    f"-lv={level}",
                ],
                cwd=cwd,
                check=True,
            )
        _add_bump_files(lattice, cwd)


def _add_bump_files(lattice: str, cwd: str) -> None:
    """
    Write a bump file for each SQS with a higher symmetry than in the database.

    sqs2tdb -cp does this check with the shell pipe |&, which only bash 4 and
    newer understand. Where /bin/sh is another shell, such as dash on Ubuntu or
    bash 3.2 on macOS, it writes no bump files. sqs2tdb -fit raises the energy
    of an SQS with a bump file by 5 meV/atom.
    """
    # sqs2tdb reads the lower-case variable
    atatdir = os.environ.get("atatdir") or re.sub(  # noqa: SIM112
        r".*atatdir\s*=\s*", "", (Path.home() / ".atat.rc").read_text().split("\n")[0]
    )
    database = Path(atatdir, "data", "sqsdb", lattice)
    sqs_files = {}
    for line in (database / "sqsgen.in").read_text().splitlines():
        level, *sites = line.split()
        level = level.replace("level", "lev")
        name = "_".join(["sqsdb", level, *sites])
        sqs_files[level, _get_occupations(site.split("=") for site in sites)] = (
            database / name / "bestsqs.out"
        )

    for str_in in Path(cwd, lattice).glob("sqs_lev=*/str.in"):
        folder = str_in.parent
        level, _, decoration = folder.name.removeprefix("sqs_").partition("_")
        concentrations = defaultdict(list)
        for entry in decoration.split(","):
            site, _, species = entry.partition("_")
            concentrations[site].append(species.partition("=")[2])
        occupations = (
            (site, ",".join(values)) for site, values in concentrations.items()
        )
        sqs_file = sqs_files[level, _get_occupations(occupations)]
        if _get_symmetry_count(sqs_file) < _get_symmetry_count(str_in):
            (folder / "bump").touch()


def _get_occupations(sites: Iterable[Sequence[str]]) -> frozenset:
    """Get the site occupations of an SQS from (site, "x1,x2,...") pairs."""
    return frozenset(
        (site, tuple(sorted(map(float, values.split(",")), reverse=True)))
        for site, values in sites
    )


def _get_symmetry_count(path: Path) -> int:
    """Get the number of symmetry operations that ATAT cellcvrt finds."""
    with path.open() as file:
        output = subprocess.run(
            ["cellcvrt", "-sym"],  # noqa: S607
            stdin=file,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    return int(output.split()[0])


def _get_relaxation_strain(initial: Structure, relaxed: Structure) -> float:
    """
    Get the cell distortion of a relaxation, as ATAT checkrelax reports it.

    Both cells are scaled to unit volume. The norm of the symmetric part of the
    deformation from the initial to the relaxed cell, minus the identity, is
    returned. Isotropic scaling does not count.
    """
    before, after = (
        s.lattice.matrix.T / s.volume ** (1 / 3) for s in (initial, relaxed)
    )
    deformation = np.linalg.solve(before, after)
    return float(np.linalg.norm((deformation + deformation.T) / 2 - np.eye(3)))


@job
def get_sqs_structures(
    elements: Sequence[str], lattices: Sequence[str], level: int
) -> list[dict[str, Any]]:
    """
    Get the special quasirandom structures from the ATAT database.

    Only the folders that need a calculation are returned. These hold a ``wait``
    file. Folders that link to another folder are left out. The ATAT folders
    are written to a temporary folder, which is deleted afterwards.

    Parameters
    ----------
    elements : Sequence[str]
        The two elements, e.g. ("Ni", "Re").
    lattices : Sequence[str]
        ATAT lattice names, e.g. ("FCC_A1", "HCP_A3", "LIQUID").
    level : int
        Level of the ATAT SQS database.

    Returns
    -------
    list of dict
        The lattice, folder name and structure of each SQS.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        _copy_sqs(elements, lattices, level, tmp_dir)
        return [
            {
                "lattice": lattice,
                "folder": wait.parent.name,
                "structure": Mcsqs.structure_from_str(
                    (wait.parent / "str.out").read_text()
                ),
            }
            for lattice in lattices
            for wait in sorted(Path(tmp_dir, lattice).glob("*/wait"))
        ]


@job
def run_sqs_calculations(
    sqs: list[dict[str, Any]],
    relax_maker: Maker,
    liquid_melt_maker: Maker,
    liquid_md_maker: Maker,
    n_equilibration_frames: int,
    liquid_supercell: int,
) -> Response:
    """
    Relax each solid SQS and run MD for each liquid SQS.

    Each liquid SQS is repeated liquid_supercell times along each lattice vector
    and melted with liquid_melt_maker. The melt is then run with liquid_md_maker.
    The first n_equilibration_frames frames of that run are discarded before the
    energy is averaged.

    Parameters
    ----------
    sqs : list of dict
        Output of get_sqs_structures.
    relax_maker : Maker
        Maker to relax the solid SQS.
    liquid_melt_maker : Maker
        MD maker that melts the liquid SQS.
    liquid_md_maker : Maker
        MD maker for the liquid energy. It must store the energy and structure
        of each frame.
    n_equilibration_frames : int
        Number of frames of the liquid MD that are discarded.
    liquid_supercell : int
        Number of repeats of the liquid SQS along each lattice vector.

    Returns
    -------
    Response
        Replaces this job with the calculations. The output is a list of dicts
        with the fields of SqsCalculation.
    """
    jobs = []
    outputs: list[Any] = []
    for calc in sqs:
        name = f" {calc['lattice']} {calc['folder']}"
        if calc["lattice"] == "LIQUID":
            melt = liquid_melt_maker.make(calc["structure"] * liquid_supercell)
            md = liquid_md_maker.make(melt.output.structure)
            energy = get_liquid_energy(
                md.output,
                calc["lattice"],
                calc["folder"],
                n_equilibration_frames,
                liquid_supercell**3,
            )
            for new_job in (melt, md, energy):
                new_job.append_name(name)
            jobs += [melt, md, energy]
            outputs.append(energy.output)
        else:
            relax = relax_maker.make(calc["structure"])
            relax.append_name(name)
            jobs.append(relax)
            outputs.append(
                {
                    "lattice": calc["lattice"],
                    "folder": calc["folder"],
                    "energy": relax.output.output.energy,
                    "structure": relax.output.structure,
                    "is_force_converged": relax.output.is_force_converged,
                    "dir_name": relax.output.dir_name,
                }
            )
    return Response(replace=Flow(jobs, output=outputs))


@job
def get_liquid_energy(
    md_output: Any,
    lattice: str,
    folder: str,
    n_equilibration_frames: int,
    n_cells: int,
) -> dict[str, Any]:
    """
    Get the mean potential energy and the mean squared displacement of a liquid MD.

    The energy is divided by n_cells, the number of SQS cells in the MD cell. Its
    standard error comes from the mean energies of five equal blocks of frames.
    The displacement between two frames is the minimum image of the change in
    fractional coordinates, converted with the lattice of the later frame. The
    displacement of the centre of mass is removed.

    Parameters
    ----------
    md_output : Any
        Task document of the MD, with the energy and structure of each frame.
    lattice : str
        ATAT lattice name.
    folder : str
        Name of the ATAT folder of the structure.
    n_equilibration_frames : int
        Number of frames at the start that are discarded.
    n_cells : int
        Number of SQS cells in the MD cell.

    Returns
    -------
    dict
        The fields of SqsCalculation.
    """
    steps = md_output.output.ionic_steps[n_equilibration_frames:]
    energies = np.array([step.energy for step in steps]) / n_cells
    blocks = [block.mean() for block in np.array_split(energies, 5)]
    structures = [step.structure for step in steps]
    displacement = np.zeros((len(structures[0]), 3))
    for previous, current in pairwise(structures):
        delta = current.frac_coords - previous.frac_coords
        displacement += (delta - np.round(delta)) @ current.lattice.matrix
    masses = np.array([site.specie.atomic_mass for site in structures[0]])
    displacement -= masses @ displacement / masses.sum()
    return {
        "lattice": lattice,
        "folder": folder,
        "energy": float(energies.mean()),
        "energy_standard_error": float(np.std(blocks, ddof=1) / np.sqrt(5)),
        "mean_squared_displacement": float(np.mean(np.sum(displacement**2, axis=1))),
        "dir_name": md_output.dir_name,
    }


@job(output_schema=CalphadDoc)
def fit_tdb(
    elements: Sequence[str],
    level: int,
    terms: dict[str, list[str]],
    calculations: list[dict[str, Any]],
) -> CalphadDoc:
    """
    Fit the CALPHAD models with sqs2tdb and write the TDB file.

    The ATAT folders of all lattices are copied again into the folder of this
    job, since the ordered lattices link to the end members of other lattices.
    The energy of each SQS is written to its folder. Then sqs2tdb -fit is run
    for each lattice, and sqs2tdb -tdb -oc joins the fits into one TDB file.
    A warning is given for each solid whose relaxation did not converge or has a
    relaxation_strain above 0.1.

    Parameters
    ----------
    elements : Sequence[str]
        The two elements.
    level : int
        Level of the ATAT SQS database.
    terms : dict
        Lines of the sqs2tdb terms.in file for each lattice to fit.
    calculations : list of dict
        Output of run_sqs_calculations.

    Returns
    -------
    CalphadDoc
        The energies, the fit logs and the TDB file.
    """
    lattices = list(terms)
    _copy_sqs(elements, lattices, level)
    energies = {
        (calc["lattice"], calc["folder"]): calc["energy"] for calc in calculations
    }
    for wait in Path().glob("*/*/wait"):
        key = (wait.parent.parent.name, wait.parent.name)
        if key not in energies:
            raise ValueError(f"No energy for {'/'.join(key)}.")
        (wait.parent / "energy").write_text(f"{energies[key]}\n")
    # a link points into its own lattice folder or into another lattice folder
    for link in Path().glob("*/*/link"):
        target = link.read_text().strip()
        if not (Path(link.parent.parent, target).is_dir() or Path(target).is_dir()):
            raise ValueError(f"{link.parent} links to {target}, which does not exist.")
    for calc in calculations:
        if "structure" in calc:
            initial = Mcsqs.structure_from_str(
                Path(calc["lattice"], calc["folder"], "str.out").read_text()
            )
            calc["relaxation_strain"] = _get_relaxation_strain(
                initial, calc["structure"]
            )
            if calc.get("is_force_converged") is False or (
                calc["relaxation_strain"] > 0.1
            ):
                warnings.warn(
                    f"The relaxation of {calc['lattice']}/{calc['folder']} did not "
                    "converge or has a relaxation_strain above 0.1.",
                    stacklevel=1,
                )

    cmd = shlex.split(SETTINGS.SQS2TDB_CMD)
    fit_logs = {}
    for lattice in lattices:
        Path(lattice, "terms.in").write_text("\n".join(terms[lattice]) + "\n")
        fit_logs[lattice] = subprocess.run(
            [*cmd, "-fit"], cwd=lattice, check=True, capture_output=True, text=True
        ).stdout
    subprocess.run([*cmd, "-tdb", "-oc"], check=True)
    tdb = next(Path().glob("*.tdb")).read_text()
    # the ordered lattices use the energy of each element's stable lattice, which
    # sqs2tdb leaves undefined if that lattice is not fitted
    if re.search(r"298\.15\s+;", tdb):
        raise ValueError(
            "The TDB file has an empty parameter. A Redlich-Kister level in terms "
            "needs more compositions, so use a higher SQS level."
        )
    if undefined := sorted(set(re.findall(r"ABIN_\w+", tdb))):
        raise ValueError(
            f"The TDB file uses {', '.join(undefined)}, which is not defined. Add "
            "the stable lattice of each element to the lattices."
        )

    return CalphadDoc(
        elements=list(elements),
        lattices=lattices,
        level=level,
        terms=terms,
        calculations=[SqsCalculation(**calc) for calc in calculations],
        fit_logs=fit_logs,
        tdb=tdb,
    )
