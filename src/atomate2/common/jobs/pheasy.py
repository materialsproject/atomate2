"""Jobs for running phonon calculations with phonopy and pheasy."""

from __future__ import annotations

import logging
import numbers
import re
import shlex
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase.io import read as ase_read
from emmet.core import __version__ as _emmet_core_version
from emmet.core.phonon import PhononBSDOSDoc
from hiphive import ClusterSpace, ForceConstantPotential, enforce_rotational_sum_rules
from hiphive import ForceConstants as HiPhiveForceConstants
from hiphive.cutoffs import estimate_maximum_cutoff
from hiphive.utilities import extract_parameters
from jobflow import job
from packaging.version import parse as parse_version
from phonopy.file_IO import parse_FORCE_CONSTANTS
from phonopy.interface.vasp import write_vasp
from phonopy.structure.symmetry import symmetrize_borns_and_epsilon
from pymatgen.core import Structure
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure
from pymatgen.io.vasp import Kpoints
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
from pymatgen.transformations.advanced_transformations import (
    CubicSupercellTransformation,
)

from atomate2.common.jobs.phonons import (
    ANGSTROM_TO_BOHR,
    _generate_phonon_object,
    _get_kpath,
    _get_num_harmonic_supercells,
    _get_num_irreducible_fcs,
    _run_band_structure_and_plot,
    _run_total_dos_and_plot,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from emmet.core.math import Matrix3D
    from phonopy.structure.atoms import PhonopyAtoms

logger = logging.getLogger(__name__)

_DEFAULT_FILE_PATHS = {
    "force_displacements": "dataset_forces.npy",
    "displacements": "dataset_disps.npy",
    "displacements_folded": "dataset_disps_array_rr.npy",
    "phonopy": "phonopy.yaml",
    "band_structure": "phonon_band_structure.yaml",
    "band_structure_plot": "phonon_band_structure.pdf",
    "dos": "phonon_dos.yaml",
    "dos_plot": "phonon_dos.pdf",
    "force_constants": "FORCE_CONSTANTS",
    "harmonic_displacements": "disp_matrix.npy",
    "anharmonic_displacements": "disp_matrix_anhar.npy",
    "harmonic_force_matrix": "force_matrix.npy",
    "anharmonic_force_matrix": "force_matrix_anhar.npy",
    "website": "phonon_website.json",
    "one_shot_dir": "one_shot",
    "anharmonic_fit_log": "pheasy_anharmonic_fit.log",
    "harmonic_fit_log": "pheasy_harmonic_fit.log",
}

# The anharmonic training set is sized so that the fit has this many force
# equations per free force constant. The floor keeps the set from becoming
# very small. Above the ceiling the job stops instead of requesting more
# displaced supercells.
_EQUATIONS_PER_FREE_FC = 100
_MIN_NUM_DISP_ANHAR = 20
_MAX_NUM_DISP_ANHAR = 600

_ANHARMONIC_FIT_METHODS = ("cocktail", "one-shot")

# many-body terms kept in the anharmonic fit for each maximum order, as passed
# to pheasy with --nbody and to ALM when counting the free force constants
_NBODY = {3: [2, 3], 4: [2, 3, 3]}


def _get_num_anharmonic_supercells(
    supercell: PhonopyAtoms,
    num_disp_anhar: int,
    anhar_max_order: int,
    fcs_cutoff_radius: Sequence[float],
    anhar_fit_methods: Sequence[str],
) -> int:
    """
    Get the number of randomly displaced supercells for the anharmonic fit.

    A non-zero num_disp_anhar is returned as given. Otherwise, each supercell
    gives 3 * natom force equations, and the number of supercells is set so
    that there are _EQUATIONS_PER_FREE_FC equations per free force constant,
    with at least _MIN_NUM_DISP_ANHAR supercells. Above _MAX_NUM_DISP_ANHAR a
    ValueError is raised. The free force constants are those of third (and
    fourth) order, plus those of second order when the one-shot fit is
    requested, since it fits all orders together. Second-order force constants
    are counted without a cutoff, as pheasy fits them. ALM counts the
    irreducible force constants before the acoustic sum rules are applied, with
    its own symmetry search, so the count is expected to be at least the number
    pheasy fits.

    Parameters
    ----------
    supercell: PhonopyAtoms
        Supercell used for the force constant fit.
    num_disp_anhar: int
        Number of displaced supercells requested by the user, 0 for automatic.
    anhar_max_order: int
        Highest force constant order in the anharmonic fit, 3 or 4.
    fcs_cutoff_radius: Sequence[float]
        Cutoff radius in Bohr for each order, starting at second order.
    anhar_fit_methods: Sequence[str]
        Anharmonic fit methods that will use this dataset.

    Returns
    -------
    int
        Number of anharmonic displaced supercells.
    """
    if num_disp_anhar != 0:
        return num_disp_anhar
    # pheasy fits the second-order force constants without a cutoff
    cutoffs = [-1, *fcs_cutoff_radius[1:]]
    n_irred = _get_num_irreducible_fcs(
        supercell, anhar_max_order, cutoffs, nbody=_NBODY[anhar_max_order]
    )
    n_free = sum(n_irred[1:])
    if "one-shot" in anhar_fit_methods:
        n_free += n_irred[0]
    natom = len(supercell.numbers)
    num = int(np.ceil(_EQUATIONS_PER_FREE_FC * n_free / (3.0 * natom)))
    num = max(num, _MIN_NUM_DISP_ANHAR)
    if num > _MAX_NUM_DISP_ANHAR:
        raise ValueError(
            f"{num} displaced supercells are needed for {n_free} free force "
            f"constants, more than the limit of {_MAX_NUM_DISP_ANHAR}. Reduce "
            "fcs_cutoff_radius, use a larger supercell, or set num_disp_anhar "
            "to accept the cost."
        )
    return num


def _check_anharmonic_settings(
    anhar_max_order: int,
    anhar_fit_methods: Sequence[str],
    cal_anhar_fcs: bool = True,
    fcs_cutoff_radius: Sequence[float] | None = None,
    anhar_alpha_min: int | None = None,
    num_disp_anhar: int = 0,
) -> None:
    """
    Check the settings of the anharmonic force constant fit.

    Parameters
    ----------
    anhar_max_order: int
        Highest force constant order in the anharmonic fit.
    anhar_fit_methods: Sequence[str]
        Anharmonic fit methods.
    cal_anhar_fcs: bool
        Whether the anharmonic force constants are calculated.
    fcs_cutoff_radius: Sequence[float] | None
        Cutoff radius in Bohr for each order, starting at second order.
    anhar_alpha_min: int | None
        Base-10 exponent of the smallest LASSO penalty. pheasy only accepts an
        integer, below its largest penalty of 1e-2.
    num_disp_anhar: int
        Number of anharmonic displaced supercells requested by the user.
    """
    if anhar_max_order not in (3, 4):
        raise ValueError(f"anhar_max_order must be 3 or 4, not {anhar_max_order}.")
    unknown = set(anhar_fit_methods) - set(_ANHARMONIC_FIT_METHODS)
    if (
        unknown
        or len(anhar_fit_methods) == 0
        or len(set(anhar_fit_methods)) != len(anhar_fit_methods)
    ):
        raise ValueError(
            f"anhar_fit_methods must be a non-empty subset of "
            f"{_ANHARMONIC_FIT_METHODS} without repeats, not {list(anhar_fit_methods)}."
        )
    if (
        isinstance(num_disp_anhar, bool)
        or not isinstance(num_disp_anhar, numbers.Integral)
        or num_disp_anhar < 0
    ):
        raise ValueError(
            f"num_disp_anhar must be a non-negative integer, not {num_disp_anhar!r}."
        )
    if cal_anhar_fcs and fcs_cutoff_radius is not None:
        # one cutoff per order from 3 to anhar_max_order, after the fc2 entry
        anhar_radii = list(fcs_cutoff_radius[1 : anhar_max_order - 1])
        # pheasy reads a negative cutoff as a neighbour shell, ALM as no cutoff
        if len(anhar_radii) != anhar_max_order - 2 or min(anhar_radii) <= 0:
            raise ValueError(
                f"fcs_cutoff_radius needs a positive cutoff in Bohr for each order "
                f"from 3 to {anhar_max_order}, not {list(fcs_cutoff_radius)}."
            )
    if anhar_alpha_min is not None and (
        isinstance(anhar_alpha_min, bool)
        or not isinstance(anhar_alpha_min, numbers.Integral)
        or anhar_alpha_min >= -2
    ):
        raise ValueError(
            f"anhar_alpha_min must be an integer below -2, not {anhar_alpha_min!r}."
        )


def _check_lasso_alpha(
    log_file: Path, alpha_min: int, alpha_min_name: str = "anhar_alpha_min"
) -> float:
    """
    Warn if the cross-validated LASSO penalty is on either bound of the search.

    On the lower bound, the search wanted an even weaker penalty than it was
    allowed to try, so the fitted force constants depend on that bound. On the
    upper bound, the penalty is the strongest one tried. Raises an error if
    pheasy wrote no log or no penalty.

    Parameters
    ----------
    log_file: Path
        Log file written by pheasy during the fit.
    alpha_min: int
        Base-10 exponent of the smallest penalty in the search.
    alpha_min_name: str
        Name of the setting that gives alpha_min, used in the warning.

    Returns
    -------
    float
        The penalty chosen by cross-validation.
    """
    if not log_file.exists():
        raise FileNotFoundError(
            f"pheasy exited without writing {log_file}, so the result of the "
            "fit is unknown."
        )
    match = re.search(r"alpha_opt:\s*([-+0-9.eE]+)", log_file.read_text())
    if match is None:
        raise RuntimeError(f"No LASSO alpha found in {log_file}, the fit failed.")
    alpha_opt = float(match.group(1))
    match_max = re.search(r"alpha_max:\s*1e([-+0-9]+)", log_file.read_text())
    if match_max is not None and alpha_opt >= 10.0 ** int(match_max.group(1)) * (
        1 - 1e-6
    ):
        warnings.warn(
            f"The LASSO penalty chosen by cross-validation ({alpha_opt:e}) is on "
            f"the upper bound in {log_file}. The fitted force constants may be "
            "heavily penalized. Check them.",
            stacklevel=2,
        )
    if alpha_opt <= 10.0**alpha_min * (1 + 1e-6):
        warnings.warn(
            f"The LASSO penalty chosen by cross-validation ({alpha_opt:e}) is on "
            f"the lower bound 1e{alpha_min} in {log_file}. The force constants "
            f"depend on this bound and may be wrong. Lower {alpha_min_name} and "
            "refit.",
            stacklevel=2,
        )
    return alpha_opt


def _run_harmonic_fit(
    supercell_matrix: np.ndarray,
    symprec: float,
    num_har: int,
    use_lasso: bool = True,
    rotational_sum_rule: str | None = "BHH",
    alpha_min: int | None = None,
    random_seed: int | None = 103,
    log_file: str | None = None,
) -> None:
    """
    Fit the second-order force constants with pheasy in the current folder.

    pheasy reads POSCAR, SPOSCAR and the harmonic displacement and force
    matrices, and writes FORCE_CONSTANTS. The caller removes the files of an
    earlier fit and checks the result, since the pheasy phonon workflow and
    the finite-temperature workflow check different things.

    Parameters
    ----------
    supercell_matrix: np.ndarray
        Diagonal supercell matrix.
    symprec: float
        Symmetry precision.
    num_har: int
        Number of displaced supercells in the fit.
    use_lasso: bool
        If True, fit with LASSO on standardized data. If False, fit with
        pheasy's default least squares.
    rotational_sum_rule: str | None
        Rotational sum rule passed to pheasy with --rasr, or None for none.
    alpha_min: int | None
        Base-10 exponent of the smallest LASSO penalty in the search. None
        keeps pheasy's default. Ignored without LASSO.
    random_seed: int | None
        Seed for the LASSO fit. Ignored without LASSO.
    log_file: str | None
        Log file of the fit. None keeps pheasy's default.
    """
    dim = " ".join(str(int(supercell_matrix[i][i])) for i in range(3))
    base = f"pheasy --dim {dim} -w 2 --symprec {float(symprec)}"
    fit = f"{base} -f --full_ifc"
    if use_lasso:
        fit += " -l LASSO --std"
        if alpha_min is not None:
            fit += f" --alpha_min {int(alpha_min)}"
        if random_seed is not None:
            fit += f" --seed {int(random_seed)}"
    if rotational_sum_rule is not None:
        fit += f" --rasr {rotational_sum_rule}"
    fit += (
        f" --ndata {int(num_har)} "
        f"--force_matrix_file {_DEFAULT_FILE_PATHS['harmonic_force_matrix']}"
    )
    if log_file is not None:
        fit += f" -o {log_file}"

    commands = [
        # clusters and orbits of the second-order force constants
        f"{base} -s --nbody 2",
        # null space
        f"{base} -c",
        # sensing matrix from the displacement matrix
        (
            f"{base} -d --ndata {int(num_har)} --disp_file "
            f"--disp_matrix_file {_DEFAULT_FILE_PATHS['harmonic_displacements']}"
        ),
        fit,
    ]
    for cmd in commands:
        subprocess.run(shlex.split(cmd), check=True)


def _run_anharmonic_fit(
    method: str,
    supercell_matrix: np.ndarray,
    symprec: float,
    anhar_max_order: int,
    fcs_cutoff_radius: Sequence[float],
    num_anhar: int,
    anhar_alpha_min: int,
    work_dir: Path,
    random_seed: int | None = 103,
) -> None:
    """
    Fit the anharmonic force constants with pheasy using LASSO.

    Both methods use the same randomly displaced supercells.

    - "cocktail" keeps the second-order force constants fixed to the harmonic
      fit (FORCE_CONSTANTS in work_dir) and fits only the higher orders.
    - "one-shot" fits the second- and higher-order force constants together.

    pheasy writes FORCE_CONSTANTS_3RD and fc3.hdf5 to work_dir, and
    FORCE_CONSTANTS_4TH and fc4.hdf5 for fourth order. The one-shot fit also
    writes its second-order force constants, fc2.hdf5.

    Parameters
    ----------
    method: str
        "cocktail" or "one-shot".
    supercell_matrix: np.ndarray
        Diagonal supercell matrix.
    symprec: float
        Symmetry precision, the same one used for the displacements.
    anhar_max_order: int
        Highest force constant order, 3 or 4.
    fcs_cutoff_radius: Sequence[float]
        Cutoff radius in Bohr for each order, starting at second order.
    num_anhar: int
        Number of anharmonic displaced supercells.
    anhar_alpha_min: int
        Base-10 exponent of the smallest LASSO penalty in the search.
    work_dir: Path
        Folder holding POSCAR and the anharmonic displacement and force
        matrices.
    random_seed: int | None
        Seed for the random coordinate descent in pheasy's LASSO fit. Without
        it, the cross-validated penalty and the fitted force constants change
        from run to run on the same forces.
    """
    dim = " ".join(str(int(supercell_matrix[i][i])) for i in range(3))
    base = f"pheasy --dim {dim} -w {anhar_max_order} --symprec {float(symprec)}"
    cutoffs = f"--c3 {float(fcs_cutoff_radius[1] / ANGSTROM_TO_BOHR)}"
    if anhar_max_order == 4:
        cutoffs += f" --c4 {float(fcs_cutoff_radius[2] / ANGSTROM_TO_BOHR)}"
    nbody = "--nbody " + " ".join(str(n) for n in _NBODY[anhar_max_order])
    fix_fc2 = "--fix_fc2 --fc2_fmt PHONOPY " if method == "cocktail" else ""
    log_file = _DEFAULT_FILE_PATHS["anharmonic_fit_log"]
    seed = f"--seed {int(random_seed)} " if random_seed is not None else ""

    commands = [
        # clusters and orbits
        f"{base} -s {nbody} {cutoffs}",
        # null space
        f"{base} -c",
        # sensing matrix from the displacement matrix
        (
            f"{base} -d --ndata {int(num_anhar)} --disp_file "
            f"--disp_matrix_file {_DEFAULT_FILE_PATHS['anharmonic_displacements']}"
        ),
        # LASSO fit. OLS, pheasy's default, gives dense force constants.
        # --std and --rasr are not passed to the anharmonic fit.
        (
            f"{base} -f {fix_fc2}-l LASSO --alpha_min {anhar_alpha_min} {seed}"
            f"--ndata {int(num_anhar)} --hdf5 "
            f"--force_matrix_file {_DEFAULT_FILE_PATHS['anharmonic_force_matrix']} "
            f"-o {log_file}"
        ),
    ]
    # pheasy appends to its log, so remove the log of an earlier fit
    (work_dir / log_file).unlink(missing_ok=True)
    for cmd in commands:
        subprocess.run(shlex.split(cmd), cwd=work_dir, check=True)

    _check_lasso_alpha(work_dir / log_file, anhar_alpha_min)


@job
def get_supercell_size(
    structure: Structure,
    min_length: float,
    max_atoms: int,
    force_90_degrees: bool,
    force_diagonal: bool,
) -> list[list[float]]:
    """
    Determine the supercell matrix with pymatgen's CubicSupercellTransformation.

    Parameters
    ----------
    structure: Structure Object
        Input structure that will be used to determine supercell
    min_length: float
        minimum length of cell in Angstrom
    max_atoms: int
        maximum number of atoms in the supercell
    force_90_degrees: bool
        if True, only supercells with three 90 degree angles are allowed
    force_diagonal: bool
        if True, only diagonal supercell matrices are allowed
    """
    transformation = CubicSupercellTransformation(
        min_length=min_length,
        max_atoms=max_atoms,
        force_90_degrees=force_90_degrees,
        force_diagonal=force_diagonal,
        angle_tolerance=1e-2,
        allow_orthorhombic=False,
    )
    transformation.apply_transformation(structure=structure)
    return transformation.transformation_matrix.transpose().tolist()


@job(data=[Structure])
def generate_phonon_displacements(
    structure: Structure,
    supercell_matrix: np.array,
    displacement: float,
    num_displaced_supercells: int,
    cal_anhar_fcs: bool,
    displacement_anhar: float,
    num_disp_anhar: int,
    fcs_cutoff_radius: list[float],
    sym_reduce: bool,
    symprec: float,
    use_symmetrized_structure: str | None,
    kpath_scheme: str,
    code: str,
    anhar_max_order: int = 3,
    anhar_fit_methods: Sequence[str] = ("cocktail",),
    random_seed: int | None = 103,
    verbose: bool = False,
) -> list[Structure]:
    """Generate the displaced supercells with phonopy.

    The small-distance set for the harmonic force constants uses the
    finite-displacement method (one displaced atom) when phonopy needs at most
    three displacements, and the random-displacement method (all atoms
    displaced) otherwise. With cal_anhar_fcs, a large-distance set of randomly
    displaced supercells is added for the anharmonic force constants. The
    undisplaced supercell is added last, for the residual forces.

    Parameters
    ----------
    structure: Structure object
        Fully optimized input structure for phonon run
    supercell_matrix: np.array
        array to describe supercell matrix
    displacement: float
        displacement in Angstrom
    num_displaced_supercells: int
        number of harmonic random displacements, 0 for automatic. Not used when
        phonopy needs at most three finite displacements.
    cal_anhar_fcs: bool
        if True, also generate the large-distance displacements used for the
        anharmonic force constants
    displacement_anhar: float
        displacement in Angstrom for the anharmonic force constants
    num_disp_anhar: int
        number of anharmonic displaced supercells, 0 to set it from the number
        of free force constants
    fcs_cutoff_radius: list[float]
        cutoff radius in Bohr for each force constant order, from second order
    sym_reduce: bool
        if True, symmetry will be used to generate displacements
    symprec: float
        precision to determine symmetry
    use_symmetrized_structure: str or None
        primitive, conventional or None
    kpath_scheme: str
        scheme to generate kpath
    code: str
        code to perform the computations
    anhar_max_order: int
        highest anharmonic force constant order, 3 or 4
    anhar_fit_methods: Sequence[str]
        anharmonic fit methods, used to size the anharmonic dataset
    random_seed : int | None = 103
        Random seed for the harmonic random displacements. The anharmonic set
        uses random_seed + 1.
    verbose : bool = False
        Whether to log warnings and the numbers of displaced supercells.

    """
    phonon = _generate_phonon_object(
        structure,
        supercell_matrix,
        displacement,
        sym_reduce,
        symprec,
        use_symmetrized_structure,
        kpath_scheme,
        code,
        verbose=verbose,
    )

    supercell_ph = phonon.supercell
    num_har = _get_num_harmonic_supercells(phonon, num_displaced_supercells)

    if verbose:
        (n_fp,) = _get_num_irreducible_fcs(supercell_ph, 2)
        logger.info(
            f"There are {n_fp} free parameters for the second-order force "
            f"constants (FCs), and {num_har} displaced supercells are used to "
            "fit them. CAUTION: you may need to increase the number "
            "of displacements in some cases. If the number of atoms in the "
            "supercell is less than 100 and all lattice constants are less than "
            "10 Å, the user is advised to use 1-2 more randomly-displaced "
            "configurations."
        )

    # if phonopy needs more than three finite displacements, we use the random
    # displacement method instead
    if len(phonon.displacements) > 3:
        phonon.generate_displacements(
            distance=displacement,
            number_of_snapshots=num_har,
            random_seed=random_seed,
        )

    supercells = phonon.supercells_with_displacements
    displacements = [get_pmg_structure(cell) for cell in supercells]

    if cal_anhar_fcs:
        _check_anharmonic_settings(
            anhar_max_order,
            anhar_fit_methods,
            fcs_cutoff_radius=fcs_cutoff_radius,
            num_disp_anhar=num_disp_anhar,
        )
        num_dis_cells_anhar = _get_num_anharmonic_supercells(
            supercell_ph,
            num_disp_anhar,
            anhar_max_order,
            fcs_cutoff_radius,
            anhar_fit_methods,
        )
        if verbose:
            logger.info(
                f"{num_dis_cells_anhar} displaced supercells are used for the "
                "anharmonic force constants."
            )
        # generate the supercells for anharmonic force constants. A different
        # seed keeps them from repeating the directions of a random harmonic set.
        phonon.generate_displacements(
            distance=displacement_anhar,
            number_of_snapshots=num_dis_cells_anhar,
            random_seed=None if random_seed is None else random_seed + 1,
        )
        supercells = phonon.supercells_with_displacements
        displacements += [get_pmg_structure(cell) for cell in supercells]

    # add the equilibrium structure to the list for calculating
    # the residual forces.
    displacements.append(get_pmg_structure(phonon.supercell))
    return displacements


@job(
    output_schema=PhononBSDOSDoc,
    data=[PhononDos, PhononBandStructureSymmLine, "force_constants"],
)
def generate_frequencies_eigenvectors(
    structure: Structure,
    supercell_matrix: np.array,
    displacement: float,
    cal_anhar_fcs: bool,
    fcs_cutoff_radius: list[float],
    sym_reduce: bool,
    symprec: float,
    use_symmetrized_structure: str | None,
    kpath_scheme: str,
    code: str,
    displacement_data: dict[str, list],
    total_dft_energy: float,
    epsilon_static: Matrix3D = None,
    born: Matrix3D = None,
    num_displaced_supercells: int = 0,
    anhar_max_order: int = 3,
    anhar_fit_methods: Sequence[str] = ("cocktail",),
    anhar_alpha_min: int = -12,
    random_seed: int | None = 103,
    **kwargs,
) -> PhononBSDOSDoc:
    """
    Fit the force constants with pheasy and summarize the phonon results.

    The harmonic force constants give the band structure, density of states
    and thermodynamic properties in the output document. With cal_anhar_fcs,
    the anharmonic force constants are also fitted and written to files in
    the job folder. They are not stored in the output document.

    Parameters
    ----------
    structure: Structure object
        Fully optimized structure used for phonon runs
    supercell_matrix: np.array
        array to describe supercell
    displacement: float
        displacement in Angstrom used for supercell computation
    cal_anhar_fcs: bool
        if True, the anharmonic force constants are fitted as well
    fcs_cutoff_radius: list[float]
        cutoff radius in Bohr for each force constant order, from second order
    sym_reduce: bool
        if True, symmetry will be used in phonopy
    symprec: float
        precision to determine symmetry
    use_symmetrized_structure: str
        primitive, conventional, None are allowed
    kpath_scheme: str
        kpath scheme for phonon band structure computation
    code: str
        code to run computations
    displacement_data: dict
        outputs from displacements
    total_dft_energy: float
        total DFT energy in eV per cell
    epsilon_static: Matrix3D
        The high-frequency dielectric constant
    born: Matrix3D
        Born charges
    num_displaced_supercells: int
        number of harmonic random displacements requested by the user,
        0 for automatic. Must match the value used to generate them.
    anhar_max_order: int
        highest anharmonic force constant order, 3 or 4
    anhar_fit_methods: Sequence[str]
        "cocktail" (fixed second-order FCs), "one-shot" (all orders fitted
        together, written to the one_shot folder), or both
    anhar_alpha_min: int
        base-10 exponent of the smallest LASSO penalty in the anharmonic fits
    random_seed : int | None = 103
        Seed for the harmonic and anharmonic LASSO fits, so that they are
        reproducible.
    kwargs: dict
        Further options read by this job, such as npoints_band, filename_bs,
        filename_dos, kpoint_density_dos and store_force_constants.
    """
    _check_anharmonic_settings(
        anhar_max_order,
        anhar_fit_methods,
        cal_anhar_fcs,
        fcs_cutoff_radius,
        anhar_alpha_min,
    )
    phonon = _generate_phonon_object(
        structure,
        supercell_matrix,
        displacement,
        sym_reduce,
        symprec,
        use_symmetrized_structure,
        kpath_scheme,
        code,
        verbose=False,
    )

    # Write the POSCAR and SPOSCAR files for the input of pheasy code
    supercell = phonon._supercell  # noqa: SLF001
    write_vasp("POSCAR", get_phonopy_structure(structure))
    write_vasp("SPOSCAR", supercell)

    # get the force-displacement dataset from previous calculations
    dataset_forces = np.array(displacement_data["forces"])
    np.save(_DEFAULT_FILE_PATHS["force_displacements"], dataset_forces)

    # To deduct the residual forces on an equilibrium structure to eliminate the
    # fitting error
    dataset_forces_array_rr = dataset_forces - dataset_forces[-1, :, :]

    # force matrix on the displaced structures
    dataset_forces_array_disp = dataset_forces_array_rr[:-1, :, :]

    # To handle the large dispalced distance in the dataset
    dataset_disps = np.array(
        [disps.frac_coords for disps in displacement_data["displaced_structures"]]
    )
    np.save(_DEFAULT_FILE_PATHS["displacements"], dataset_disps)

    supercell_scaled_positions = np.array(supercell.scaled_positions)

    dataset_disps_array_rr = np.round(
        dataset_disps - supercell_scaled_positions,
        decimals=16,
    )
    np.save(_DEFAULT_FILE_PATHS["displacements_folded"], dataset_disps_array_rr)

    dataset_disps_array_rr = np.where(
        dataset_disps_array_rr > 0.5,
        dataset_disps_array_rr - 1.0,
        dataset_disps_array_rr,
    )
    dataset_disps_array_rr = np.where(
        dataset_disps_array_rr < -0.5,
        dataset_disps_array_rr + 1.0,
        dataset_disps_array_rr,
    )

    # Transpose the displacement array on the
    # last two axes (atoms and coordinates)
    dataset_disps_array_rr_transposed = np.transpose(dataset_disps_array_rr, (0, 2, 1))

    # Perform matrix multiplication with the transposed supercell.cell
    # 'ij' for supercell.cell.T and
    # 'nkj' for the transposed dataset_disps_array_rr
    dataset_disps_array_rr_cartesian = np.einsum(
        "ij,njk->nik", supercell.cell.T, dataset_disps_array_rr_transposed
    )
    # Transpose back to the original format
    dataset_disps_array_rr_cartesian = np.transpose(
        dataset_disps_array_rr_cartesian, (0, 2, 1)
    )

    dataset_disps_array_use = dataset_disps_array_rr_cartesian[:-1, :, :]

    # separate the dataset into harmonic and anharmonic parts
    num_har = dataset_disps_array_use.shape[0]
    if cal_anhar_fcs:
        num_har = _get_num_harmonic_supercells(phonon, num_displaced_supercells)

    np.save(
        _DEFAULT_FILE_PATHS["harmonic_displacements"],
        dataset_disps_array_use[:num_har, :, :],
    )
    np.save(
        _DEFAULT_FILE_PATHS["harmonic_force_matrix"],
        dataset_forces_array_disp[:num_har, :, :],
    )

    # get the born charges and dielectric constant
    if born is not None and epsilon_static is not None:
        if len(structure) == len(born):
            borns, epsilon = symmetrize_borns_and_epsilon(
                ucell=phonon.unitcell,
                borns=np.array(born),
                epsilon=np.array(epsilon_static),
                symprec=symprec,
                primitive_matrix=phonon.primitive_matrix,
                supercell_matrix=phonon.supercell_matrix,
                is_symmetry=kwargs.get("symmetrize_born", True),
            )
        else:
            raise ValueError(
                "Number of born charges does not agree with number of atoms"
            )

        if code == "vasp" and not np.all(np.isclose(borns, 0.0)):
            phonon.nac_params = {
                "born": borns,
                "dielectric": epsilon,
                "factor": 14.399652,
            }
        # Other codes could be added here

    else:
        borns = None
        epsilon = None

    prim = ase_read("POSCAR")
    supercell = ase_read("SPOSCAR")

    # With more than 3 displacements, the random-displacement data are fitted
    # with LASSO. Otherwise, least squares is used. The rotational sum rule
    # BHH is enforced in both.
    logger.info("Start running pheasy in cluster")
    _run_harmonic_fit(
        supercell_matrix,
        symprec,
        num_har,
        use_lasso=len(phonon.displacements) > 3,
        random_seed=random_seed,
    )

    fc_file = Path(_DEFAULT_FILE_PATHS["force_constants"])
    if cal_anhar_fcs and not fc_file.exists():
        raise RuntimeError(
            "The harmonic pheasy fit did not write FORCE_CONSTANTS, so the "
            "anharmonic force constants cannot be fitted."
        )

    if cal_anhar_fcs:
        np.save(
            _DEFAULT_FILE_PATHS["anharmonic_displacements"],
            dataset_disps_array_use[num_har:, :, :],
        )
        np.save(
            _DEFAULT_FILE_PATHS["anharmonic_force_matrix"],
            dataset_forces_array_disp[num_har:, :, :],
        )
        num_anhar = dataset_forces_array_disp.shape[0] - num_har

        # The cocktail fit runs here, next to the harmonic FORCE_CONSTANTS it
        # keeps fixed. The one-shot fit runs in its own folder, because both
        # fits write FORCE_CONSTANTS_3RD and fc3.hdf5.
        for method in anhar_fit_methods:
            work_dir = Path.cwd()
            if method == "one-shot":
                work_dir = Path(_DEFAULT_FILE_PATHS["one_shot_dir"]).resolve()
                work_dir.mkdir(exist_ok=True)
                for filename in (
                    "POSCAR",
                    _DEFAULT_FILE_PATHS["anharmonic_displacements"],
                    _DEFAULT_FILE_PATHS["anharmonic_force_matrix"],
                ):
                    shutil.copy(filename, work_dir / filename)
            _run_anharmonic_fit(
                method=method,
                supercell_matrix=supercell_matrix,
                symprec=symprec,
                anhar_max_order=anhar_max_order,
                fcs_cutoff_radius=fcs_cutoff_radius,
                num_anhar=num_anhar,
                anhar_alpha_min=anhar_alpha_min,
                work_dir=work_dir,
                random_seed=random_seed,
            )

    if fc_file.exists():
        # Read the force constants from the output file of pheasy code
        force_constants = parse_FORCE_CONSTANTS(filename=fc_file)
        phonon.force_constants = force_constants
        # symmetrize the force constants to make them physically correct based on
        # the space group symmetry of the crystal structure.
        phonon.symmetrize_force_constants()

    # with phonopy.load("phonopy.yaml") the phonopy API can be used
    phonon.save(_DEFAULT_FILE_PATHS["phonopy"])

    # get phonon band structure
    kpath_dict, kpath_concrete = _get_kpath(
        structure=get_pmg_structure(phonon.primitive),
        kpath_scheme=kpath_scheme,
        symprec=symprec,
    )

    bs_plot_file = kwargs.get("filename_bs", _DEFAULT_FILE_PATHS["band_structure_plot"])
    dos_plot_file = kwargs.get("filename_dos", _DEFAULT_FILE_PATHS["dos_plot"])
    npoints_band = kwargs.get("npoints_band", 101)

    bs_symm_line, imaginary_modes = _run_band_structure_and_plot(
        phonon,
        kpath_dict,
        kpath_concrete,
        _DEFAULT_FILE_PATHS["band_structure"],
        has_nac=born is not None,
        npoints_band=npoints_band,
        with_eigenvectors=kwargs.get("band_structure_eigenvectors", False),
        is_band_connection=kwargs.get("band_structure_eigenvectors", False),
        filename_bs=bs_plot_file,
        units=kwargs.get("units", "THz"),
        tol_imaginary_modes=kwargs.get("tol_imaginary_modes", 1e-5),
    )

    # If imaginary modes are present, we first use the hiphive code to enforce
    # some symmetry constraints to eliminate the imaginary modes (generally work
    # for small imaginary modes near Gamma point). If the imaginary modes are
    # still present, a pheasy refit with a shorter cutoff (10 A) follows, but
    # its result is currently not used (see the NOTE below).

    if imaginary_modes:
        # Define a cluster space using the largest cutoff you can
        max_cutoff = estimate_maximum_cutoff(supercell) - 0.01
        cutoffs = [max_cutoff]  # only second order needed
        cs = ClusterSpace(prim, cutoffs)

        # import the phonopy force constants using the correct supercell also
        # provided by phonopy
        fcs = HiPhiveForceConstants.read_phonopy(supercell, "FORCE_CONSTANTS")

        # Find the parameters that best fits the force constants given you
        # cluster space
        parameters = extract_parameters(fcs, cs)

        # Enforce the rotational sum rules
        parameters_rot = enforce_rotational_sum_rules(
            cs, parameters, ["Huang", "Born-Huang"], alpha=1e-6
        )

        # use the new parameters to make a fcp and then create the force
        # constants and write to a phonopy file
        fcp = ForceConstantPotential(cs, parameters_rot)
        fcs = fcp.get_force_constants(supercell)
        new_fc_file = f"{_DEFAULT_FILE_PATHS['force_constants']}_short_cutoff"
        fcs.write_to_phonopy(new_fc_file, format="text")

        force_constants = parse_FORCE_CONSTANTS(filename=new_fc_file)
        phonon.force_constants = force_constants
        phonon.symmetrize_force_constants()

        bs_symm_line, imaginary_modes = _run_band_structure_and_plot(
            phonon,
            kpath_dict,
            kpath_concrete,
            _DEFAULT_FILE_PATHS["band_structure"],
            has_nac=born is not None,
            npoints_band=kwargs.get("npoints_band", 101),
            with_eigenvectors=True,
            filename_bs=bs_plot_file,
            units=kwargs.get("units", "THz"),
            tol_imaginary_modes=kwargs.get("tol_imaginary_modes", 1e-5),
        )

    # Using a shorter cutoff (10 A) to generate the force constants to
    # eliminate the imaginary modes near Gamma point in pheasy code.
    # NOTE: the force constants are read back below from
    # FORCE_CONSTANTS_short_cutoff, the file the hiPhive step above wrote, so
    # the result of this pheasy refit is not used. If the refit succeeds, it
    # overwrites FORCE_CONSTANTS in the job folder.
    if imaginary_modes:
        pheasy_cmd_11 = (
            f"pheasy --dim {int(supercell_matrix[0][0])} "
            f"{int(supercell_matrix[1][1])} "
            f"{int(supercell_matrix[2][2])} -s -w 2 --c2 "
            f"10.0 --symprec {float(symprec)} "
            f"--nbody 2"
        )

        pheasy_cmd_12 = (
            f"pheasy --dim {int(supercell_matrix[0][0])} "
            f"{int(supercell_matrix[1][1])} "
            f"{int(supercell_matrix[2][2])} -c --symprec "
            f"{float(symprec)} --c2 10.0 -w 2"
        )

        pheasy_cmd_13 = (
            f"pheasy --dim {int(supercell_matrix[0][0])} "
            f"{int(supercell_matrix[1][1])} "
            f"{int(supercell_matrix[2][2])} -w 2 -d --symprec "
            f"{float(symprec)} --c2 10.0 "
            f"--ndata {int(num_har)} --disp_file"
        )

        phonon.generate_displacements(distance=displacement)

        if len(phonon.displacements) > 3:
            pheasy_cmd_14 = (
                f"pheasy --dim {int(supercell_matrix[0][0])} "
                f"{int(supercell_matrix[1][1])} "
                f"{int(supercell_matrix[2][2])} -f --c2 10.0 "
                f"--full_ifc -w 2 --symprec {float(symprec)} "
                f"-l LASSO --std --rasr BHH --ndata {int(num_har)}"
            )

        else:
            pheasy_cmd_14 = (
                f"pheasy --dim {int(supercell_matrix[0][0])} "
                f"{int(supercell_matrix[1][1])} "
                f"{int(supercell_matrix[2][2])} -f --full_ifc "
                f"--c2 10.0 -w 2 --symprec {float(symprec)} "
                f"--rasr BHH --ndata {int(num_har)}"
            )

        subprocess.call(shlex.split(pheasy_cmd_11))
        subprocess.call(shlex.split(pheasy_cmd_12))
        subprocess.call(shlex.split(pheasy_cmd_13))
        subprocess.call(shlex.split(pheasy_cmd_14))

        force_constants = parse_FORCE_CONSTANTS(filename=new_fc_file)
        phonon.force_constants = force_constants
        phonon.symmetrize_force_constants()

        phonon.save(_DEFAULT_FILE_PATHS["phonopy"])

        # get phonon band structure
        kpath_dict, kpath_concrete = _get_kpath(
            structure=get_pmg_structure(phonon.primitive),
            kpath_scheme=kpath_scheme,
            symprec=symprec,
        )

        # phonon band structures will always be computed
        bs_symm_line, imaginary_modes = _run_band_structure_and_plot(
            phonon,
            kpath_dict,
            kpath_concrete,
            _DEFAULT_FILE_PATHS["band_structure"],
            has_nac=born is not None,
            npoints_band=kwargs.get("npoints_band", 101),
            with_eigenvectors=True,
            filename_bs=bs_plot_file,
            units=kwargs.get("units", "THz"),
            tol_imaginary_modes=kwargs.get("tol_imaginary_modes", 1e-5),
        )

    # gets data for visualization on website - yaml is also enough
    if kwargs.get("band_structure_eigenvectors"):
        bs_symm_line.write_phononwebsite(_DEFAULT_FILE_PATHS["website"])

    # get phonon density of states
    kpoint_density_dos = kwargs.get("kpoint_density_dos", 7_000)
    kpoint = Kpoints.automatic_density(
        structure=get_pmg_structure(phonon.primitive),
        kppa=kpoint_density_dos,
        force_gamma=True,
    )
    dos = _run_total_dos_and_plot(
        phonon,
        kpoint,
        _DEFAULT_FILE_PATHS["dos"],
        filename_dos=dos_plot_file,
        units=kwargs.get("units", "THz"),
    )

    # will compute thermal displacement matrices
    # for the primitive cell (phonon.primitive!)
    # only this is available in phonopy
    if kwargs.get("create_thermal_displacements"):
        phonon.run_mesh(kpoint.kpts[0], with_eigenvectors=True, is_mesh_symmetry=False)
        freq_min_thermal_displacements = kwargs.get(
            "freq_min_thermal_displacements", 0.0
        )
        phonon.run_thermal_displacement_matrices(
            t_min=kwargs.get("tmin_thermal_displacements", 0),
            t_max=kwargs.get("tmax_thermal_displacements", 500),
            t_step=kwargs.get("tstep_thermal_displacements", 100),
            freq_min=freq_min_thermal_displacements,
        )

        temperature_range_thermal_displacements = np.arange(
            kwargs.get("tmin_thermal_displacements", 0),
            kwargs.get("tmax_thermal_displacements", 500),
            kwargs.get("tstep_thermal_displacements", 100),
        )
        for idx, temp in enumerate(temperature_range_thermal_displacements):
            phonon.thermal_displacement_matrices.write_cif(
                phonon.primitive, idx, filename=f"tdispmat_{temp}K.cif"
            )
        _disp_mat = phonon._thermal_displacement_matrices  # noqa: SLF001
        tdisp_mat = _disp_mat.thermal_displacement_matrices.tolist()

        tdisp_mat_cif = _disp_mat.thermal_displacement_matrices_cif.tolist()

    else:
        tdisp_mat = None
        tdisp_mat_cif = None

    formula_units = (
        structure.composition.num_atoms
        / structure.composition.reduced_composition.num_atoms
    )

    total_dft_energy_per_formula_unit = (
        total_dft_energy / formula_units if total_dft_energy is not None else None
    )

    cls_constructor = (
        "migrate_fields"
        if parse_version(_emmet_core_version) >= parse_version("0.85.1")
        else "from_structure"
    )
    return getattr(PhononBSDOSDoc, cls_constructor)(
        structure=structure,
        meta_structure=structure,
        phonon_bandstructure=bs_symm_line,
        phonon_dos=dos,
        total_dft_energy=total_dft_energy_per_formula_unit,
        has_imaginary_modes=imaginary_modes,
        force_constants=(
            {"force_constants": phonon.force_constants.tolist()}
            if kwargs.get("store_force_constants")
            else None
        ),
        born=borns.tolist() if borns is not None else None,
        epsilon_static=epsilon.tolist() if epsilon is not None else None,
        supercell_matrix=phonon.supercell_matrix.tolist(),
        primitive_matrix=phonon.primitive_matrix.tolist(),
        code=code,
        thermal_displacement_data={
            "temperatures_thermal_displacements": temperature_range_thermal_displacements.tolist(),  # noqa: E501
            "thermal_displacement_matrix_cif": tdisp_mat_cif,
            "thermal_displacement_matrix": tdisp_mat,
            "freq_min_thermal_displacements": freq_min_thermal_displacements,
        }
        if kwargs.get("create_thermal_displacements")
        else None,
        jobdirs={
            "displacements_job_dirs": displacement_data["dirs"],
            "static_run_job_dir": kwargs["static_run_job_dir"],
            "born_run_job_dir": kwargs["born_run_job_dir"],
            "optimization_run_job_dir": kwargs["optimization_run_job_dir"],
            "taskdoc_run_job_dir": str(Path.cwd()),
        },
        uuids={
            "displacements_uuids": displacement_data["uuids"],
            "born_run_uuid": kwargs["born_run_uuid"],
            "optimization_run_uuid": kwargs["optimization_run_uuid"],
            "static_run_uuid": kwargs["static_run_uuid"],
        },
        post_process_settings={
            "npoints_band": npoints_band,
            "kpath_scheme": kpath_scheme,
            "kpoint_density_dos": kpoint_density_dos,
        },
    )
