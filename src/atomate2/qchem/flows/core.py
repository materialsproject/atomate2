"""Define core QChem flows."""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from jobflow import Flow, Maker, Response, job
from pymatgen.core.structure import Molecule

from atomate2.qchem.jobs.core import FreqMaker, OptMaker

if TYPE_CHECKING:
    from pathlib import Path

    from emmet.core.qc_tasks import TaskDoc
    from jobflow import Job, OutputReference

    from atomate2.qchem.jobs.base import BaseQCMaker

logger = logging.getLogger(__name__)


@dataclass
class DoubleOptMaker(Maker):
    """
    Maker to perform a double Qchem relaxation.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    relax_maker1 : .BaseVaspMaker
        Maker to use to generate the first relaxation.
    relax_maker2 : .BaseVaspMaker
        Maker to use to generate the second relaxation.
    """

    name: str = "double opt"
    opt_maker1: BaseQCMaker | None = field(default_factory=OptMaker)
    opt_maker2: BaseQCMaker = field(default_factory=OptMaker)

    def make(self, molecule: Molecule, prev_dir: str | Path | None = None) -> Flow:
        """
        Create a flow with two chained molecular optimizations.

        Parameters
        ----------
        molecule : .Molecule
            A pymatgen Molecule object.
        prev_dir : str or Path or None
            A previous QChem calculation directory to copy output files from.

        Returns
        -------
        Flow
            A flow containing two geometric optimizations.
        """
        jobs: list[Job] = []
        if self.opt_maker1:
            # Run a pre-relaxation
            opt1 = self.opt_maker1.make(molecule, prev_dir=prev_dir)
            opt1.name += " 1"
            jobs += [opt1]
            molecule = opt1.output.optimized_molecule
            prev_dir = opt1.output.dir_name

        opt2 = self.opt_maker2.make(molecule, prev_dir=prev_dir)
        opt2.name += " 2"
        jobs += [opt2]

        return Flow(jobs, output=opt2.output, name=self.name)

    @classmethod
    def from_opt_maker(cls, opt_maker: BaseQCMaker) -> DoubleOptMaker:
        """
        Instantiate the DoubleRelaxMaker with two relax makers of the same type.

        Parameters
        ----------
        opt_maker : .BaseQCMaker
            Maker to use to generate the first and second geometric optimizations.
        """
        return cls(relax_maker1=deepcopy(opt_maker), relax_maker2=deepcopy(opt_maker))


@dataclass
class FrequencyOptMaker(Maker):
    """
    Maker to perform a frequency calculation after an optimization.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    opt_maker : .BaseQCMaker
        Maker to use to generate the opt maker
    freq_maker : .BaseQCMaker
        Maker to use to generate the freq maker
    """

    name: str = "opt frequency"
    opt_maker: BaseQCMaker = field(default_factory=OptMaker)
    freq_maker: BaseQCMaker = field(default_factory=FreqMaker)

    def make(self, molecule: Molecule, prev_dir: str | Path | None = None) -> Flow:
        """
        Create a flow with optimization followed by frequency calculation.

        Parameters
        ----------
        molecule : .Molecule
            A pymatgen Molecule object.
        prev_dir : str or Path or None
            A previous QChem calculation directory to copy output files from.

        Returns
        -------
        Flow
            A flow containing with optimization and frequency calculation.
        """
        jobs: list[Job] = []
        opt = self.opt_maker.make(molecule, prev_dir=prev_dir)
        opt.name = "Geometry Optimization"
        jobs += [opt]

        freq = self.freq_maker.make(
            molecule=opt.output.output.optimized_molecule,
            prev_dir=opt.output.dir_name,
        )
        freq.name = "Frequency Analysis"
        jobs += [freq]

        return Flow(
            jobs, output={"opt": opt.output, "freq": freq.output}, name=self.name
        )


@dataclass
class FrequencyOptFlatteningMaker(Maker):
    """
    Maker to optimize a molecule until it has no imaginary frequencies.

    Each iteration optimizes the geometry and runs a frequency analysis, until the
    molecule is a minimum or ``max_ffopt_runs`` iterations have run. The defaults
    follow the frequency flattener of custodian's
    ``QCJob.opt_with_frequency_flattener`` (used by atomate's
    ``FrequencyFlatteningOptimizeFW``):

    - a single imaginary frequency smaller than ``imag_freq_tolerance`` (15 cm^-1)
      counts as a minimum, and the iterations also stop once two consecutive
      optimizations differ in energy by less than ``energy_diff_cutoff`` (1e-6 Ha);
    - with ``linked`` (the default), each new optimization restarts from the
      previous optimized geometry and reads the exact Hessian of the previous
      frequency analysis as its initial Hessian, so that the optimizer follows the
      negative curvature downhill;
    - without ``linked``, the geometry is instead displaced along the imaginary
      mode, either by ``scale`` times the raw mode vector or, with
      ``adaptive_perturbation``, by at most ``scale`` Angstrom per atom without
      changing the bonding, and, with ``backtrack``, the opposite displacement is
      tried when the first one does not remove an imaginary frequency.

    :meth:`custodian_like` returns either custodian flattener. The behavior of
    atomate2 before these options were added (every imaginary frequency triggers
    another iteration, which displaces the geometry by the raw mode vector) is::

        FrequencyOptFlatteningMaker(
            scale=1.0,
            max_ffopt_runs=5,
            imag_freq_tolerance=0.0,
            energy_diff_cutoff=None,
            linked=False,
        )

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    opt_maker : .BaseQCMaker
        Maker to use to generate the opt maker
    freq_maker : .BaseQCMaker
        Maker to use to generate the freq maker
    scale : float
        Size of the displacement along the imaginary mode when ``linked`` is
        disabled (and of the initial displacement along a ``mode`` passed to
        :meth:`make`). Without ``adaptive_perturbation`` every atom is moved by
        ``scale`` times its component of the mode vector. With
        ``adaptive_perturbation`` it is the largest displacement of any single atom,
        in Angstrom.
    max_ffopt_runs : int
        Maximum number of optimization and frequency iterations.
    imag_freq_tolerance : float
        A single imaginary frequency smaller in magnitude than this value (in
        cm^-1) is treated as numerical noise, and the molecule as a minimum. 0
        accepts no imaginary frequencies.
    energy_diff_cutoff : float or None
        Stop iterating when the final energies of two consecutive optimizations
        differ by less than this value (in Hartree). None disables the check.
    linked : bool
        Restart each optimization from the previous optimized geometry with the
        exact Hessian of the previous frequency analysis (Q-Chem file 132.0) as its
        initial Hessian, instead of displacing the geometry. Cannot be combined
        with ``adaptive_perturbation``.
    freq_before_opt : bool
        Run a frequency analysis of the input geometry before the first
        optimization, which then starts from its Hessian. Requires ``linked``.
    adaptive_perturbation : bool
        Normalize the mode so that the most displaced atom moves by ``scale``, and if
        that displacement changes the bonding of the molecule, retry with smaller
        displacements, down to ``min_scale`` in ``scale_grid`` steps. An error is
        raised if no displacement preserves the bonding. Comparing the bonding
        requires openbabel.
    min_scale : float
        Smallest displacement (in Angstrom) tried when ``adaptive_perturbation`` is
        enabled.
    scale_grid : int
        Number of steps between ``scale`` and ``min_scale`` when
        ``adaptive_perturbation`` is enabled.
    backtrack : bool
        With ``adaptive_perturbation``: if an iteration has as many imaginary
        frequencies as the geometry it was displaced from, displace that geometry
        in the opposite direction instead; if both directions fail, continue from
        the lower-energy of the two results. An error is raised if neither result
        is an improvement.
    """

    name: str = "frequency flattening opt"
    opt_maker: BaseQCMaker = field(default_factory=OptMaker)
    freq_maker: BaseQCMaker = field(default_factory=FreqMaker)
    scale: float = 0.3
    max_ffopt_runs: int = 10
    imag_freq_tolerance: float = 15.0
    energy_diff_cutoff: float | None = 1e-6
    linked: bool = True
    freq_before_opt: bool = False
    adaptive_perturbation: bool = False
    min_scale: float = 0.1
    scale_grid: int = 10
    backtrack: bool = True

    def __post_init__(self) -> None:
        """Check that the flattening options are compatible."""
        if self.linked and self.adaptive_perturbation:
            raise ValueError(
                "linked and adaptive_perturbation are alternative ways to leave a "
                "saddle point and cannot both be enabled."
            )
        if self.freq_before_opt and not self.linked:
            raise ValueError(
                "freq_before_opt only helps when the optimization reads its Hessian; "
                "enable linked."
            )

    @classmethod
    def custodian_like(
        cls, linked: bool = True, **kwargs
    ) -> FrequencyOptFlatteningMaker:
        """
        Create a maker that reproduces custodian's frequency flattener.

        Parameters
        ----------
        linked : bool
            Reproduce custodian's linked flattener (reads the Hessian of the
            previous frequency analysis, the default in custodian and atomate) or,
            if False, its unlinked flattener (displaces the geometry without
            changing its bonding, and back-tracks).
        **kwargs
            Other keyword arguments of :obj:`FrequencyOptFlatteningMaker`, for
            example ``opt_maker`` and ``freq_maker``.

        Returns
        -------
        FrequencyOptFlatteningMaker
            The configured maker.
        """
        settings: dict[str, Any] = {
            "scale": 0.3,
            "max_ffopt_runs": 10,
            "imag_freq_tolerance": 15.0,
            "energy_diff_cutoff": 1e-6,
            "linked": linked,
            "adaptive_perturbation": not linked,
            "min_scale": 0.1,
            "scale_grid": 10,
            "backtrack": True,
        }
        return cls(**{**settings, **kwargs})

    @job
    def make(
        self,
        molecule: Molecule,
        mode: list | None = None,
        lowest_freq: float = -1.0,
        ffopt_runs: int = 0,
        overwrite_inputs: dict | None = None,
        prev_dir: str | Path | None = None,
        freq_output: TaskDoc | None = None,
        frequencies: list[float] | None = None,
        energy_history: list[float] | None = None,
        freq_dir: str | Path | None = None,
        history: list[dict] | None = None,
        parent_index: int | None = None,
    ) -> Flow:
        """
        Optimize geometry and perturb negative frequency modes.

        Parameters
        ----------
        molecule : .Molecule
            A pymatgen Molecule object.
        mode : list or None
            Normal mode along which ``molecule`` is displaced before the next
            optimization (one Cartesian vector per atom).
        lowest_freq : float
            Lowest frequency of the previous iteration; the default of -1 makes the
            first call always optimize.
        ffopt_runs : int
            Number of iterations run so far.
        overwrite_inputs : dict or None
            Overwrite the inputs of both the opt and freq makers.
        prev_dir : str or Path or None
            A previous QChem calculation directory to copy output files from.
        freq_output : .TaskDoc or None
            Output of the frequency calculation from the previous iteration. It is
            returned as the flow output once no further iterations are needed.
        frequencies : list of float or None
            All frequencies of the previous iteration, used for
            ``imag_freq_tolerance`` and ``backtrack``.
        energy_history : list of float or None
            Final energies of the optimizations run so far, used for
            ``energy_diff_cutoff`` and ``backtrack``.
        freq_dir : str or Path or None
            Directory of the previous frequency analysis, whose Hessian is read by
            the next optimization when ``linked`` is enabled.
        history : list of dict or None
            Iterations run so far, used for ``backtrack``.
        parent_index : int or None
            Index in ``history`` of the iteration that ``molecule`` was displaced
            from, used for ``backtrack``.

        Returns
        -------
        Flow
            A flow containing with optimization and frequency calculation. The
            output is the task document of the final frequency calculation.
        """
        mode = mode or [[0.0, 0.0, 0.0] for _ in range(len(molecule))]
        energy_history = list(energy_history or [])
        history = deepcopy(history or [])

        if overwrite_inputs is not None:
            self.opt_maker.input_set_generator.overwrite_inputs = overwrite_inputs
            self.freq_maker.input_set_generator.overwrite_inputs = overwrite_inputs

        new_flow = None
        new_output: OutputReference | TaskDoc | None = None

        if (
            self.freq_before_opt
            and ffopt_runs == 0
            and freq_dir is None
            and lowest_freq < 0
        ):
            # analyze the input geometry first so that the first optimization can
            # start from its Hessian
            freq = self.freq_maker.make(molecule, prev_dir=prev_dir)
            freq.name = "Frequency Analysis 0"
            recursive = self.make(
                molecule,
                ffopt_runs=0,
                prev_dir=prev_dir,
                freq_dir=freq.output.dir_name,
            )
            new_flow = Flow([freq, recursive], output=recursive.output)
            new_output = recursive.output

        elif (
            lowest_freq < 0
            and ffopt_runs < self.max_ffopt_runs
            and not self._is_converged(lowest_freq, frequencies, energy_history)
        ):
            jobs: list[Job] = []

            if self.linked and freq_dir is not None:
                opt_maker = self._linked_opt_maker()
                opt_prev_dir = freq_dir
            elif self.adaptive_perturbation and self.backtrack and ffopt_runs > 0:
                history.append(
                    {
                        "molecule": molecule,
                        "mode": mode,
                        "num_neg_freqs": sum(freq < 0 for freq in frequencies or []),
                        "energy": energy_history[-1] if energy_history else None,
                        "parent": parent_index,
                        "children": [],
                    }
                )
                parent_index, reversed_direction = plan_backtracking_perturbation(
                    history
                )
                start = history[parent_index]
                molecule = perturb_preserving_connectivity(
                    start["molecule"],
                    start["mode"],
                    self.scale,
                    self.min_scale,
                    self.scale_grid,
                    reversed_direction=reversed_direction,
                )
                opt_maker, opt_prev_dir = self.opt_maker, prev_dir
            else:
                molecule = self._perturb(molecule, mode)
                opt_maker, opt_prev_dir = self.opt_maker, prev_dir

            opt = opt_maker.make(molecule, prev_dir=opt_prev_dir)
            opt.name = f"Geometry Optimization {ffopt_runs + 1}"
            jobs += [opt]
            molecule = opt.output.output.optimized_molecule

            freq = self.freq_maker.make(molecule, prev_dir=opt.output.dir_name)
            freq.name = f"Frequency Analysis {ffopt_runs + 1}"
            jobs += [freq]

            recursive = self.make(
                molecule,
                mode=freq.output.output.frequency_modes[0],
                lowest_freq=freq.output.output.frequencies[0],
                ffopt_runs=ffopt_runs + 1,
                prev_dir=prev_dir,
                freq_output=freq.output,
                frequencies=freq.output.output.frequencies,
                energy_history=[*energy_history, opt.output.output.final_energy],
                freq_dir=freq.output.dir_name,
                history=history,
                parent_index=parent_index,
            )
            new_flow = Flow([*jobs, recursive], output=recursive.output)
            new_output = recursive.output

        elif ffopt_runs == 0:
            freq = self.freq_maker.make(molecule, prev_dir=prev_dir)
            freq.name = f"Frequency Analysis {ffopt_runs + 1}"
            new_flow = [freq]
            new_output = freq.output

        else:
            if lowest_freq < 0 and not self._is_converged(
                lowest_freq, frequencies, energy_history
            ):
                logger.warning(
                    f"Lowest frequency is still negative ({lowest_freq}) after "
                    f"{ffopt_runs} frequency flattening optimizations."
                )
            new_output = freq_output

        return Response(replace=new_flow, output=new_output)

    def _is_converged(
        self,
        lowest_freq: float,
        frequencies: list[float] | None,
        energy_history: list[float],
    ) -> bool:
        """Whether a molecule with a negative lowest frequency counts as converged."""
        if (
            frequencies
            and abs(lowest_freq) < self.imag_freq_tolerance
            and (len(frequencies) < 2 or frequencies[1] > 0)
        ):
            logger.info(
                f"Treating the single small imaginary frequency ({lowest_freq}) as "
                f"converged (tolerance {self.imag_freq_tolerance})."
            )
            return True
        if (
            self.energy_diff_cutoff is not None
            and len(energy_history) > 1
            and abs(energy_history[-1] - energy_history[-2]) < self.energy_diff_cutoff
        ):
            logger.info(
                "Energy change between the last two optimizations is below "
                f"{self.energy_diff_cutoff}; stopping frequency flattening."
            )
            return True
        return False

    def _linked_opt_maker(self) -> BaseQCMaker:
        """Copy of the opt maker that reads the Hessian of the previous freq job."""
        opt_maker = deepcopy(self.opt_maker)
        generator = opt_maker.input_set_generator
        overwrite_inputs: dict[str, Any] = deepcopy(generator.overwrite_inputs) or {}
        overwrite_inputs["rem"] = {
            **overwrite_inputs.get("rem", {}),
            "geom_opt_hessian": "read",
        }
        generator.overwrite_inputs = overwrite_inputs
        if generator.new_geom_opt:
            # libopt3 ($geom_opt section) takes the initial Hessian from here
            generator.new_geom_opt = {
                **generator.new_geom_opt,
                "initial_hessian": "read",
            }
        # custodian moves 132.0 into the Q-Chem scratch directory before running
        files = tuple(opt_maker.copy_qchem_kwargs.get("additional_qchem_files", ()))
        if "132.0" not in files:
            opt_maker.copy_qchem_kwargs = {
                **opt_maker.copy_qchem_kwargs,
                "additional_qchem_files": (*files, "132.0"),
            }
        return opt_maker

    def _perturb(self, molecule: Molecule, mode: list) -> Molecule:
        """Displace a molecule along a normal mode."""
        if not self.adaptive_perturbation:
            for idx in range(len(molecule)):
                molecule.translate_sites(
                    indices=[idx], vector=[self.scale * v for v in mode[idx]]
                )
            return molecule
        return perturb_preserving_connectivity(
            molecule, mode, self.scale, self.min_scale, self.scale_grid
        )


def perturb_preserving_connectivity(
    molecule: Molecule,
    mode: list,
    max_scale: float,
    min_scale: float = 0.1,
    scale_grid: int = 10,
    reversed_direction: bool = False,
) -> Molecule:
    """
    Displace a molecule along a normal mode without changing its bonding.

    The mode is normalized so that its most displaced atom moves by ``max_scale``
    Angstrom. If the displaced molecule has a different bonding than the original
    one, the displacement is reduced in ``scale_grid`` steps down to ``min_scale``.

    Parameters
    ----------
    molecule : .Molecule
        The molecule to displace.
    mode : list
        Normal mode vector, one Cartesian vector per atom.
    max_scale : float
        Largest displacement of any atom to try, in Angstrom.
    min_scale : float
        Smallest displacement of any atom to try, in Angstrom.
    scale_grid : int
        Number of steps between ``max_scale`` and ``min_scale``.
    reversed_direction : bool
        Displace opposite to the mode vector.

    Returns
    -------
    .Molecule
        The displaced molecule (a new object; ``molecule`` is not modified).

    Raises
    ------
    ValueError
        If every displacement changes the bonding of the molecule.
    """
    from pymatgen.io.qchem.outputs import check_for_structure_changes

    vectors = np.asarray(mode, dtype=float)
    max_norm = np.linalg.norm(vectors, axis=1).max()
    if max_norm == 0:
        return molecule.copy()
    unit_vectors = vectors / max_norm * (-1.0 if reversed_direction else 1.0)

    scales = np.linspace(max_scale, min_scale, scale_grid + 1)
    for scale in scales if max_scale > min_scale else [max_scale]:
        perturbed = Molecule(
            species=molecule.species,
            coords=molecule.cart_coords + scale * unit_vectors,
            charge=molecule.charge,
            spin_multiplicity=int(molecule.spin_multiplicity),
            site_properties=molecule.site_properties,
        )
        if check_for_structure_changes(molecule, perturbed) == "no_change":
            if scale != max_scale:
                logger.info(
                    f"Reduced the displacement along the imaginary mode to {scale:.3f}"
                    " Angstrom to keep the bonding unchanged."
                )
            return perturbed
    raise ValueError(
        "Unable to displace the molecule along its imaginary mode without changing "
        f"its bonding (tried maximum atomic displacements of {max_scale} to "
        f"{min_scale} Angstrom)."
    )


def plan_backtracking_perturbation(history: list[dict]) -> tuple[int, bool]:
    """
    Choose the geometry and direction of the next displacement along a mode.

    This follows the unlinked frequency flattener of custodian's
    ``QCJob.opt_with_frequency_flattener``. Each entry of ``history`` describes an
    iteration with the keys ``molecule`` (optimized geometry), ``mode`` (its lowest
    normal mode), ``num_neg_freqs``, ``energy``, ``parent`` (index of the iteration
    it was displaced from, or None) and ``children`` (indices of the iterations
    displaced from it). The last entry is the latest iteration.

    - If the latest iteration has fewer imaginary frequencies than its parent (or
      has no parent), it is displaced along its own mode.
    - Otherwise, if only the positive displacement of the parent has been tried, the
      parent is displaced in the opposite direction.
    - If both directions have been tried and the parent had one imaginary
      frequency, the lower-energy result is kept if it is lower in energy than the
      parent or has a clearly different mode, and is displaced along its own mode.

    ``history`` is updated in place: the chosen iteration records the next one as
    its child, and in the last case a copy of the kept result is appended.

    Parameters
    ----------
    history : list of dict
        The iterations run so far.

    Returns
    -------
    tuple of int and bool
        Index in ``history`` of the iteration to displace, and whether to displace
        it opposite to its mode.

    Raises
    ------
    ValueError
        If neither displacement direction leads to an improvement.
    """
    current = history[-1]
    start_index, reversed_direction = len(history) - 1, False
    standard = True
    parent_index = current["parent"]

    if parent_index is not None:
        parent = history[parent_index]
        if current["num_neg_freqs"] >= parent["num_neg_freqs"]:
            if len(parent["children"]) == 1:
                # only the positive displacement has been tried: try the other one
                start_index, reversed_direction = parent_index, True
                standard = False
                parent["children"].append(len(history))
            elif len(parent["children"]) == 2:
                if parent["num_neg_freqs"] != 1:
                    raise ValueError(
                        "Unable to remove more than one imaginary frequency by "
                        "displacing in either direction."
                    )
                first_child = history[parent["children"][0]]
                good_child = (
                    first_child
                    if first_child["energy"] < current["energy"]
                    else current
                )
                if good_child["num_neg_freqs"] > 1:
                    raise ValueError(
                        "The lower-energy displacement has more than one imaginary "
                        "frequency."
                    )
                mode_change = np.linalg.norm(
                    np.asarray(good_child["mode"]) - np.asarray(parent["mode"]),
                    axis=1,
                ).sum()
                if good_child["energy"] >= parent["energy"] and mode_change <= 0.2:
                    raise ValueError(
                        "Neither displacement direction lowers the energy or changes "
                        "the imaginary mode."
                    )
                history.append({**deepcopy(good_child), "children": []})
                start_index = len(history) - 1
            else:
                raise ValueError("An iteration cannot have more than two children.")

    if standard:
        history[start_index]["children"].append(len(history))
    return start_index, reversed_direction
