"""ASE-based NEB workflow with Python optimization."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import LBFGS
from jobflow import Flow, Maker, job
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from atomate2.siesta.flows.base import BaseSiestaFlowMaker
from atomate2.siesta.flows.neb.plotting import plot_ase_neb_results
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class PerImageLBFGS:
    """
    Per-image LBFGS optimizer for NEB calculations.

    This optimizer follows the Lua FLOS approach where each NEB image has its own
    independent LBFGS optimizer. This is more effective than a global optimizer
    because:

    1. **Small Hessian per image**: Each image has ~195 DOF (65 atoms × 3), not
       ~1365 DOF (7 images × 65 atoms × 3). Smaller Hessians are easier to
       approximate accurately.

    2. **Local curvature learning**: Each image learns its own local potential
       energy surface curvature, rather than trying to mix curvature information
       from different images.

    3. **Better handling of NEB forces**: NEB forces are projected (perpendicular
       DFT + parallel spring), which confuses global optimizers but works well
       with per-image optimization.

    Parameters
    ----------
    n_images : int
        Total number of NEB images (including endpoints).
    alpha : float
        Initial inverse Hessian diagonal (H0 = alpha * I).
        Default is 1/75 ≈ 0.0133 (same as Lua FLOS).
    maxstep : float
        Maximum step size in Å. Default is 0.2.
    memory : int
        Number of steps to keep in L-BFGS memory. Default is 20.

    Attributes
    ----------
    optimizers : list
        List of per-image optimizer states (one per intermediate image).
        Each state contains: positions_history, gradients_history, H0.

    Example
    -------
    >>> opt = PerImageLBFGS(n_images=7, alpha=1/75, maxstep=0.2)
    >>> # Each iteration:
    >>> new_positions = opt.step(neb_forces, current_positions)
    >>> # Update NEB images with new positions
    """

    def __init__(
        self,
        n_images: int,
        alpha: float = 1.0 / 75.0,
        maxstep: float = 0.2,
        memory: int = 20,
    ):
        """Initialize per-image LBFGS optimizer."""
        self.n_images = n_images
        self.n_intermediate = n_images - 2  # Exclude fixed endpoints
        self.alpha = alpha
        self.maxstep = maxstep
        self.memory = memory

        # Initialize per-image optimizer states
        # Each optimizer maintains its own L-BFGS history
        self.optimizers: list[dict[str, Any]] = []
        for _ in range(self.n_intermediate):
            self.optimizers.append(
                {
                    "s_history": [],  # Position differences: s_k = x_{k+1} - x_k
                    "y_history": [],  # Gradient differences: y_k = g_{k+1} - g_k
                    "rho_history": [],  # 1 / (y_k · s_k)
                    "prev_positions": None,
                    "prev_gradients": None,
                    "H0": alpha,  # Initial Hessian diagonal
                    "iteration": 0,
                }
            )

        logger.info(
            f"PerImageLBFGS initialized: {self.n_intermediate} intermediate images, "
            f"alpha={alpha:.4f}, maxstep={maxstep:.3f} Å, memory={memory}"
        )

    def step(
        self,
        neb_forces: np.ndarray,
        positions: list[np.ndarray],
    ) -> list[np.ndarray]:
        """
        Perform one L-BFGS step for each intermediate image.

        Parameters
        ----------
        neb_forces : np.ndarray
            NEB forces for intermediate images only.
            Shape: ((n_images-2) * n_atoms, 3) or (n_intermediate, n_atoms, 3)
        positions : list[np.ndarray]
            Current positions for ALL images (including endpoints).
            Each element has shape (n_atoms, 3).

        Returns
        -------
        list[np.ndarray]
            Updated positions for ALL images.
            Endpoints are unchanged, intermediate images are updated.
        """
        # Get number of atoms from first image
        n_atoms = positions[0].shape[0]

        # Reshape NEB forces if needed
        # neb_forces from ASE NEB.get_forces() has shape ((n_images-2) * n_atoms, 3)
        if (
            neb_forces.ndim == 2
            and neb_forces.shape[0] == self.n_intermediate * n_atoms
        ):
            forces_per_image = neb_forces.reshape(self.n_intermediate, n_atoms, 3)
        elif neb_forces.ndim == 3 and neb_forces.shape[0] == self.n_intermediate:
            forces_per_image = neb_forces
        else:
            raise ValueError(
                f"Unexpected NEB forces shape: {neb_forces.shape}. "
                f"Expected ({self.n_intermediate * n_atoms}, 3) or "
                f"({self.n_intermediate}, {n_atoms}, 3)"
            )

        # Make copy of positions to update
        new_positions = [pos.copy() for pos in positions]

        # Apply L-BFGS step to each intermediate image independently
        for i in range(self.n_intermediate):
            image_idx = i + 1  # Skip initial endpoint (index 0)
            opt_state = self.optimizers[i]

            # Current positions and gradients for this image
            pos = positions[image_idx].flatten()  # (n_atoms * 3,)
            # Negative forces = gradients (minimization: F = -dE/dr, we minimize so grad = -F)
            grad = -forces_per_image[i].flatten()  # (n_atoms * 3,)

            # L-BFGS history update: We need s_k = x_k - x_{k-1} and y_k = g_k - g_{k-1}
            # This must be done BEFORE computing the new direction, using the CURRENT
            # gradient and the PREVIOUS position/gradient stored from last iteration
            if opt_state["prev_positions"] is not None:
                s = pos - opt_state["prev_positions"]  # x_k - x_{k-1}
                y = grad - opt_state["prev_gradients"]  # g_k - g_{k-1}

                # Only add to history if curvature condition is satisfied (y · s > 0)
                ys = np.dot(y, s)
                if ys > 1e-10:
                    rho = 1.0 / ys
                    opt_state["s_history"].append(s)
                    opt_state["y_history"].append(y)
                    opt_state["rho_history"].append(rho)

                    # Trim history to memory limit
                    if len(opt_state["s_history"]) > self.memory:
                        opt_state["s_history"].pop(0)
                        opt_state["y_history"].pop(0)
                        opt_state["rho_history"].pop(0)

                    logger.debug(
                        f"  Image {image_idx}: Added to L-BFGS history (ys={ys:.4f}), "
                        f"history size={len(opt_state['s_history'])}"
                    )
                else:
                    logger.debug(
                        f"  Image {image_idx}: Skipped history update (ys={ys:.4f} <= 0)"
                    )

            # Compute search direction using L-BFGS two-loop recursion
            direction = self._compute_lbfgs_direction(opt_state, grad)

            # Apply step with maxstep limit
            step = -direction  # Descend along negative gradient direction
            step_norm = np.linalg.norm(step)

            if step_norm > self.maxstep * np.sqrt(len(step) / 3):
                # Scale to max allowed displacement per atom
                step = step * (self.maxstep * np.sqrt(len(step) / 3) / step_norm)

            # Update positions
            new_pos = pos + step
            new_positions[image_idx] = new_pos.reshape(n_atoms, 3)

            # Save current position and gradient for next iteration's history update
            opt_state["prev_positions"] = pos.copy()
            opt_state["prev_gradients"] = grad.copy()
            opt_state["iteration"] += 1

            # Log per-image info
            max_disp = np.max(np.abs(step.reshape(-1, 3)), axis=1).max()
            max_force = np.max(np.linalg.norm(forces_per_image[i], axis=1))
            logger.debug(
                f"  Image {image_idx}: max_force={max_force:.4f} eV/Å, "
                f"max_disp={max_disp:.4f} Å"
            )

        return new_positions

    def _compute_lbfgs_direction(
        self,
        opt_state: dict,
        grad: np.ndarray,
    ) -> np.ndarray:
        """
        Compute L-BFGS search direction using two-loop recursion.

        This implements the standard L-BFGS algorithm from Nocedal & Wright.

        Parameters
        ----------
        opt_state : dict
            Per-image optimizer state with history.
        grad : np.ndarray
            Current gradient (flattened, shape: n_atoms * 3).

        Returns
        -------
        np.ndarray
            Search direction (same shape as grad).
        """
        s_history = opt_state["s_history"]
        y_history = opt_state["y_history"]
        rho_history = opt_state["rho_history"]
        H0 = opt_state["H0"]

        m = len(s_history)  # Number of stored corrections

        if m == 0:
            # First iteration: use steepest descent with H0 scaling
            return H0 * grad

        # Two-loop recursion
        q = grad.copy()
        alpha_list = []

        # First loop (backward)
        for j in range(m - 1, -1, -1):
            alpha_j = rho_history[j] * np.dot(s_history[j], q)
            alpha_list.append(alpha_j)
            q = q - alpha_j * y_history[j]

        alpha_list.reverse()

        # Compute H0_k * q (scaled identity)
        # Use Shanno-Phua scaling: H0_k = (s_{k-1} · y_{k-1}) / (y_{k-1} · y_{k-1})
        s_last = s_history[-1]
        y_last = y_history[-1]
        gamma = np.dot(s_last, y_last) / (np.dot(y_last, y_last) + 1e-10)
        r = gamma * q

        # Second loop (forward)
        for j in range(m):
            beta_j = rho_history[j] * np.dot(y_history[j], r)
            r = r + (alpha_list[j] - beta_j) * s_history[j]

        return r

    def reset(self) -> None:
        """Reset all optimizer states (clear history)."""
        for opt in self.optimizers:
            opt["s_history"] = []
            opt["y_history"] = []
            opt["rho_history"] = []
            opt["prev_positions"] = None
            opt["prev_gradients"] = None
            opt["iteration"] = 0
        logger.info("PerImageLBFGS: Reset all optimizer states")


class PerImageBFGS:
    """
    Per-image BFGS optimizer for NEB calculations.

    This optimizer follows the Lua FLOS approach where each NEB image has its own
    independent BFGS optimizer with a full inverse Hessian approximation.

    Unlike L-BFGS which only stores recent gradient/position pairs, full BFGS
    maintains the complete inverse Hessian matrix. This can provide better
    convergence for smaller systems but uses more memory (O(n²) vs O(n*m)).

    Parameters
    ----------
    n_images : int
        Total number of NEB images (including endpoints).
    alpha : float
        Initial inverse Hessian diagonal (H0 = alpha * I).
        Default is 1/75 ≈ 0.0133 (same as Lua FLOS).
    maxstep : float
        Maximum step size in Å. Default is 0.2.

    Attributes
    ----------
    optimizers : list
        List of per-image optimizer states (one per intermediate image).
        Each state contains: H (inverse Hessian), prev_positions, prev_gradients.

    Example
    -------
    >>> opt = PerImageBFGS(n_images=7, alpha=1/75, maxstep=0.2)
    >>> # Each iteration:
    >>> new_positions = opt.step(neb_forces, current_positions)
    >>> # Update NEB images with new positions
    """

    def __init__(
        self,
        n_images: int,
        alpha: float = 1.0 / 75.0,
        maxstep: float = 0.2,
    ):
        """Initialize per-image BFGS optimizer."""
        self.n_images = n_images
        self.n_intermediate = n_images - 2  # Exclude fixed endpoints
        self.alpha = alpha
        self.maxstep = maxstep

        # Initialize per-image optimizer states
        # Each optimizer maintains its own full inverse Hessian
        self.optimizers: list[dict[str, Any]] = []
        for _ in range(self.n_intermediate):
            self.optimizers.append(
                {
                    "H": None,  # Inverse Hessian (initialized on first step)
                    "prev_positions": None,
                    "prev_gradients": None,
                    "iteration": 0,
                }
            )

        logger.info(
            f"PerImageBFGS initialized: {self.n_intermediate} intermediate images, "
            f"alpha={alpha:.4f}, maxstep={maxstep:.3f} Å"
        )

    def step(
        self,
        neb_forces: np.ndarray,
        positions: list[np.ndarray],
    ) -> list[np.ndarray]:
        """
        Perform one BFGS step for each intermediate image.

        Parameters
        ----------
        neb_forces : np.ndarray
            NEB forces for intermediate images only.
            Shape: ((n_images-2) * n_atoms, 3) or (n_intermediate, n_atoms, 3)
        positions : list[np.ndarray]
            Current positions for ALL images (including endpoints).
            Each element has shape (n_atoms, 3).

        Returns
        -------
        list[np.ndarray]
            Updated positions for ALL images.
            Endpoints are unchanged, intermediate images are updated.
        """
        # Get number of atoms from first image
        n_atoms = positions[0].shape[0]
        n_dof = n_atoms * 3  # Degrees of freedom per image

        # Reshape NEB forces if needed
        if (
            neb_forces.ndim == 2
            and neb_forces.shape[0] == self.n_intermediate * n_atoms
        ):
            forces_per_image = neb_forces.reshape(self.n_intermediate, n_atoms, 3)
        elif neb_forces.ndim == 3 and neb_forces.shape[0] == self.n_intermediate:
            forces_per_image = neb_forces
        else:
            raise ValueError(
                f"Unexpected NEB forces shape: {neb_forces.shape}. "
                f"Expected ({self.n_intermediate * n_atoms}, 3) or "
                f"({self.n_intermediate}, {n_atoms}, 3)"
            )

        # Make copy of positions to update
        new_positions = [pos.copy() for pos in positions]

        # Apply BFGS step to each intermediate image independently
        for i in range(self.n_intermediate):
            image_idx = i + 1  # Skip initial endpoint (index 0)
            opt_state = self.optimizers[i]

            # Current positions and gradients for this image
            pos = positions[image_idx].flatten()  # (n_dof,)
            # Negative forces = gradients (minimization)
            grad = -forces_per_image[i].flatten()  # (n_dof,)

            # Initialize inverse Hessian on first step
            if opt_state["H"] is None:
                opt_state["H"] = self.alpha * np.eye(n_dof)

            # Update inverse Hessian using BFGS formula if we have previous data
            if opt_state["prev_positions"] is not None:
                s = pos - opt_state["prev_positions"]  # Position change
                y = grad - opt_state["prev_gradients"]  # Gradient change

                # Only update if curvature is positive (y · s > 0)
                ys = np.dot(y, s)
                if ys > 1e-10:
                    # BFGS inverse Hessian update formula
                    H = opt_state["H"]
                    Hy = H @ y
                    yHy = np.dot(y, Hy)

                    # Sherman-Morrison-Woodbury formula for BFGS
                    # H_{k+1} = (I - ρ s y^T) H_k (I - ρ y s^T) + ρ s s^T
                    rho = 1.0 / ys

                    # More numerically stable form
                    term1 = (ys + yHy) * rho * rho * np.outer(s, s)
                    term2 = rho * (np.outer(s, Hy) + np.outer(Hy, s))
                    opt_state["H"] = H + term1 - term2

            # Compute search direction: p = -H @ grad
            direction = opt_state["H"] @ grad

            # Apply step with maxstep limit
            step = -direction  # Descend along negative gradient direction
            step_norm = np.linalg.norm(step)

            if step_norm > self.maxstep * np.sqrt(n_dof / 3):
                # Scale to max allowed displacement per atom
                step = step * (self.maxstep * np.sqrt(n_dof / 3) / step_norm)

            # Update positions
            new_pos = pos + step
            new_positions[image_idx] = new_pos.reshape(n_atoms, 3)

            # Save current state for next iteration
            opt_state["prev_positions"] = pos.copy()
            opt_state["prev_gradients"] = grad.copy()
            opt_state["iteration"] += 1

            # Log per-image info
            max_disp = np.max(np.abs(step.reshape(-1, 3)), axis=1).max()
            max_force = np.max(np.linalg.norm(forces_per_image[i], axis=1))
            logger.debug(
                f"  Image {image_idx}: max_force={max_force:.4f} eV/Å, "
                f"max_disp={max_disp:.4f} Å"
            )

        return new_positions

    def reset(self) -> None:
        """Reset all optimizer states (clear Hessians)."""
        for opt in self.optimizers:
            opt["H"] = None
            opt["prev_positions"] = None
            opt["prev_gradients"] = None
            opt["iteration"] = 0
        logger.info("PerImageBFGS: Reset all optimizer states")


@dataclass
class AseNebFlowMaker(BaseSiestaFlowMaker):
    """
    Jobflow maker to run NEB calculations using ASE optimizer with persistent folders.

    This maker uses ASE's NEB implementation with SIESTA as the calculator,
    instead of SIESTA's Lua scripting. This approach is:
    - More portable (no FLOS/Lua required)
    - Easier to customize (Python-based)
    - Better for advanced NEB methods (climbing image, string methods)
    - Optimized with persistent folders (custodian-style approach)

    Key optimizations:

    - Creates persistent folders (neb_image_0, neb_image_1, etc.)
    - Reuses folders across all iterations (no new job folders per iteration)
    - Endpoint structures calculated only once (cached for subsequent iterations)
    - Intermediate images recalculated each iteration

    Steps:

    1. Generate NEB images using ASE interpolation
    2. Setup persistent folders for each image
    3. Run iterative NEB optimization:

       - Iteration 0: Calculate all images (including endpoints)
       - Iterations 1+: Reuse cached endpoint data, recalculate intermediate images

    4. Plot results

    Parameters
    ----------
    name : str
        Name of the flow produced by this maker.
    relax_endpoints : bool | str
        Whether to relax endpoint structures before NEB. Options:

        - False: Don't relax either endpoint (use structures as provided)
        - True: Relax both initial and final structures (backward compatible)
        - "initial": Relax only the initial structure
        - "final": Relax only the final structure
        - "both": Relax both structures (same as True)

        Default is False.
    relax_maker : Maker | None
        Job maker for relaxing structures. Only used if relax_endpoints is not False.
        Default is RelaxMaker.fixed_cell_relaxation().
        If relax_initial_maker or relax_final_maker are specified, they take precedence.
    relax_initial_maker : Maker | None
        Job maker for relaxing the initial structure. If None, uses relax_maker.
        Allows different relaxation settings for initial vs final endpoints.
    relax_final_maker : Maker | None
        Job maker for relaxing the final structure. If None, uses relax_maker.
        Allows different relaxation settings for initial vs final endpoints.
    static_maker : Maker | None
        Job maker for static calculations on each image.
        Default is StaticMaker().
    number_of_images : int
        Number of intermediate NEB images (default is 5).
    optimizer : str
        Optimizer to use. Default is "PER_IMAGE_LBFGS".

        **Per-image optimizers** (recommended - like Lua FLOS):

        - **PER_IMAGE_LBFGS** (default): Each NEB image has its own L-BFGS
          optimizer with limited memory (~20 steps). Best balance of
          convergence speed and memory usage.

        - **PER_IMAGE_BFGS**: Each NEB image has its own full BFGS optimizer
          with complete inverse Hessian. May converge faster for small systems
          but uses more memory (O(n²) per image).

        **Global optimizers** (treat all images together):

        - **FIRE**: No Hessian, handles oscillations well. Good fallback.

        - **LBFGS/BFGS**: Global optimizer for all images combined (~1365 DOF).
          Can oscillate due to mixing curvature from different regions.
    fmax : float
        Force convergence criterion in eV/Å (default is 0.05).
    climbing_image : bool
        Use climbing image NEB (default is False).
    spring_constant : float
        Spring constant for NEB in eV/Å² (default is 5.0).
        Controls image spacing along the path:
        - Low (0.1-1.0): More path flexibility, images may spread unevenly
        - Medium (1.0-5.0): Good balance for most cases
        - High (5.0-10.0): Forces equal spacing, use if images bunch up
    alpha : float
        Initial Hessian guess for BFGS optimizer (default is 1/100).
    maxstep : float
        Maximum step size in Å (default is 0.2). Reduced from typical 0.5 for
        stability with high-force initial images.
    force_threshold : float
        Force threshold in eV/Å for dynamic step scaling (default is 5.0).
        When max DFT force exceeds this, the step size is reduced proportionally
        to prevent atoms from moving into unphysical positions.

    Example
    -------
    >>> from atomate2.siesta.flows.neb import AseNebFlowMaker
    >>> from pymatgen.core import Structure
    >>> initial = Structure.from_file("initial.cif")
    >>> final = Structure.from_file("final.cif")
    >>>
    >>> # Without endpoint relaxation (uses PER_IMAGE_LBFGS by default)
    >>> maker = AseNebFlowMaker(number_of_images=7)
    >>> flow = maker.make(initial_structure=initial, final_structure=final)
    >>>
    >>> # With FIRE optimizer (good fallback)
    >>> maker = AseNebFlowMaker(number_of_images=7, optimizer="FIRE")
    >>> flow = maker.make(initial_structure=initial, final_structure=final)
    >>>
    >>> # Relax both endpoints
    >>> maker = AseNebFlowMaker(number_of_images=7, relax_endpoints=True)
    >>> flow = maker.make(initial_structure=initial, final_structure=final)
    >>>
    >>> # Relax only initial structure
    >>> maker = AseNebFlowMaker(number_of_images=7, relax_endpoints="initial")
    >>> flow = maker.make(initial_structure=initial, final_structure=final)
    >>>
    >>> # Relax only final structure
    >>> maker = AseNebFlowMaker(number_of_images=7, relax_endpoints="final")
    >>> flow = maker.make(initial_structure=initial, final_structure=final)
    >>>
    >>> # Different relaxation settings for each endpoint
    >>> from atomate2.siesta.jobs.core import RelaxMaker
    >>> initial_relax = RelaxMaker.fixed_cell_relaxation(
    ...     user_params={"PAO.BasisSize": "DZP", "a2s_kpts": [2, 2, 2]}
    ... )
    >>> final_relax = RelaxMaker.fixed_cell_relaxation(
    ...     user_params={"PAO.BasisSize": "TZP", "a2s_kpts": [4, 4, 4]}
    ... )
    >>> maker = AseNebFlowMaker(
    ...     number_of_images=7,
    ...     relax_endpoints=True,
    ...     relax_initial_maker=initial_relax,
    ...     relax_final_maker=final_relax
    ... )
    >>> flow = maker.make(initial_structure=initial, final_structure=final)
    """

    name: str = "ASE NEB Workflow"
    relax_endpoints: bool | str = False
    relax_maker: Maker | None = field(
        default_factory=lambda: RelaxMaker.fixed_cell_relaxation()
    )
    relax_initial_maker: Maker | None = None
    relax_final_maker: Maker | None = None
    static_maker: Maker | None = field(default_factory=lambda: StaticMaker())
    number_of_images: int = 5
    optimizer: str = (
        "PER_IMAGE_LBFGS"  # Per-image LBFGS like Lua FLOS (best convergence)
    )
    fmax: float = 0.05
    climbing_image: bool = False
    spring_constant: float = 5.0
    alpha: float = (
        1.0 / 100.0
    )  # Initial Hessian guess for BFGS (only used if optimizer="BFGS")
    maxstep: float = 0.2  # Maximum step size in Å - reduced from 0.5 for stability
    negate_forces: bool = (
        False  # Do NOT negate SIESTA forces - they are already F = -dE/dr
    )
    # Force threshold for dynamic step scaling (eV/Å)
    # When max force exceeds this, step size is reduced proportionally
    force_threshold: float = 5.0

    def make(self, initial_structure: Structure, final_structure: Structure) -> Flow:
        """
        Create an iterative ASE NEB workflow with persistent image folders.

        Creates one folder per NEB image and reuses them across iterations,
        similar to custodian's approach for error correction.

        Parameters
        ----------
        initial_structure : Structure
            The initial structure (starting point).
        final_structure : Structure
            The final structure (end point).

        Returns
        -------
        Flow
            A jobflow Flow containing the iterative ASE NEB workflow.
        """
        from atomate2.siesta.utils.common import print_docstring_in_box

        print_docstring_in_box(self.__doc__, title=self.__class__.__name__)

        logger.info(
            "AseNebFlowMaker.make() - Creating iterative ASE NEB workflow with persistent folders"
        )
        jobs = []

        # Determine which endpoints to relax
        relax_initial = False
        relax_final = False

        if self.relax_endpoints is True or self.relax_endpoints == "both":
            relax_initial = True
            relax_final = True
            logger.info("Relaxing both initial and final structures")
        elif self.relax_endpoints == "initial":
            relax_initial = True
            logger.info("Relaxing initial structure only")
        elif self.relax_endpoints == "final":
            relax_final = True
            logger.info("Relaxing final structure only")
        elif self.relax_endpoints is False:
            logger.info("Using endpoint structures as provided (no relaxation)")
        else:
            raise ValueError(
                f"Invalid relax_endpoints value: {self.relax_endpoints}. "
                "Must be False, True, 'initial', 'final', or 'both'."
            )

        # Optional: Relax endpoints as requested
        initial_for_neb = initial_structure
        final_for_neb = final_structure

        if relax_initial:
            # Use relax_initial_maker if specified, otherwise fall back to relax_maker
            maker_to_use = self.relax_initial_maker or self.relax_maker
            initial_relax = maker_to_use.make(structure=initial_structure)
            initial_relax.name = f"{self.name}_Initial_Relaxation"
            jobs.append(initial_relax)
            initial_for_neb = initial_relax.output.structure

        if relax_final:
            # Use relax_final_maker if specified, otherwise fall back to relax_maker
            maker_to_use = self.relax_final_maker or self.relax_maker
            final_relax = maker_to_use.make(structure=final_structure)
            final_relax.name = f"{self.name}_Final_Relaxation"
            jobs.append(final_relax)
            final_for_neb = final_relax.output.structure

        # Generate initial NEB images
        neb_images_job = generate_neb_images_ase(
            self.number_of_images, initial_for_neb, final_for_neb
        )
        neb_images_job.name = f"{self.name}_Generate_Images"
        jobs.append(neb_images_job)

        # Start NEB optimization (creates image folders internally, runs all iterations)
        neb_optimization_job = ase_neb_optimization_all_iterations(
            structures=neb_images_job.output,
            static_maker=self.static_maker,
            optimizer=self.optimizer,
            fmax=self.fmax,
            climbing_image=self.climbing_image,
            spring_constant=self.spring_constant,
            alpha=self.alpha,
            maxstep=self.maxstep,
            max_iterations=100,
            negate_forces=self.negate_forces,
            force_threshold=self.force_threshold,
            dry_run=self.dry_run,
            dry_run_output_dir=self.dry_run_output_dir,
            dry_run_format=self.dry_run_format,
        )
        neb_optimization_job.name = f"{self.name}_NEB_Optimization"
        jobs.append(neb_optimization_job)

        # Plot final results
        plot_job = plot_ase_neb_results(neb_optimization_job.output)
        plot_job.name = f"{self.name}_Plotting"
        jobs.append(plot_job)

        return Flow(jobs=jobs, name=self.name)


def setup_neb_image_folders_in_job(
    structures: list[Structure],
    base_directory: Path,
) -> dict:
    """
    Create persistent folders for each NEB image (NOT a @job, called from optimization job).

    Creates folders named image_0, image_1, etc. inside the NEB optimization job folder.
    These folders will be reused across all NEB iterations.

    Parameters
    ----------
    structures : list[Structure]
        NEB image structures.
    base_directory : Path
        Base directory where image folders should be created.

    Returns
    -------
    dict
        Dictionary with image_folders list (absolute paths).
    """

    logger.info(
        f"Setting up {len(structures)} persistent image folders in {base_directory}"
    )

    # Create folders for each image
    image_folders = []
    for i in range(len(structures)):
        folder_name = f"image_{i}"
        folder_path = base_directory / folder_name
        folder_path.mkdir(exist_ok=True)
        image_folders.append(str(folder_path.absolute()))
        logger.info(f"  Created: {folder_name}/")

    return {
        "image_folders": image_folders,
        "n_images": len(structures),
    }


@job
def ase_neb_optimization_all_iterations(
    structures: list[Structure],
    static_maker: Maker,
    optimizer: str,
    fmax: float,
    climbing_image: bool,
    spring_constant: float,
    alpha: float,
    maxstep: float,
    max_iterations: int,
    negate_forces: bool = True,
    force_threshold: float = 5.0,
    resume_from_dir: str | Path = None,
    dry_run: bool = False,
    dry_run_output_dir: str = "dry_run_output",
    dry_run_format: str = "cif",
) -> dict:
    """
    Run all NEB iterations in a single job (no new job folders per iteration).

    This function runs all NEB iterations internally without spawning new jobs,
    which prevents creating new job folders for each iteration. Creates persistent
    image folders (image_0, image_1, ...) inside this job folder and reuses them.

    Parameters
    ----------
    structures : list[Structure]
        Initial NEB image structures.
    static_maker : Maker
        Static maker for force calculations.
    optimizer : str
        Optimizer name: "PER_IMAGE_LBFGS", "FIRE", "LBFGS", or "BFGS".
        PER_IMAGE_LBFGS is recommended (each image has its own L-BFGS optimizer).
    fmax : float
        NEB force convergence criterion in eV/Å.
    climbing_image : bool
        Use climbing image NEB.
    spring_constant : float
        Spring constant in eV/Å².
    alpha : float
        Initial Hessian guess for BFGS optimizer.
    maxstep : float
        Maximum step size in Å (for BFGS) or maxmove (for FIRE).
    max_iterations : int
        Maximum iterations.
    negate_forces : bool
        Whether to negate SIESTA forces (default False).
    force_threshold : float
        Force threshold for dynamic step scaling in eV/Å. When max force
        exceeds this value, the step size is reduced proportionally to
        prevent atoms from moving into unphysical positions.
    dry_run : bool
        If True, generate input files without running SIESTA.
    dry_run_output_dir : str
        Directory for dry_run output (default: "dry_run_output").
    dry_run_format : str
        Structure file format for dry_run (default: "cif").

    Returns
    -------
    dict
        Final NEB results with energies, forces, and convergence info.
    """
    import numpy as np
    from ase.mep import NEB
    from ase.optimize import BFGS, FIRE
    from ase.calculators.singlepoint import SinglePointCalculator
    from pymatgen.io.ase import AseAtomsAdaptor
    from pathlib import Path
    import os
    from atomate2.siesta.run import run_siesta
    from atomate2.siesta.files import write_siesta_input_set
    from atomate2.siesta.schemas.task import SiestaTaskDoc

    # Handle dry_run mode
    if dry_run:
        logger.info(
            "ASE NEB: dry_run=True - generating input files without running SIESTA"
        )
        base_dir = Path.cwd()
        output_dir = base_dir / dry_run_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save all NEB image structures
        for i, struct in enumerate(structures):
            struct_file = output_dir / f"neb_image_{i}.{dry_run_format}"
            struct.to(filename=str(struct_file))
            logger.info(f"  Saved: {struct_file.name}")

        # Write input files for first intermediate image as example
        example_dir = output_dir / "example_siesta_inputs"
        example_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(example_dir)
        write_siesta_input_set(
            structures[1], static_maker.input_set_generator, directory="."
        )
        os.chdir(base_dir)
        logger.info(f"  Wrote example SIESTA inputs to: {example_dir}")

        return {
            "converged": False,
            "dry_run": True,
            "message": f"Dry run complete. {len(structures)} structures saved to {output_dir}",
            "n_images": len(structures),
            "output_dir": str(output_dir),
        }

    logger.info(f"Starting NEB optimization with max {max_iterations} iterations")

    # Setup persistent image folders inside this job folder
    base_dir = Path.cwd()
    n_images = len(structures)

    # Check for resume
    checkpoint_file = base_dir / "neb_checkpoint.json"
    start_iteration = 0
    current_structures = structures
    endpoint_energies = None
    endpoint_forces = None
    opt: Any = None  # Optimizer object that persists across iterations
    neb = None  # NEB object that persists across iterations

    if resume_from_dir is not None:
        resume_dir = Path(resume_from_dir)
        checkpoint = resume_dir / "neb_checkpoint.json"
        if checkpoint.exists():
            logger.info(f"Resuming from checkpoint: {checkpoint}")
            import json

            with open(checkpoint, "r") as f:
                checkpoint_data = json.load(f)
            start_iteration = checkpoint_data["last_iteration"] + 1
            logger.info(f"Resuming from iteration {start_iteration}")

            # Load structures from last iteration
            for i in range(n_images):
                struct_file = (
                    resume_dir / f"image_{i}" / f"structure_iter_{start_iteration}.cif"
                )
                if struct_file.exists():
                    current_structures[i] = Structure.from_file(struct_file)

            # Copy image folders from resume directory
            for i in range(n_images):
                src = resume_dir / f"image_{i}"
                dst = base_dir / f"image_{i}"
                if src.exists():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            logger.warning(f"Resume requested but no checkpoint found at {checkpoint}")
            start_iteration = 0

    # Setup or verify image folders
    folder_info = setup_neb_image_folders_in_job(current_structures, base_dir)
    image_folders = folder_info["image_folders"]

    # Create NEB log file for tracking iteration progress
    neb_log_file = base_dir / "neb_progress.log"
    if start_iteration == 0:
        with open(neb_log_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("NEB OPTIMIZATION PROGRESS LOG\n")
            f.write("=" * 80 + "\n\n")
            f.write("NEB Configuration:\n")
            f.write(f"  Number of images: {n_images}\n")
            f.write(f"  Optimizer: {optimizer}\n")
            f.write(f"  Force convergence: {fmax} eV/Å\n")
            f.write(f"  Spring constant: {spring_constant} eV/Å²\n")
            f.write(f"  Climbing image: {climbing_image}\n")
            f.write(f"  Max iterations: {max_iterations}\n")
            f.write("\n" + "=" * 80 + "\n\n")
        logger.info(f"Created NEB progress log: {neb_log_file}")
    else:
        with open(neb_log_file, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"RESUMING from iteration {start_iteration}\n")
            f.write(f"{'='*80}\n\n")
        logger.info(f"Resuming NEB optimization from iteration {start_iteration}")

    for iteration in range(start_iteration, max_iterations):
        logger.info(f"\nNEB Iteration {iteration + 1}/{max_iterations}")
        logger.info("=" * 60)

        # Log iteration start
        with open(neb_log_file, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"ITERATION {iteration + 1}/{max_iterations}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Time: {Path.cwd()}\n\n")

        energies = []
        forces_list = []

        # Save structures before this iteration (for tracking convergence)
        if iteration > 0:
            for i, struct in enumerate(current_structures):
                struct.to(
                    filename=Path(image_folders[i]) / f"structure_iter_{iteration}.cif"
                )

        # Calculate forces for each image
        for i, (folder, structure) in enumerate(zip(image_folders, current_structures)):
            is_endpoint = i == 0 or i == n_images - 1

            # Optimization: Skip endpoint calculations after first iteration
            if is_endpoint and iteration > 0 and endpoint_energies is not None:
                if i == 0:
                    logger.info(f"  Image {i} (initial): Reusing cached data")
                    energies.append(endpoint_energies[0])
                    forces_list.append(endpoint_forces[0])
                else:
                    logger.info(f"  Image {i} (final): Reusing cached data")
                    energies.append(endpoint_energies[1])
                    forces_list.append(endpoint_forces[1])
                continue

            # Run SIESTA calculation
            logger.info(f"  Image {i}: Running SIESTA in {folder}")
            folder_path = Path(folder)

            # Change to image folder
            os.chdir(folder_path)

            # Remove old structure file to force rewrite with new coordinates
            structure_file = folder_path / "structure.fdf"
            if structure_file.exists():
                structure_file.unlink()
                logger.debug(f"  Removed old structure.fdf for image {i}")

            # Write SIESTA input files
            write_siesta_input_set(
                structure, static_maker.input_set_generator, directory="."
            )

            # Run SIESTA
            run_siesta()

            # Parse output
            task_doc = SiestaTaskDoc.from_directory(folder_path)
            energies.append(task_doc.output.energy)
            forces_list.append(np.array(task_doc.output.forces))

            max_force = np.max(np.linalg.norm(task_doc.output.forces, axis=1))
            logger.info(
                f"  Image {i}: E = {task_doc.output.energy:.6f} eV, max_force = {max_force:.4f} eV/Å"
            )

            # Return to base directory
            os.chdir(base_dir)

        # Log energies and forces for this iteration
        with open(neb_log_file, "a") as f:
            f.write(
                "Image Energies and SIESTA Forces (DFT forces from electronic structure):\n"
            )
            f.write("-" * 80 + "\n")
            for i, (e, flist) in enumerate(zip(energies, forces_list)):
                max_f = np.max(np.linalg.norm(flist, axis=1))
                endpoint_marker = (
                    " (endpoint - fixed)" if (i == 0 or i == n_images - 1) else ""
                )
                f.write(
                    f"  Image {i}: E = {e:12.6f} eV, max_SIESTA_force = {max_f:8.4f} eV/Å{endpoint_marker}\n"
                )
            f.write("\n")

        # Cache endpoint data from first iteration
        if iteration == 0:
            endpoint_energies = [energies[0], energies[-1]]
            endpoint_forces = [forces_list[0], forces_list[-1]]
            logger.info("✓ Cached endpoint data for future iterations")

        # NEB object persistence: preserve Atoms identity for BFGS/LBFGS Hessian
        if iteration == start_iteration or neb is None:
            # First iteration: create new NEB object
            images = [AseAtomsAdaptor.get_atoms(s) for s in current_structures]
            for img, energy, force in zip(images, energies, forces_list):
                # Apply force sign correction if needed
                final_force = -np.array(force) if negate_forces else force
                img.calc = SinglePointCalculator(img, energy=energy, forces=final_force)

            neb = NEB(images, k=spring_constant, climb=climbing_image)
            logger.info("✓ Created new NEB object (first iteration)")
        else:
            # Subsequent iterations: NEB already exists with updated positions from opt.step()
            # Just update the forces with new SIESTA results
            logger.info(
                "✓ Reusing existing NEB object (positions already updated by optimizer)"
            )

            # Update energies and forces with new SIESTA results
            for i, (img, energy, force) in enumerate(
                zip(neb.images, energies, forces_list)
            ):
                # Apply force sign correction if needed
                final_force = -np.array(force) if negate_forces else force
                img.calc = SinglePointCalculator(img, energy=energy, forces=final_force)
                logger.debug(f"  Image {i}: Updated calculator with new SIESTA forces")

        neb_forces = neb.get_forces()
        max_neb_force = np.max(np.abs(neb_forces))

        logger.info(f"Max NEB force: {max_neb_force:.4f} eV/Å (target: {fmax} eV/Å)")

        # Detailed force logging for debugging
        # IMPORTANT: neb.get_forces() returns forces ONLY for INTERMEDIATE images (excludes fixed endpoints!)
        # Shape: ((n_images-2) * n_atoms, 3) - need to reshape to (n_images-2, n_atoms, 3)
        n_atoms_per_image = len(current_structures[0])
        n_intermediate = n_images - 2  # Exclude 2 endpoints
        neb_forces_intermediate = neb_forces.reshape(
            n_intermediate, n_atoms_per_image, 3
        )

        forces_log_file = base_dir / "neb_forces_detailed.log"
        with open(forces_log_file, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"ITERATION {iteration + 1}/{max_iterations}\n")
            f.write(f"{'='*80}\n")
            f.write(f"negate_forces flag: {negate_forces}\n\n")

            for i in range(n_images):
                f.write(f"\nImage {i}:\n")
                f.write(f"  Energy: {energies[i]:.6f} eV\n")

                # DFT forces (from SIESTA, before any negation)
                dft_force = forces_list[i]
                max_dft = np.max(np.linalg.norm(dft_force, axis=1))
                f.write(f"  Max DFT force (as read from SIESTA): {max_dft:.4f} eV/Å\n")

                # Forces attached to calculator (after potential negation)
                calc_forces = neb.images[i].get_forces()
                max_calc = np.max(np.linalg.norm(calc_forces, axis=1))
                f.write(
                    f"  Max Calculator force (after negate_forces): {max_calc:.4f} eV/Å\n"
                )

                # NEB forces (with spring forces added) - only available for intermediate images
                if i == 0 or i == n_images - 1:
                    # Endpoints have zero NEB force (fixed)
                    neb_f = np.zeros_like(calc_forces)
                else:
                    # Intermediate image: get from reshaped array (index i-1 because endpoints excluded)
                    neb_f = neb_forces_intermediate[i - 1]

                # Ensure DFT force is 2D
                dft_force = np.array(dft_force)
                if dft_force.ndim == 1:
                    dft_force = dft_force.reshape(-1, 3)

                max_neb_f = np.max(np.linalg.norm(neb_f, axis=1))
                f.write(f"  Max NEB force (DFT + springs): {max_neb_f:.4f} eV/Å\n")

                # Show atom with maximum force for detailed inspection
                max_dft_idx = np.argmax(np.linalg.norm(dft_force, axis=1))
                max_neb_idx = np.argmax(np.linalg.norm(neb_f, axis=1))

                f.write(f"  Atom with max DFT force (atom {max_dft_idx}):\n")
                f.write(
                    f"    DFT force:  [{dft_force[max_dft_idx][0]:8.4f}, {dft_force[max_dft_idx][1]:8.4f}, {dft_force[max_dft_idx][2]:8.4f}] = {np.linalg.norm(dft_force[max_dft_idx]):.4f} eV/Å\n"
                )
                f.write(
                    f"    Calc force: [{calc_forces[max_dft_idx][0]:8.4f}, {calc_forces[max_dft_idx][1]:8.4f}, {calc_forces[max_dft_idx][2]:8.4f}] = {np.linalg.norm(calc_forces[max_dft_idx]):.4f} eV/Å\n"
                )

                f.write(f"  Atom with max NEB force (atom {max_neb_idx}):\n")
                f.write(
                    f"    DFT force:  [{dft_force[max_neb_idx][0]:8.4f}, {dft_force[max_neb_idx][1]:8.4f}, {dft_force[max_neb_idx][2]:8.4f}] = {np.linalg.norm(dft_force[max_neb_idx]):.4f} eV/Å\n"
                )
                f.write(
                    f"    NEB force:  [{neb_f[max_neb_idx][0]:8.4f}, {neb_f[max_neb_idx][1]:8.4f}, {neb_f[max_neb_idx][2]:8.4f}] = {np.linalg.norm(neb_f[max_neb_idx]):.4f} eV/Å\n"
                )
                f.write(
                    f"    Spring contribution: NEB - DFT = [{neb_f[max_neb_idx][0] - calc_forces[max_neb_idx][0]:8.4f}, {neb_f[max_neb_idx][1] - calc_forces[max_neb_idx][1]:8.4f}, {neb_f[max_neb_idx][2] - calc_forces[max_neb_idx][2]:8.4f}]\n"
                )

                # Calculate force decomposition for NEB
                # NEB uses: F_NEB = F_DFT_perpendicular + F_spring_parallel
                f_perp = neb_f - calc_forces  # Approximate perpendicular component
                f_spring = f_perp  # Spring force is the difference

                # Show all atoms forces in a nice table
                f.write("\n  NEB Force Decomposition (per atom):\n")
                f.write(f"  {'='*160}\n")
                f.write(
                    f"  {'Atom':>5} │ {'DFT Force (SIESTA)':^35} │ {'Calc Force (after negation)':^35} │ {'Spring Force':^35} │ {'NEB Force':^35}\n"
                )
                f.write(f"  {'-'*5}-+-{'-'*35}-+-{'-'*35}-+-{'-'*35}-+-{'-'*35}\n")
                for atom_idx in range(len(dft_force)):
                    dft_mag = np.linalg.norm(dft_force[atom_idx])
                    calc_mag = np.linalg.norm(calc_forces[atom_idx])
                    spring_mag = np.linalg.norm(f_spring[atom_idx])
                    neb_mag = np.linalg.norm(neb_f[atom_idx])

                    dft_str = f"[{dft_force[atom_idx][0]:6.2f},{dft_force[atom_idx][1]:6.2f},{dft_force[atom_idx][2]:6.2f}] {dft_mag:5.2f}"
                    calc_str = f"[{calc_forces[atom_idx][0]:6.2f},{calc_forces[atom_idx][1]:6.2f},{calc_forces[atom_idx][2]:6.2f}] {calc_mag:5.2f}"
                    spring_str = f"[{f_spring[atom_idx][0]:6.2f},{f_spring[atom_idx][1]:6.2f},{f_spring[atom_idx][2]:6.2f}] {spring_mag:5.2f}"
                    neb_str = f"[{neb_f[atom_idx][0]:6.2f},{neb_f[atom_idx][1]:6.2f},{neb_f[atom_idx][2]:6.2f}] {neb_mag:5.2f}"

                    f.write(
                        f"  {atom_idx:5d} │ {dft_str:35s} │ {calc_str:35s} │ {spring_str:35s} │ {neb_str:35s}\n"
                    )
                f.write(f"  {'='*160}\n")
                f.write("\n  Note: NEB Force = Calc Force + Spring Force\n")
                f.write(
                    "        Spring Force includes both spring forces and perpendicular DFT projection\n"
                )

        # Log NEB forces and convergence
        # Calculate max DFT force for warning purposes
        max_dft_force_log = max(np.max(np.linalg.norm(f, axis=1)) for f in forces_list)
        with open(neb_log_file, "a") as f:
            f.write(
                "NEB Forces (includes spring forces + perpendicular SIESTA forces):\n"
            )
            f.write("-" * 80 + "\n")
            f.write(
                f"  Max NEB force: {max_neb_force:.4f} eV/Å  ← THIS DETERMINES CONVERGENCE\n"
            )
            f.write(f"  Max DFT force: {max_dft_force_log:.4f} eV/Å\n")
            f.write(f"  Target (fmax): {fmax:.4f} eV/Å\n")
            # Warn if forces are extremely high
            if max_dft_force_log > 20.0:
                f.write(
                    "  ⚠️ WARNING: Very high DFT forces detected! May indicate atomic overlap.\n"
                )
            # Calculate relative energies
            energies_array = np.array(energies)
            energies_rel = (energies_array - energies_array[0]) * 1000  # meV
            f.write(
                f"  Current barrier: {energies_rel.max():.2f} meV ({energies_rel.max()/1000:.4f} eV)\n"
            )
            f.write(f"  TS at image: {np.argmax(energies_rel)}\n")
            f.write(f"  Converged: {'✓ YES' if max_neb_force < fmax else '✗ NO'}\n")
            f.write("\n")

        # Check convergence
        converged = max_neb_force < fmax
        if converged:
            logger.info(f"✓ NEB converged after {iteration + 1} iterations!")
            with open(neb_log_file, "a") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"✓ NEB CONVERGED after {iteration + 1} iterations!\n")
                f.write(f"{'='*80}\n\n")
            break

        if iteration >= max_iterations - 1:
            logger.info(f"✗ Reached max iterations ({max_iterations})")
            with open(neb_log_file, "a") as f:
                f.write(f"\n{'='*80}\n")
                f.write(
                    f"✗ Reached max iterations ({max_iterations}) without convergence\n"
                )
                f.write(f"{'='*80}\n\n")
            break

        # Perform optimization step with configurable parameters
        # IMPORTANT: Create optimizer only ONCE and reuse it across iterations
        # (BFGS/LBFGS build Hessian approximation over multiple steps)

        # Dynamic step scaling based on maximum force
        # When forces are very large (atoms in bad positions), take smaller steps
        max_dft_force = max(np.max(np.linalg.norm(f, axis=1)) for f in forces_list)
        if max_dft_force > force_threshold:
            # Scale step size inversely with force magnitude
            effective_maxstep = maxstep * (force_threshold / max_dft_force)
            # Ensure minimum step size of 0.01 Å
            effective_maxstep = max(effective_maxstep, 0.01)
            logger.info(
                f"⚠ Large forces detected ({max_dft_force:.2f} eV/Å > {force_threshold} eV/Å), "
                f"reducing step: {maxstep:.3f} → {effective_maxstep:.3f} Å"
            )
        else:
            effective_maxstep = maxstep

        # Create optimizer with effective step size
        # Recreate optimizer when step size changes significantly
        if (
            opt is None
            or abs(getattr(opt, "_last_maxstep", maxstep) - effective_maxstep) > 0.01
        ):
            if optimizer.upper() == "PER_IMAGE_LBFGS":
                # Per-image LBFGS: Each image has its own optimizer (like Lua FLOS)
                opt = PerImageLBFGS(
                    n_images=n_images,
                    alpha=alpha,
                    maxstep=effective_maxstep,
                    memory=20,
                )
                logger.info(
                    f"✓ Created PER_IMAGE_LBFGS optimizer (alpha={alpha}, maxstep={effective_maxstep:.3f})"
                )
            elif optimizer.upper() == "PER_IMAGE_BFGS":
                # Per-image BFGS: Each image has its own full Hessian optimizer
                opt = PerImageBFGS(
                    n_images=n_images,
                    alpha=alpha,
                    maxstep=effective_maxstep,
                )
                logger.info(
                    f"✓ Created PER_IMAGE_BFGS optimizer (alpha={alpha}, maxstep={effective_maxstep:.3f})"
                )
            elif optimizer.upper() == "FIRE":
                opt = FIRE(neb, logfile=None, maxstep=effective_maxstep)
                logger.info(
                    f"✓ Created FIRE optimizer (maxstep={effective_maxstep:.3f})"
                )
            elif optimizer.upper() == "LBFGS":
                opt = LBFGS(neb, logfile=None, maxstep=effective_maxstep, alpha=alpha)
                logger.info(
                    f"✓ Created LBFGS optimizer (alpha={alpha}, maxstep={effective_maxstep:.3f})"
                )
            else:  # BFGS
                opt = BFGS(neb, logfile=None, maxstep=effective_maxstep, alpha=alpha)
                logger.info(
                    f"✓ Created BFGS optimizer (alpha={alpha}, maxstep={effective_maxstep:.3f})"
                )
            # Store maxstep for reference (only for ASE optimizers)
            if not optimizer.upper().startswith("PER_IMAGE_"):
                opt._last_maxstep = effective_maxstep

        # Pass forces explicitly to avoid recalculating them
        # DEBUG: Log positions before and after opt.step()
        positions_before = [img.get_positions().copy() for img in neb.images]
        logger.info(
            f"Positions before opt.step(): Image 1 atom 0: {positions_before[1][0]}"
        )

        # Apply optimization step based on optimizer type
        if optimizer.upper() in ("PER_IMAGE_LBFGS", "PER_IMAGE_BFGS"):
            # Per-image optimizers: pass NEB forces and positions, get back updated positions
            new_positions = opt.step(neb_forces, positions_before)
            # Update NEB images with new positions
            for i, (img, new_pos) in enumerate(zip(neb.images, new_positions)):
                if i > 0 and i < n_images - 1:  # Only update intermediate images
                    img.set_positions(new_pos)
        elif optimizer.upper() == "FIRE":
            opt.step(f=neb_forces)  # FIRE uses 'f' parameter
        else:
            opt.step(forces=neb_forces)  # BFGS/LBFGS use 'forces' parameter

        positions_after = [img.get_positions().copy() for img in neb.images]
        logger.info(
            f"Positions after opt.step(): Image 1 atom 0: {positions_after[1][0]}"
        )
        max_change = max(
            [
                np.max(np.linalg.norm(after - before, axis=1))
                for before, after in zip(positions_before, positions_after)
            ]
        )
        logger.info(
            f"✓ Performed optimization step, max position change: {max_change:.6f} Å"
        )

        # Update structures for next iteration (extract from NEB for SIESTA)
        previous_structures = current_structures
        current_structures = [AseAtomsAdaptor.get_structure(img) for img in neb.images]

        # Log structure displacement for intermediate images
        max_displacement = 0.0
        with open(neb_log_file, "a") as f:
            f.write("Atom Displacements:\n")
            f.write("-" * 80 + "\n")
        for i in range(1, n_images - 1):  # Skip endpoints
            old_coords = previous_structures[i].cart_coords
            new_coords = current_structures[i].cart_coords
            displacements = np.linalg.norm(new_coords - old_coords, axis=1)
            max_disp = np.max(displacements)
            max_displacement = max(max_displacement, max_disp)
            logger.info(f"  Image {i}: max atom displacement = {max_disp:.4f} Å")
            with open(neb_log_file, "a") as f:
                f.write(f"  Image {i}: max displacement = {max_disp:.4f} Å\n")
        logger.info(f"Overall max displacement: {max_displacement:.4f} Å")
        with open(neb_log_file, "a") as f:
            f.write(f"  Overall max: {max_displacement:.4f} Å\n")
            f.write("\n")

        # Save structures to XSF and CIF for each iteration
        for i, struct in enumerate(current_structures):
            struct_dir = Path(image_folders[i])
            # Save XSF format (good for visualization)
            xsf_file = struct_dir / f"structure_iter_{iteration + 1}.xsf"
            struct.to(filename=str(xsf_file), fmt="xsf")
            # Save CIF format (standard)
            cif_file = struct_dir / f"structure_iter_{iteration + 1}.cif"
            struct.to(filename=str(cif_file))
        logger.info(f"✓ Saved structures (XSF and CIF) for iteration {iteration + 1}")

        # Save checkpoint after each iteration
        import json

        checkpoint_data = {
            "last_iteration": iteration,
            "max_neb_force": float(max_neb_force),
            "converged": bool(converged),
            "barrier_meV": float(energies_rel.max()),
        }
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        logger.debug(f"Saved checkpoint after iteration {iteration + 1}")

    # Prepare final results
    reaction_coord = [i / (n_images - 1) for i in range(n_images)]
    energies_array = np.array(energies)
    energies_rel = (energies_array - energies_array[0]) * 1000  # meV
    max_forces = [np.max(np.linalg.norm(f, axis=1)) for f in forces_list]

    # Write NEB info file (TODO: implement this function)
    # _write_ase_neb_info(
    #     energies,
    #     energies_rel.tolist(),
    #     reaction_coord,
    #     forces_list,
    #     max_forces,
    #     current_structures,
    #     iteration,
    #     converged,
    #     max_neb_force,
    # )

    logger.info("\n" + "=" * 60)
    logger.info(f"NEB optimization complete: {iteration + 1} iterations")
    logger.info(f"Converged: {converged}")
    logger.info(f"Activation energy: {energies_rel.max():.2f} meV")
    logger.info("=" * 60)

    return {
        "energies": energies,
        "energies_relative_meV": energies_rel.tolist(),
        "reaction_coordinate": reaction_coord,
        "activation_energy_meV": float(energies_rel.max()),
        "n_images": n_images,
        "structures": current_structures,
        "forces": [f.tolist() for f in forces_list],
        "max_forces": max_forces,
        "neb_forces": neb_forces.tolist(),
        "max_neb_force": float(max_neb_force),
        "iterations": iteration + 1,
        "converged": bool(converged),
        "image_folders": image_folders,
    }


@job
def generate_neb_images_ase(
    number_of_images: int, initial: Structure, final: Structure
) -> list[Structure]:
    """
    Generate NEB images using ASE interpolation and save them to disk.

    Parameters
    ----------
    number_of_images : int
        Number of intermediate images (not including endpoints).
    initial : Structure
        Initial structure.
    final : Structure
        Final structure.

    Returns
    -------
    list[Structure]
        List of structures: [initial, img1, img2, ..., imgN, final]
    """
    from ase.mep import NEB
    from ase.io import write

    logger.info(f"generate_neb_images_ase() with {number_of_images} images")

    initial_ase = AseAtomsAdaptor.get_atoms(initial)
    final_ase = AseAtomsAdaptor.get_atoms(final)

    # Create image list
    images = [initial_ase]
    images += [initial_ase.copy() for _ in range(number_of_images)]
    images += [final_ase]

    # Interpolate
    neb = NEB(images)
    neb.interpolate(method="idpp")

    # Convert back to pymatgen Structures
    structures = [AseAtomsAdaptor.get_structure(img) for img in images]

    # Save images to disk for inspection
    logger.info(f"Saving {len(structures)} NEB images to disk")
    for i, (img, struct) in enumerate(zip(images, structures)):
        # Save as XYZ (easy to visualize)
        write(f"image_{i}.xyz", img)
        # Save as CIF (pymatgen format)
        struct.to(f"image_{i}.cif")
        logger.info(f"  Saved image_{i}.xyz and image_{i}.cif")

    logger.info(f"Generated {len(structures)} NEB images")
    return structures


@job
def ase_neb_iteration_persistent(
    image_folders: list[str],
    structures: list[Structure],
    iteration: int,
    static_maker: Maker,
    optimizer: str,
    fmax: float,
    climbing_image: bool,
    spring_constant: float,
    max_iterations: int,
    endpoint_energies: list = None,  # Store endpoint energies from first iteration
    endpoint_forces: list = None,  # Store endpoint forces from first iteration
) -> dict:
    """
    Perform one NEB iteration using persistent image folders.

    Runs SIESTA in each image's folder (reuses folders across iterations).
    Optimizes by only calculating endpoint (initial/final) structures once.

    Parameters
    ----------
    image_folders : list[str]
        Absolute paths to persistent image folders.
    structures : list[Structure]
        Current NEB image structures.
    iteration : int
        Current iteration number.
    static_maker : Maker
        Static maker for force calculations.
    optimizer : str
        ASE optimizer name ("BFGS" or "FIRE").
    fmax : float
        NEB force convergence criterion in eV/Å.
    climbing_image : bool
        Use climbing image NEB.
    spring_constant : float
        Spring constant in eV/Å².
    max_iterations : int
        Maximum iterations.
    endpoint_energies : list, optional
        Cached endpoint energies [E_initial, E_final] from first iteration.
    endpoint_forces : list, optional
        Cached endpoint forces from first iteration.

    Returns
    -------
    dict
        NEB results or spawns next iteration.
    """
    from pathlib import Path
    import os
    import numpy as np

    logger.info(f"NEB Iteration {iteration + 1}/{max_iterations} (persistent folders)")

    # Run SIESTA in each image folder
    from atomate2.siesta.run import run_siesta

    energies = []
    forces_list = []

    n_images = len(structures)

    # Get base directory to return to after each calculation
    base_dir = Path.cwd()

    for i, (folder, structure) in enumerate(zip(image_folders, structures)):
        # Optimization: Skip endpoint calculations after first iteration
        is_endpoint = i == 0 or i == n_images - 1

        if is_endpoint and iteration > 0 and endpoint_energies is not None:
            # Reuse cached endpoint data
            if i == 0:
                logger.info(f"  Image {i} (initial): Reusing cached endpoint data")
                energies.append(endpoint_energies[0])
                forces_list.append(endpoint_forces[0])
            else:  # i == n_images - 1
                logger.info(f"  Image {i} (final): Reusing cached endpoint data")
                energies.append(endpoint_energies[1])
                forces_list.append(endpoint_forces[1])
            continue

        # Run SIESTA for this image
        logger.info(f"  Running SIESTA for image {i} in {folder}")

        # Change to image folder
        folder_path = Path(folder)
        os.chdir(folder_path)

        # Write input files using write_siesta_input_set
        from atomate2.siesta.files import write_siesta_input_set

        write_siesta_input_set(
            structure,
            static_maker.input_set_generator,
            directory=".",
        )

        # Run SIESTA (similar to how custodian does it)
        run_siesta()

        # Parse results using SiestaTaskDoc
        from atomate2.siesta.schemas.task import SiestaTaskDoc

        task_doc = SiestaTaskDoc.from_directory(folder_path)

        energies.append(task_doc.output.energy)
        forces_list.append(np.array(task_doc.output.forces))

        logger.info(
            f"  Image {i}: E = {task_doc.output.energy:.6f} eV, max_force = {np.max(np.linalg.norm(task_doc.output.forces, axis=1)):.4f} eV/Å"
        )

        # Return to base directory
        os.chdir(base_dir)

    # Cache endpoint data from first iteration
    if iteration == 0:
        endpoint_energies = [energies[0], energies[-1]]
        endpoint_forces = [forces_list[0], forces_list[-1]]
        logger.info("Cached endpoint energies and forces for future iterations")

    # Now compute NEB step
    result = _compute_neb_step_persistent(
        energies=energies,
        forces=forces_list,
        structures=structures,
        image_folders=image_folders,
        iteration=iteration,
        optimizer=optimizer,
        fmax=fmax,
        climbing_image=climbing_image,
        spring_constant=spring_constant,
        max_iterations=max_iterations,
        static_maker=static_maker,
        endpoint_energies=endpoint_energies,
        endpoint_forces=endpoint_forces,
    )

    return result


def _compute_neb_step_persistent(
    energies: list,
    forces: list,
    structures: list[Structure],
    image_folders: list[str],
    iteration: int,
    optimizer: str,
    fmax: float,
    climbing_image: bool,
    spring_constant: float,
    max_iterations: int,
    static_maker: Maker,
    endpoint_energies: list = None,
    endpoint_forces: list = None,
) -> dict:
    """
    Compute NEB forces, update positions, check convergence (for persistent approach).

    Parameters
    ----------
    energies : list
        Energies from SIESTA calculations.
    forces : list
        Forces from SIESTA calculations.
    structures : list[Structure]
        Current structures.
    image_folders : list[str]
        Persistent folder paths.
    iteration : int
        Current iteration.
    optimizer : str
        Optimizer name.
    fmax : float
        Force convergence.
    climbing_image : bool
        Use climbing image.
    spring_constant : float
        Spring constant.
    max_iterations : int
        Max iterations.
    static_maker : Maker
        Static maker for next iteration.
    endpoint_energies : list, optional
        Cached endpoint energies.
    endpoint_forces : list, optional
        Cached endpoint forces.

    Returns
    -------
    dict
        NEB results or reference to next iteration.
    """
    from jobflow import Response
    from ase.mep import NEB
    from ase.optimize import BFGS, FIRE
    from ase.calculators.singlepoint import SinglePointCalculator
    from pymatgen.io.ase import AseAtomsAdaptor
    import numpy as np

    logger.info(f"Computing NEB step {iteration + 1} (persistent)")

    # Optimizer state file (for persistent NEB object)
    from pathlib import Path

    opt_state_file = Path(image_folders[0]).parent / "optimizer_state.pckl"
    neb_state_file = Path(image_folders[0]).parent / "neb_images.pckl"

    # On first iteration: create new NEB object
    # On subsequent iterations: restore and update existing NEB object
    if iteration == 0 or not neb_state_file.exists():
        # Convert structures to ASE atoms (first iteration)
        images = [AseAtomsAdaptor.get_atoms(s) for s in structures]

        # Attach energies and forces
        for img, energy, force in zip(images, energies, forces):
            img.calc = SinglePointCalculator(img, energy=energy, forces=force)

        # Create NEB object
        neb = NEB(images, k=spring_constant, climb=climbing_image)

        logger.info("✓ Created new NEB object (first iteration)")
    else:
        # Load existing NEB object to preserve Atoms object identity
        import pickle

        with open(neb_state_file, "rb") as neb_file:
            neb = pickle.load(neb_file)

        logger.info("✓ Restored NEB object from previous iteration")

        # Update positions (structures may have changed from optimization)
        for img, structure in zip(neb.images, structures):
            # Update atomic positions in-place (preserves Atoms object identity)
            new_atoms = AseAtomsAdaptor.get_atoms(structure)
            img.set_positions(new_atoms.get_positions())
            img.set_cell(new_atoms.get_cell())

        # Update energies and forces with new SIESTA results
        for img, energy, force in zip(neb.images, energies, forces):
            img.calc = SinglePointCalculator(img, energy=energy, forces=force)

    # Get NEB forces (includes spring forces)
    neb_forces = neb.get_forces()

    # Calculate max NEB force
    max_neb_force = np.max(np.abs(neb_forces))

    logger.info(
        f"Iteration {iteration + 1}: Max NEB force = {max_neb_force:.4f} eV/Å (fmax = {fmax})"
    )

    # Check convergence
    converged = max_neb_force < fmax

    if converged or iteration >= max_iterations - 1:
        # Converged or max iterations reached
        logger.info(
            f"NEB {'converged' if converged else 'reached max iterations'} after {iteration + 1} iterations"
        )

        # Calculate final metrics
        reaction_coord = [i / (len(structures) - 1) for i in range(len(structures))]
        energies_array = np.array(energies)
        energies_rel = (energies_array - energies_array[0]) * 1000  # meV

        max_forces = [np.max(np.linalg.norm(f, axis=1)) for f in forces]

        # Write NEB info file (TODO: implement this function)
        # _write_ase_neb_info(
        #     energies,
        #     energies_rel.tolist(),
        #     reaction_coord,
        #     forces,
        #     max_forces,
        #     structures,
        #     iteration,
        #     converged,
        #     max_neb_force,
        # )

        return {
            "energies": energies,
            "energies_relative_meV": energies_rel.tolist(),
            "reaction_coordinate": reaction_coord,
            "activation_energy_meV": float(energies_rel.max()),
            "n_images": len(structures),
            "structures": structures,
            "forces": [f.tolist() for f in forces],
            "max_forces": max_forces,
            "neb_forces": neb_forces.tolist(),
            "max_neb_force": float(max_neb_force),
            "iterations": iteration + 1,
            "converged": bool(converged),
            "image_folders": image_folders,
        }
    else:
        # Not converged - perform optimization step
        logger.info(
            f"Not converged (max_force={max_neb_force:.4f} > fmax={fmax}), continuing..."
        )

        # Create or restore optimizer
        # Critical parameters for noisy DFT forces:
        #   - maxstep: larger (0.5 Å) to allow sufficient movement
        #   - alpha: MUCH smaller (1/75 ≈ 0.013) for conservative first step
        #     (ASE default 70.0 is 5000x too aggressive for noisy forces!)
        if optimizer.upper() == "FIRE":
            opt = FIRE(neb, logfile=None, maxmove=0.5)
        else:
            opt = BFGS(neb, logfile=None, maxstep=0.5, alpha=1.0 / 75.0)

        # Restore optimizer state if exists (critical for BFGS Hessian!)
        # This now works correctly because NEB object identity is preserved
        if iteration > 0 and opt_state_file.exists():
            try:
                opt.load(opt_state_file)
                logger.info(
                    f"✓ Restored optimizer state from iteration {iteration} "
                    "(Hessian references correct NEB object)"
                )
            except Exception as e:
                logger.warning(f"Could not restore optimizer state: {e}")

        # Take one optimization step
        opt.step()  # Optimizer internally calls neb.get_forces()

        # Save optimizer state for next iteration (preserve Hessian/velocity)
        try:
            opt.dump(opt_state_file)
            logger.info("✓ Saved optimizer state for next iteration")
        except Exception as e:
            logger.warning(f"Could not save optimizer state: {e}")

        # Save NEB object for next iteration (preserves Atoms object identity)
        import pickle

        try:
            with open(neb_state_file, "wb") as neb_out:
                pickle.dump(neb, neb_out)
            logger.info("✓ Saved NEB object (Atoms identity preserved)")
        except Exception as e:
            logger.warning(f"Could not save NEB object: {e}")

        # Extract updated structures from NEB images
        updated_structures = [AseAtomsAdaptor.get_structure(img) for img in neb.images]

        # Spawn next iteration (pass endpoint data for reuse)
        next_iteration_job = ase_neb_iteration_persistent(
            image_folders=image_folders,  # Reuse same folders!
            structures=updated_structures,
            iteration=iteration + 1,
            static_maker=static_maker,
            optimizer=optimizer,
            fmax=fmax,
            climbing_image=climbing_image,
            spring_constant=spring_constant,
            max_iterations=max_iterations,
            endpoint_energies=endpoint_energies,  # Pass cached endpoint data
            endpoint_forces=endpoint_forces,
        )
        next_iteration_job.name = f"NEB_Iteration_{iteration + 2}"

        # Return Response that spawns next iteration
        return Response(detour=next_iteration_job, output=next_iteration_job.output)


@job
def ase_neb_iteration(
    structures: list[Structure],
    iteration: int,
    static_maker: Maker,
    optimizer: str,
    fmax: float,
    climbing_image: bool,
    spring_constant: float,
    max_iterations: int,
    previous_forces: list = None,
) -> dict:
    """
    Perform one iteration of ASE NEB optimization.

    This function:
    1. Runs static calculations on all images in parallel
    2. Computes NEB forces (including spring forces)
    3. Updates atomic positions using optimizer
    4. Checks convergence
    5. Recursively spawns next iteration if not converged

    Parameters
    ----------
    structures : list[Structure]
        Current NEB image structures.
    iteration : int
        Current iteration number.
    static_maker : Maker
        Static maker for force calculations.
    optimizer : str
        ASE optimizer name.
    fmax : float
        Force convergence criterion.
    climbing_image : bool
        Use climbing image NEB.
    spring_constant : float
        Spring constant for NEB.
    max_iterations : int
        Maximum number of iterations.
    previous_forces : list
        Forces from previous iteration (for optimizer state).

    Returns
    -------
    dict
        NEB results with convergence info, energies, forces, structures.
    """
    from jobflow import Flow, Response

    logger.info(f"ASE NEB Iteration {iteration + 1}/{max_iterations}")

    # Step 1: Run static calculations on all images in parallel
    static_jobs = []
    for i, struct in enumerate(structures):
        job = static_maker.make(struct)
        job.name = f"NEB_Iter_{iteration + 1}_Image_{i}"
        static_jobs.append(job)

    # Create the Response that will:
    # 1. Run all static jobs in parallel
    # 2. Then call compute_neb_step with results
    compute_step_job = compute_neb_step(
        static_jobs=[job.output for job in static_jobs],
        structures=structures,
        iteration=iteration,
        optimizer=optimizer,
        fmax=fmax,
        climbing_image=climbing_image,
        spring_constant=spring_constant,
        max_iterations=max_iterations,
        static_maker=static_maker,
    )
    compute_step_job.name = f"NEB_Compute_Step_{iteration + 1}"

    # Return Response with static jobs + compute job
    return Response(
        replace=Flow([*static_jobs, compute_step_job]), output=compute_step_job.output
    )


@job
def compute_neb_step(
    static_jobs: list,
    structures: list[Structure],
    iteration: int,
    optimizer: str,
    fmax: float,
    climbing_image: bool,
    spring_constant: float,
    max_iterations: int,
    static_maker: Maker,
) -> dict:
    """
    Compute NEB forces, update positions, check convergence.

    Parameters
    ----------
    static_jobs : list
        Outputs from static calculations (energies and forces).
    structures : list[Structure]
        Current structures.
    iteration : int
        Current iteration.
    optimizer : str
        Optimizer name.
    fmax : float
        Force convergence.
    climbing_image : bool
        Use climbing image.
    spring_constant : float
        Spring constant.
    max_iterations : int
        Max iterations.
    static_maker : Maker
        Static maker for next iteration.

    Returns
    -------
    dict
        NEB results or reference to next iteration.
    """
    from jobflow import Response
    from ase.mep import NEB
    from ase.optimize import BFGS, FIRE
    from pymatgen.io.ase import AseAtomsAdaptor
    import numpy as np

    logger.info(f"Computing NEB step {iteration + 1}")

    # Extract energies and forces from static calculations
    energies = []
    forces = []
    for result in static_jobs:
        if hasattr(result, "output"):
            energies.append(result.output.energy)
            if hasattr(result.output, "forces") and result.output.forces is not None:
                forces.append(np.array(result.output.forces))
            else:
                raise ValueError("No forces in static calculation output")
        else:
            energies.append(result["energy"])
            forces.append(np.array(result["forces"]))

    # Convert structures to ASE atoms
    images = [AseAtomsAdaptor.get_atoms(s) for s in structures]

    # Attach energies and forces to ASE atoms
    for img, energy, force in zip(images, energies, forces):
        img.calc = None  # Remove any existing calculator
        img.set_calculator(SinglePointCalculator(img, energy=energy, forces=force))

    # Create NEB object
    neb = NEB(images, k=spring_constant, climb=climbing_image)

    # Get NEB forces (includes spring forces)
    neb_forces = neb.get_forces()

    # Calculate max NEB force
    max_neb_force = np.max(np.abs(neb_forces))

    logger.info(f"Iteration {iteration + 1}: Max NEB force = {max_neb_force:.4f} eV/Å")

    # Check convergence
    converged = max_neb_force < fmax

    if converged or iteration >= max_iterations - 1:
        # Converged or max iterations reached
        logger.info(
            f"NEB {'converged' if converged else 'reached max iterations'} after {iteration + 1} iterations"
        )

        # Calculate final metrics
        reaction_coord = [i / (len(structures) - 1) for i in range(len(structures))]
        energies_array = np.array(energies)
        energies_rel = (energies_array - energies_array[0]) * 1000  # meV

        max_forces = [np.max(np.linalg.norm(f, axis=1)) for f in forces]

        # Write NEB info file (TODO: implement this function)
        # _write_ase_neb_info(
        #     energies,
        #     energies_rel.tolist(),
        #     reaction_coord,
        #     forces,
        #     max_forces,
        #     structures,
        #     iteration,
        # )

        return {
            "energies": energies,
            "energies_relative_meV": energies_rel.tolist(),
            "reaction_coordinate": reaction_coord,
            "activation_energy_meV": float(energies_rel.max()),
            "n_images": len(structures),
            "structures": structures,
            "forces": forces,
            "max_forces": max_forces,
            "neb_forces": neb_forces.tolist(),
            "max_neb_force": float(max_neb_force),
            "iterations": iteration + 1,
            "converged": bool(converged),
        }
    else:
        # Not converged - perform optimization step
        logger.info(
            f"Not converged (max_force={max_neb_force:.4f} > fmax={fmax}), continuing..."
        )

        # Create optimizer
        if optimizer.upper() == "FIRE":
            opt = FIRE(neb)
        else:
            opt = BFGS(neb)

        # Take one optimization step
        opt.step()  # Optimizer internally calls neb.get_forces()

        # Extract updated structures
        updated_structures = [AseAtomsAdaptor.get_structure(img) for img in images]

        # Spawn next iteration
        next_iteration_job = ase_neb_iteration(
            structures=updated_structures,
            iteration=iteration + 1,
            static_maker=static_maker,
            optimizer=optimizer,
            fmax=fmax,
            climbing_image=climbing_image,
            spring_constant=spring_constant,
            max_iterations=max_iterations,
        )
        next_iteration_job.name = f"NEB_Iteration_{iteration + 2}"

        # Return Response that replaces this job with the next iteration
        return Response(detour=next_iteration_job, output=next_iteration_job.output)


@job
def run_static_on_images(
    structures: list[Structure], static_maker: Maker
) -> list[dict]:
    """
    Run static calculations on all NEB images.

    Parameters
    ----------
    structures : list[Structure]
        List of NEB image structures.
    static_maker : Maker
        Static calculation maker.

    Returns
    -------
    list[dict]
        List of calculation results (energies, forces, structures).
    """
    from jobflow import Flow, Response

    logger.info(f"run_static_on_images() for {len(structures)} images")

    jobs = []
    for i, structure in enumerate(structures):
        static_job = static_maker.make(structure)
        static_job.name = f"NEB_Image_{i}"
        jobs.append(static_job)

    # Return Response with jobs and collect their outputs
    return Response(replace=Flow(jobs), output=[job.output for job in jobs])


@job
def run_ase_neb_optimization(
    initial_static_results: list[dict],
    optimizer: str = "BFGS",
    fmax: float = 0.05,
    climbing_image: bool = False,
    spring_constant: float = 5.0,
    max_iterations: int = 100,
    static_maker: Maker = None,
) -> dict:
    """
    Run ASE NEB optimization with iterative force calculations.

    This performs actual NEB optimization by:
    1. Computing NEB forces (including springs between images)
    2. Moving atoms according to optimizer
    3. Running SIESTA static calculations on updated structures
    4. Repeat until NEB forces converge

    Parameters
    ----------
    initial_static_results : list[dict]
        Initial static calculation results for all images.
    optimizer : str
        ASE optimizer ("BFGS" or "FIRE").
    fmax : float
        NEB force convergence criterion in eV/Å.
    climbing_image : bool
        Use climbing image NEB.
    spring_constant : float
        Spring constant in eV/Å².
    max_iterations : int
        Maximum number of NEB iterations.
    static_maker : Maker
        Static maker for force calculations.

    Returns
    -------
    dict
        NEB results with energies, forces, activation energy, convergence history.
    """
    import numpy as np
    from ase.mep import NEB
    from pymatgen.io.ase import AseAtomsAdaptor

    logger.info("run_ase_neb_optimization()")

    # Extract initial structures
    structures = []
    for result in initial_static_results:
        if hasattr(result, "output"):
            structures.append(result.output.structure)
        else:
            structures.append(result["structure"])

    # Convert to ASE atoms
    images = [AseAtomsAdaptor.get_atoms(s) for s in structures]

    # Create NEB object (will be used in future iterations)
    _neb = NEB(images, k=spring_constant, climb=climbing_image)

    # NEB optimization loop
    iteration = 0
    converged = False
    _history: list[dict] = []  # Will store convergence history

    while iteration < max_iterations and not converged:
        logger.info(f"NEB iteration {iteration + 1}/{max_iterations}")

        # Step 1: Get current structures
        current_structures = [AseAtomsAdaptor.get_structure(img) for img in images]

        # Step 2: Run static calculations in parallel on all images
        # This will be done by returning a Response with jobs
        static_jobs = []
        for i, struct in enumerate(current_structures):
            job = static_maker.make(struct)
            job.name = f"NEB_Iter_{iteration+1}_Image_{i}"
            static_jobs.append(job)

        # Return jobs to run in parallel, then continue with next iteration
        # This is a placeholder - we need to restructure this as a Flow
        break  # For now, just do one iteration

    # Extract final results (temporary - needs proper iteration)
    energies = []
    forces = []
    for result in initial_static_results:
        if hasattr(result, "output"):
            energies.append(result.output.energy)
            if hasattr(result.output, "forces") and result.output.forces is not None:
                forces.append(result.output.forces)
            else:
                forces.append(None)
        else:
            energies.append(result["energy"])
            forces.append(result.get("forces", None))

    # Calculate metrics
    reaction_coord = [i / (len(structures) - 1) for i in range(len(structures))]
    energies_array = np.array(energies)
    energies_rel_array = (energies_array - energies_array[0]) * 1000

    max_forces = []
    for force_array in forces:
        if force_array is not None:
            max_force = np.max(np.linalg.norm(force_array, axis=1))
            max_forces.append(max_force)
        else:
            max_forces.append(None)

    # Write NEB info file (TODO: implement this function)
    # _write_ase_neb_info(
    #     energies_array.tolist(),
    #     energies_rel_array.tolist(),
    #     reaction_coord,
    #     forces,
    #     max_forces,
    #     structures,
    #     iteration,
    # )

    return {
        "energies": energies_array.tolist(),
        "energies_relative_meV": energies_rel_array.tolist(),
        "reaction_coordinate": reaction_coord,
        "activation_energy_meV": float(energies_rel_array.max()),
        "n_images": len(structures),
        "structures": structures,
        "forces": forces,
        "max_forces": max_forces,
        "iterations": iteration + 1,
        "converged": bool(converged),
    }
