"""NEB workflow using direct initial and final structures."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jobflow import Flow, Maker

from atomate2.siesta.flows.base import BaseSiestaFlowMaker
from atomate2.siesta.flows.neb.common import generate_neb_band
from atomate2.siesta.flows.neb.plotting import plot_neb_results
from atomate2.siesta.jobs.core import LuaMaker, RelaxMaker

if TYPE_CHECKING:
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


@dataclass
class NebDirectFlowMaker(BaseSiestaFlowMaker):
    """
    Jobflow maker to run NEB calculations from two structures directly.

    This maker accepts two structures (initial and final) and generates
    intermediate images using ASE's NEB interpolation. This is useful when
    you already know the initial and final structures and want to calculate
    the minimum energy path between them.

    Steps:
    1. (Optional) Relax the initial and/or final structures.
    2. Generate NEB images from the (relaxed) structures using ASE interpolation.
    3. Run the NEB calculation using Lua scripting (neb.lua).

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
    neb_maker : Maker | None
        Job maker for running NEB calculation. Default is LuaMaker.neb().
    number_of_images : int
        Number of intermediate NEB images to generate (default is 5).
    interpolation_method : str
        ASE NEB interpolation method. Options: "idpp" (image-dependent pair
        potential, recommended), "linear" (simple linear interpolation).
        Default is "idpp".

    Example
    -------
    >>> from atomate2.siesta.flows.neb import NebDirectFlowMaker
    >>> from pymatgen.core import Structure
    >>> initial = Structure.from_file("initial.cif")
    >>> final = Structure.from_file("final.cif")
    >>>
    >>> # Without endpoint relaxation
    >>> maker = NebDirectFlowMaker(number_of_images=7)
    >>> flow = maker.make(initial_structure=initial, final_structure=final)
    >>>
    >>> # Relax both endpoints
    >>> maker = NebDirectFlowMaker(number_of_images=7, relax_endpoints=True)
    >>> flow = maker.make(initial_structure=initial, final_structure=final)
    >>>
    >>> # Relax only initial structure
    >>> maker = NebDirectFlowMaker(number_of_images=7, relax_endpoints="initial")
    >>> flow = maker.make(initial_structure=initial, final_structure=final)
    >>>
    >>> # Relax only final structure
    >>> maker = NebDirectFlowMaker(number_of_images=7, relax_endpoints="final")
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
    >>> maker = NebDirectFlowMaker(
    ...     number_of_images=7,
    ...     relax_endpoints=True,
    ...     relax_initial_maker=initial_relax,
    ...     relax_final_maker=final_relax,
    ... )
    >>> flow = maker.make(initial_structure=initial, final_structure=final)
    """

    name: str = "NEB Direct Workflow"
    relax_endpoints: bool | str = False
    relax_maker: Maker | None = field(default_factory=RelaxMaker.fixed_cell_relaxation)
    relax_initial_maker: Maker | None = None
    relax_final_maker: Maker | None = None
    neb_maker: Maker | None = field(default_factory=LuaMaker.neb)
    number_of_images: int = 5
    interpolation_method: str = "idpp"

    def make(self, initial_structure: Structure, final_structure: Structure) -> Flow:
        """
        Create a NEB workflow from two structures.

        Parameters
        ----------
        initial_structure : Structure
            The initial structure (starting point).
        final_structure : Structure
            The final structure (end point).

        Returns
        -------
        Flow
            A jobflow Flow containing the NEB workflow.
        """
        from atomate2.siesta.utils.common import print_docstring_in_box

        print_docstring_in_box(self.__doc__, title=self.__class__.__name__)

        logger.info("NebDirectFlowMaker.make()")
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

        # Step 1: Relax endpoints as requested
        relaxed_initial: Any = initial_structure
        relaxed_final: Any = final_structure

        if relax_initial:
            # Use relax_initial_maker if specified, otherwise fall back to relax_maker
            maker_to_use = self.relax_initial_maker or self.relax_maker
            initial_relax = maker_to_use.make(structure=initial_structure)
            initial_relax.name = f"{self.name}_Initial_Relaxation"
            jobs.append(initial_relax)
            relaxed_initial = initial_relax.output.structure

        if relax_final:
            # Use relax_final_maker if specified, otherwise fall back to relax_maker
            maker_to_use = self.relax_final_maker or self.relax_maker
            final_relax = maker_to_use.make(structure=final_structure)
            final_relax.name = f"{self.name}_Final_Relaxation"
            jobs.append(final_relax)
            relaxed_final = final_relax.output.structure

        # Step 2: Generate NEB images from the structures
        neb_image_job = generate_neb_band(
            self.number_of_images,
            relaxed_initial,
            relaxed_final,
            self.interpolation_method,
        )
        neb_image_job.name = f"{self.name}_NEB_Image_Generation"
        jobs.append(neb_image_job)

        # Step 3: NEB calculation using LuaMaker
        image_dir = neb_image_job.output
        logger.debug(f"NEB image directory: {image_dir}")

        neb_job = self.neb_maker.make(relaxed_initial, extra_dir=image_dir)
        neb_job.name = f"{self.name}_NEB_Calculation"
        jobs.append(neb_job)

        # Step 4: Plot NEB results
        plot_job = plot_neb_results(neb_job.output.dir_name)
        plot_job.name = f"{self.name}_NEB_Plotting"
        jobs.append(plot_job)

        return Flow(jobs=jobs, name=self.name)
