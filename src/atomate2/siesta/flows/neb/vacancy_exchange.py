"""NEB workflow for vacancy exchange calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Maker

from atomate2.siesta.flows.base import BaseSiestaFlowMaker
from atomate2.siesta.flows.neb.common import generate_neb_band
from atomate2.siesta.flows.neb.plotting import plot_neb_results
from atomate2.siesta.jobs.core import LuaMaker, RelaxMaker

if TYPE_CHECKING:
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


@dataclass
class NebVacancyExchangeFlowMaker(BaseSiestaFlowMaker):
    """
    Jobflow maker to run NEB calculations for vacancy exchange.

    This maker is specifically designed for vacancy exchange or atom swap
    scenarios where you want to calculate the energy barrier for swapping
    two atoms (A and B) in a structure.

    Steps:
    1. Generate initial and final images by swapping atoms at indices A and B.
    2. Relax the initial and final images.
    3. Regenerate NEB images from relaxed structures using ASE interpolation.
    4. Run the NEB calculation using Lua scripting (neb.lua).

    Parameters
    ----------
    name : str
        Name of the flow produced by this maker.
    relax_maker : Maker
        Job maker for relaxing structures. Default is RelaxMaker.fixed_cell_relaxation().
    neb_maker : Maker
        Job maker for running NEB calculation. Default is LuaMaker.neb().
    number_of_images : int
        Number of intermediate NEB images to generate, default is 5.
    A : int
        Index of the first atom to swap.
    B : int
        Index of the second atom to swap.

    Example
    -------
    >>> from atomate2.siesta.flows.neb import NebVacancyExchangeFlowMaker
    >>> from pymatgen.core import Structure
    >>> structure = Structure.from_file("supercell.cif")
    >>> maker = NebVacancyExchangeFlowMaker(A=10, B=15, number_of_images=7)
    >>> flow = maker.make(structure)
    """

    name: str = "NEB Vacancy Exchange Workflow"
    relax_maker: Maker | None = field(
        default_factory=RelaxMaker.fixed_cell_relaxation
    )  # Maker for structure relaxation
    neb_maker: Maker | None = field(default_factory=LuaMaker.neb)
    number_of_images: int = 5
    A: int = None
    B: int = None
    interpolation_method: str = "idpp"

    def generate_neb_images(
        self,
        structure: Structure,
        A: int,
        B: int,  # noqa: N803
    ) -> list[Structure]:
        """
        Generate initial and final NEB images by swapping atoms at indices A and B.

        Parameters
        ----------
        structure : Structure
            The input structure.
        A : int
            Index of the first atom.
        B : int
            Index of the second atom.

        Returns
        -------
        List[Structure]
            List containing [initial_structure, final_structure] with atoms swapped.
        """
        logger.info("NebVacancyExchangeFlowMaker.generate_neb_images()")
        initial_pymatgen = structure.copy()
        final_pymatgen = structure.copy()

        # Remove the atoms at indices A and B (for vacancy exchange)
        initial_pymatgen.remove_sites([A, B])
        final_pymatgen.remove_sites([A, B])

        # Append atoms to their new positions (A to B, B to A)
        initial_pymatgen.append(
            species=structure[A].specie, coords=structure[A].frac_coords
        )
        final_pymatgen.append(
            species=structure[B].specie, coords=structure[B].frac_coords
        )

        return [initial_pymatgen, final_pymatgen]

    def make(self, structure: Structure) -> Flow:
        """
        Create a NEB vacancy exchange workflow.

        Parameters
        ----------
        structure : Structure
            The input structure containing the atoms to be swapped.

        Returns
        -------
        Flow
            A jobflow Flow containing the NEB vacancy exchange workflow.
        """
        from atomate2.siesta.utils.common import print_docstring_in_box

        print_docstring_in_box(self.__doc__, title=self.__class__.__name__)

        logger.info("NebVacancyExchangeFlowMaker.make()")
        jobs = []

        # Step 1: Generate initial and final images
        # neb_images = self.generate_neb_images(structure, A, B)
        neb_images = self.generate_neb_images(structure, self.A, self.B)

        # Step 2: Relax the initial and final images
        initial_relax = self.relax_maker.make(structure=neb_images[0])
        final_relax = self.relax_maker.make(structure=neb_images[1])

        initial_relax.name = f"{self.name}_Initial_Relaxation"
        final_relax.name = f"{self.name}_Final_Relaxation"

        jobs.append(initial_relax)
        jobs.append(final_relax)

        # Step 3: Generate NEB images from the relaxed structures
        relaxed_initial = jobs[0].output.structure  # initial_relax.output.structure
        relaxed_final = jobs[1].output.structure  # final_relax.output.structure
        # print(f"{jobs[0].output['dir_name']=}")
        logger.debug(f"Relaxed initial structure: {relaxed_initial}")
        logger.debug(f"Relaxed final structure: {relaxed_final}")

        neb_image_job = generate_neb_band(
            self.number_of_images,
            relaxed_initial,
            relaxed_final,
            self.interpolation_method,
        )
        neb_image_job.name = f"{self.name}_NEB_Image_Generation"
        jobs.append(neb_image_job)

        # Step 4: NEB calculation (this part would depend on specific NEB jobs you're using)
        # Assuming you have a NEB maker or job to run the calculation
        # neb_job = self.neb_maker.make(images=neb_band)
        # neb_job.name = f"{self.name} NEB Calculation"
        # Step 4: NEB calculation using LuaMaker
        # neb_job = self.run_neb_calculation(relaxed_initial,neb_image_jobs.output)
        # neb_job = self.neb_maker.make(relaxed_initial) # Pass the list of NEB images to be handled)
        # jobs.append(neb_job)

        # images = neb_image_jobs.output["images"]  # Access the images from the job's output

        neb_maker_instance = self.neb_maker  # Create an instance of LuaMaker
        # neb_maker_instance.write_additional_data = {"siesta.1.txt": neb_image_jobs.images[0].positions}  # Pass NEB images
        # neb_maker_instance.write_additional_data = {f"siesta.{i+1}.xyz": file for i, file in enumerate(neb_image_jobs.images[0])}
        # neb_maker_instance.write_additional_data = {f"siesta.{i+1}.xyz": image for i, image in enumerate(images)}
        # print("Job output:")
        image_dir = neb_image_job.output  # neb_image_job.output.job_dir
        logger.debug(f"NEB image directory: {image_dir}")
        # print (f"{neb_image_job.output.output_dir["dir_name"]=}")
        # print(f"{neb_image_job.output.output. .output.job_dir=}")
        # write(neb_image_job[0])
        # neb_maker_instance.write_additional_data = {f"siesta.{i+1}.xyz": file for i, file in enumerate(neb_image_job.output.neb_image_files)}
        neb_job = neb_maker_instance.make(
            relaxed_initial, extra_dir=image_dir
        )  # ,prev_dir=neb_image_job.output.dir_name) #.output.dir_name)
        jobs.append(neb_job)

        # Step 5: Plot NEB results
        plot_job = plot_neb_results(neb_job.output.dir_name)
        plot_job.name = f"{self.name}_NEB_Plotting"
        jobs.append(plot_job)

        # Return the flow
        return Flow(jobs=jobs, name=self.name)
