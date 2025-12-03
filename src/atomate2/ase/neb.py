"""Create NEB jobs with ASE."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ase.mep.neb import idpp_interpolate, interpolate
from emmet.core.neb import NebResult
from jobflow import Flow, Maker, Response, job
from pymatgen.core import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from atomate2.ase.jobs import _ASE_DATA_OBJECTS, AseMaker
from atomate2.ase.utils import AseNebInterface
from atomate2.common.jobs.neb import NebInterpolation, _get_images_from_endpoints

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal

    from ase.atoms import Atoms
    from ase.calculators.calculator import Calculator


@dataclass
class AseNebFromImagesMaker(AseMaker):
    """Define scheme for performing ASE NEB calculations."""

    name: str = "ASE NEB maker"
    neb_kwargs: dict = field(default_factory=dict)
    fix_symmetry: bool = False
    symprec: float | None = 1e-2
    steps: int = 500
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    traj_file: str | None = None
    traj_file_fmt: Literal["pmg", "ase", "xdatcar"] = "ase"
    traj_interval: int = 1
    neb_doc_kwargs: dict = field(default_factory=dict)

    def run_ase(
        self,
        images: list[Atoms | Structure | Molecule],
        prev_dir: str | Path | None = None,
    ) -> NebResult:
        """
        Run an ASE NEB job from a list of images.

        Parameters
        ----------
        images: list of pymatgen .Molecule or .Structure
            pymatgen molecule or structure images
        prev_dir : str or Path or None
            A previous calculation directory to copy output files from. Unused, just
            added to match the method signature of other makers.
        """
        return AseNebInterface(
            calculator=self.calculator,
            fix_symmetry=self.fix_symmetry,
            symprec=self.symprec,
        ).run_neb(
            images,
            steps=self.steps,
            traj_file=self.traj_file,
            traj_file_fmt=self.traj_file_fmt,
            interval=self.traj_interval,
            neb_doc_kwargs=self.neb_doc_kwargs,
            neb_kwargs=self.neb_kwargs,
            optimizer_kwargs=self.optimizer_kwargs,
            **self.relax_kwargs,
        )

    @job(data=_ASE_DATA_OBJECTS, schema=NebResult)
    def make(
        self,
        images: list[Structure | Molecule],
        prev_dir: str | Path | None = None,
    ) -> NebResult:
        """
        Run an ASE NEB job from a list of images.

        Parameters
        ----------
        images: list of pymatgen .Molecule or .Structure
            pymatgen molecule or structure images
        prev_dir : str or Path or None
            A previous calculation directory to copy output files from. Unused, just
            added to match the method signature of other makers.
        """
        # Note that images are copied to prevent them from being overwritten
        # by ASE during the NEB run
        return self.run_ase([image.copy() for image in images], prev_dir=prev_dir)


@dataclass
class AseNebFromEndpointsMaker(AseNebFromImagesMaker):
    """Maker to create ASE NEB jobs from two endpoints.

    Optionally relax the two endpoints and return a full NEB hop analysis.
    If a maker to relax the endpoints is not specified, this job
    interpolates the provided endpoints and performs NEB on the
    interpolated images.

    Parameters
    ----------
    endpoint_relax_maker : Maker or None (default)
        Optional maker to initially relax the endpoints.
    """

    endpoint_relax_maker: Maker | None = None

    @job
    def interpolate_endpoints(
        self,
        endpoints: tuple[Structure | Molecule, Structure | Molecule],
        num_images: int,
        interpolation_method: NebInterpolation | str = NebInterpolation.LINEAR,
        **interpolation_kwargs,
    ) -> list[Atoms]:
        """
        Interpolate between two endpoints using ASE's methods, as a job.

        Note that `num_images` specifies the number of intermediate images
        between two endpoints. Thus, specifying `num_images = 5` will return
        the endpoints and 5 intermediate images.

        Parameters
        ----------
        endpoints : tuple[Structure,Structure] or list[Structure]
            A set of two endpoints to interpolate NEB images from.
        num_images : int
            The number of images to include in the interpolation.
        interpolation_method : .NebInterpolation
            The method to use to interpolate between images.
        **interpolation_kwargs
            kwargs to pass to the interpolation function.
        """
        # return interpolate_endpoints_ase(
        #     endpoints, num_images, interpolation_method, **interpolation_kwargs
        # )
        interpolated = _get_images_from_endpoints(
            endpoints,
            num_images,
            interpolation_method=interpolation_method,
            **interpolation_kwargs,
        )
        adaptor = AseAtomsAdaptor()
        return [adaptor.get_atoms(image) for image in interpolated]

    @job(data=_ASE_DATA_OBJECTS)
    def make(
        self,
        endpoints: tuple[Structure | Molecule, Structure | Molecule]
        | list[Structure | Molecule],
        num_images: int,
        prev_dir: str | Path = None,
        interpolation_method: NebInterpolation | str = NebInterpolation.LINEAR,
        **interpolation_kwargs,
    ) -> Flow:
        """
        Make an NEB job from a set of endpoints.

        Parameters
        ----------
        endpoints : tuple[Structure,Structure] or list[Structure]
            A set of two endpoints to interpolate NEB images from.
        num_images : int
            The number of images to include in the interpolation.
        prev_dir : str or Path or None (default)
            A previous directory to copy outputs from.
        interpolation_method : .NebInterpolation
            The method to use to interpolate between images.
        **interpolation_kwargs
            kwargs to pass to the interpolation function.
        """
        if len(endpoints) != 2:
            raise ValueError("Please specify exactly two endpoint structures.")

        endpoint_jobs = []
        if self.endpoint_relax_maker is not None:
            endpoint_jobs += [
                self.endpoint_relax_maker.make(endpoint, prev_dir=prev_dir)
                for endpoint in endpoints
            ]
            for idx in range(2):
                endpoint_jobs[idx].append_name(f" endpoint {idx + 1}")
            endpoints = [relax_job.output.structure for relax_job in endpoint_jobs]

        get_images = self.interpolate_endpoints(
            endpoints,
            num_images,
            interpolation_method=interpolation_method,
            **interpolation_kwargs,
        )

        neb_from_images = super().make(get_images.output)

        flow = Flow(
            [*endpoint_jobs, get_images, neb_from_images],
            output=neb_from_images.output,
        )

        return Response(replace=flow, output=neb_from_images.output)


def interpolate_endpoints_ase(
    endpoints: tuple[Structure | Molecule | Atoms, Structure | Molecule | Atoms],
    num_images: int,
    interpolation_method: NebInterpolation | str = NebInterpolation.LINEAR,
    **interpolation_kwargs,
) -> list[Atoms]:
    """
    Interpolate between two endpoints using ASE's methods.

    Note that `num_images` specifies the number of intermediate images
    between two endpoints. Thus, specifying `num_images = 5` will return
    the endpoints and 5 intermediate images.

    Parameters
    ----------
    endpoints : tuple[Structure,Structure] or list[Structure]
        A set of two endpoints to interpolate NEB images from.
    num_images : int
        The number of images to include in the interpolation.
    interpolation_method : .NebInterpolation
        The method to use to interpolate between images.
    **interpolation_kwargs
        kwargs to pass to the interpolation function.

    Returns
    -------
    list of Atoms : the atoms interpolated between endpoints.
    """
    endpoint_atoms = [
        AseAtomsAdaptor().get_atoms(ions)
        if isinstance(ions, Structure | Molecule)
        else ions.copy()
        for ions in endpoints
    ]
    images = [
        endpoint_atoms[0],
        *[endpoint_atoms[0].copy() for _ in range(num_images)],
        endpoint_atoms[1],
    ]

    interp_method = NebInterpolation(interpolation_method)
    if interp_method == NebInterpolation.LINEAR:
        interpolate(images, **interpolation_kwargs)
    elif interp_method == NebInterpolation.IDPP:
        idpp_interpolate(images, **interpolation_kwargs)
    return images


class EmtNebFromImagesMaker(AseNebFromImagesMaker):
    """EMT NEB from images maker."""

    name: str = "EMT NEB from images maker"

    @property
    def calculator(self) -> Calculator:
        """EMT calculator."""
        from ase.calculators.emt import EMT

        return EMT(**self.calculator_kwargs)
