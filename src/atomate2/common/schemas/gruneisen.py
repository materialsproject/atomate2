"""General schemas for Grueneisen parameter workflow outputs."""

import logging
from pathlib import Path
from typing import Union

import phonopy
from emmet.core.structure import StructureMetadata
from phonopy.api_gruneisen import PhonopyGruneisen
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from pydantic import BaseModel, Field
from pymatgen.core.structure import Structure
from pymatgen.io.phonopy import (
    get_gruneisen_ph_bs_symm_line,
    get_gruneisenparameter,
    get_pmg_structure,
)
from pymatgen.io.vasp import Kpoints
from pymatgen.phonon.gruneisen import (
    GruneisenParameter,
    GruneisenPhononBandStructureSymmLine,
)
from pymatgen.phonon.plotter import GruneisenPhononBSPlotter, GruneisenPlotter
from typing_extensions import Self

from atomate2.common.jobs.phonons import _get_kpath

logger = logging.getLogger(__name__)


class GruneisenInputDirs(BaseModel):
    """Collection with all input directories relevant for the Grueneisen run."""

    ground: str | None = Field(
        None, description="The directory with ground state structure phonopy yaml"
    )
    plus: str | None = Field(
        None, description="The directory with expanded structure phonopy yaml"
    )
    minus: str | None = Field(
        None, description="The directory with contracted structure phonopy yaml"
    )


class PhononRunsImaginaryModes(BaseModel):
    """Collection with information whether structure has imaginary modes.

    Information extracted from phonon run for ground, expanded and contracted structures
    """

    ground: bool | None = Field(
        None, description="if true, ground state structure has imaginary modes"
    )
    plus: bool | None = Field(
        None, description="if true, expanded structure has imaginary modes"
    )
    minus: bool | None = Field(
        None, description="if true, contracted structure has imaginary modes"
    )


class GruneisenDerivedProperties(BaseModel):
    """Collection of data derived from the Grueneisen workflow."""

    average_gruneisen: float | None = Field(
        None, description="The average Grueneisen parameter"
    )
    thermal_conductivity_slack: float | None = Field(
        None,
        description="The thermal conductivity at the acoustic "
        "Debye temperature with the Slack formula.",
    )


class GruneisenParameterDocument(StructureMetadata):
    """Collection to data from the Grueneisen computation."""

    code: str = Field("String describing the code for the computation")
    gruneisen_parameter_inputs: GruneisenInputDirs = Field(
        None, description="The directories where the phonon jobs were run."
    )
    phonon_runs_has_imaginary_modes: PhononRunsImaginaryModes | None = Field(
        None,
        description="Collection indicating whether the structures from the "
        "phonon runs have imaginary modes",
    )
    gruneisen_parameter: GruneisenParameter | None = Field(
        None, description="Grueneisen parameter object"
    )
    gruneisen_band_structure: GruneisenPhononBandStructureSymmLine | None = Field(
        None, description="Grueneisen phonon band structure symmetry line object"
    )
    derived_properties: GruneisenDerivedProperties | None = Field(
        None, description="Properties derived from the Grueneisen parameter."
    )

    @classmethod
    def from_phonon_yamls(
        cls,
        phonopy_yaml_paths_dict: dict,
        structure: Structure,
        kpath_scheme: str,
        mesh: Union[float, tuple[float, float, float]],
        phonon_imaginary_modes_info: dict,
        symprec: float,
        code: str,
        compute_gruneisen_param_kwargs: dict,
    ) -> Self:
        """Generate the GruneisenParameterDocument from phonopy yamls.

        Parameters
        ----------
        phonopy_yaml_paths_dict:
            phonopy yaml files path for ground, expanded and
            contracted structure phonon runs
        structure: .Structure
            pymatgen structure object at ground state
        kpath_scheme: str
            scheme to generate kpoints. Please be aware that
            you can only use seekpath with any kind of cell
            Otherwise, please use the standard primitive structure
            Available schemes are:
            "seekpath", "hinuma", "setyawan_curtarolo", "latimer_munro".
            "seekpath" and "hinuma" are the same definition but
            seekpath can be used with any kind of unit cell as
            it relies on phonopy to handle the relationship
            to the primitive cell and not pymatgen
        mesh: float or int or tuple(int, int, int)
            kpoint density (float, int) or sampling mesh (tuple(int, int, int))
        phonon_imaginary_modes_info:
            dict with bool indicating if structure
            has imaginary modes
        symprec: float
            Symmetry precision for symmetry checks and phonon runs.
        code: str
            Code to compute forces
        compute_gruneisen_param_kwargs:
            kwargs for phonopy Grueneisen
            api and pymatgen plotters

        Returns
        -------
        .GruneisenParameterDocument
        """
        ground = phonopy.load(
            Path(phonopy_yaml_paths_dict["ground"]) / "ground_phonopy.yaml"
        )
        plus = phonopy.load(Path(phonopy_yaml_paths_dict["plus"]) / "plus_phonopy.yaml")
        minus = phonopy.load(
            Path(phonopy_yaml_paths_dict["minus"]) / "minus_phonopy.yaml"
        )
        gru = PhonopyGruneisen(phonon=ground, phonon_plus=plus, phonon_minus=minus)
        if type(mesh) is tuple:
            gru.set_mesh(
                mesh=mesh,
                shift=compute_gruneisen_param_kwargs.get("shift"),
                is_gamma_center=compute_gruneisen_param_kwargs.get(
                    "is_gamma_center", False
                ),
                is_time_reversal=compute_gruneisen_param_kwargs.get(
                    "is_time_reversal", True
                ),
                is_mesh_symmetry=compute_gruneisen_param_kwargs.get(
                    "is_mesh_symmetry", True
                ),
            )
        else:
            # kpoint mesh relative to primitive cell
            kpoint = Kpoints.automatic_density(
                structure=get_pmg_structure(ground.primitive),
                kppa=mesh,
                force_gamma=True,
            )
            gru.set_mesh(
                mesh=kpoint.kpts[0],
                shift=compute_gruneisen_param_kwargs.get("shift"),
                is_gamma_center=compute_gruneisen_param_kwargs.get(
                    "is_gamma_center", False
                ),
                is_time_reversal=compute_gruneisen_param_kwargs.get(
                    "is_time_reversal", True
                ),
                is_mesh_symmetry=compute_gruneisen_param_kwargs.get(
                    "is_mesh_symmetry", True
                ),
            )
        gruneisen_mesh_yaml = compute_gruneisen_param_kwargs.get(
            "filename_mesh_yaml", "gruneisen_mesh.yaml"
        )
        gru._mesh.write_yaml(filename=gruneisen_mesh_yaml)  # noqa: SLF001
        gruneisen_parameter = get_gruneisenparameter(
            gruneisen_path=gruneisen_mesh_yaml, structure=structure
        )
        gp_plot = GruneisenPlotter(gruneisen=gruneisen_parameter)
        gruneisen_mesh_plot = compute_gruneisen_param_kwargs.get(
            "gruneisen_mesh", "gruneisen_mesh.pdf"
        )
        gp_plot.save_plot(
            filename=gruneisen_mesh_plot,
            units=compute_gruneisen_param_kwargs.get("units", "thz"),
            img_format=compute_gruneisen_param_kwargs.get("img_format", "pdf"),
        )
        # get phonon band structure
        kpath_dict, kpath_concrete = _get_kpath(
            structure=structure, kpath_scheme=kpath_scheme, symprec=symprec
        )
        qpoints, _connections = get_band_qpoints_and_path_connections(
            kpath_concrete,
            npoints=compute_gruneisen_param_kwargs.get("npoints_band", 101),
        )
        gruneisen_band_yaml = compute_gruneisen_param_kwargs.get(
            "filename_band_yaml", "gruneisen_band.yaml"
        )
        gru.set_band_structure(bands=qpoints)
        gru._band_structure.write_yaml(filename=gruneisen_band_yaml)  # noqa: SLF001
        gruneisen_band_structure = get_gruneisen_ph_bs_symm_line(
            gruneisen_path=gruneisen_band_yaml,
            structure=structure,
            labels_dict=kpath_dict,
        )
        gp_bs_plot = GruneisenPhononBSPlotter(bs=gruneisen_band_structure)

        gruneisen_bs_plot = compute_gruneisen_param_kwargs.get(
            "gruneisen_bs", "gruneisen_band.pdf"
        )
        gp_bs_plot.save_plot_gs(
            filename=gruneisen_bs_plot,
            plot_ph_bs_with_gruneisen=True,
            img_format=compute_gruneisen_param_kwargs.get("img_format", "pdf"),
            **compute_gruneisen_param_kwargs,
        )
        gruneisen_parameter_inputs = {
            "ground": phonopy_yaml_paths_dict["ground"],
            "plus": phonopy_yaml_paths_dict["plus"],
            "minus": phonopy_yaml_paths_dict["minus"],
        }
        try:
            average_gruneisen = gruneisen_parameter.average_gruneisen()
        except ValueError:
            average_gruneisen = None
        try:
            thermal_conductivity_slack = (
                gruneisen_parameter.thermal_conductivity_slack()
            )
        except ValueError:
            thermal_conductivity_slack = None
        derived_properties = {
            "average_gruneisen": average_gruneisen,
            "thermal_conductivity_slack": thermal_conductivity_slack,
        }
        return cls.from_structure(
            meta_structure=structure,
            code=code,
            gruneisen_parameter_inputs=gruneisen_parameter_inputs,
            phonon_runs_has_imaginary_modes=phonon_imaginary_modes_info,
            gruneisen_parameter=gruneisen_parameter,
            gruneisen_band_structure=gruneisen_band_structure,
            derived_properties=derived_properties,
        )
