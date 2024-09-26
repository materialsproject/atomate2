"""General schemas for Grueneisen parameter workflow outputs."""

import logging
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import phonopy
from emmet.core.structure import StructureMetadata
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
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
from pymatgen.phonon.plotter import (
    GruneisenPhononBSPlotter,
    GruneisenPlotter,
    freq_units,
)
from pymatgen.util.plotting import pretty_plot
from typing_extensions import Self

from atomate2.common.schemas.phonons import PhononBSDOSDoc

logger = logging.getLogger(__name__)


class GruneisenInputDirs(BaseModel):
    """Collection with all input directories relevant for the Grueneisen run."""

    ground: Optional[str] = Field(
        None, description="The directory with ground state structure phonopy yaml"
    )
    plus: Optional[str] = Field(
        None, description="The directory with expanded structure phonopy yaml"
    )
    minus: Optional[str] = Field(
        None, description="The directory with contracted structure phonopy yaml"
    )


class PhononRunsImaginaryModes(BaseModel):
    """Collection with information whether structure has imaginary modes.

    Information extracted from phonon run for ground, expanded and contracted structures
    """

    ground: Optional[bool] = Field(
        None, description="if true, ground state structure has imaginary modes"
    )
    plus: Optional[bool] = Field(
        None, description="if true, expanded structure has imaginary modes"
    )
    minus: Optional[bool] = Field(
        None, description="if true, contracted structure has imaginary modes"
    )


class GruneisenDerivedProperties(BaseModel):
    """Collection of data derived from the Grueneisen workflow."""

    average_gruneisen: Optional[float] = Field(
        None, description="The average Grueneisen parameter"
    )
    thermal_conductivity_slack: Optional[float] = Field(
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
    phonon_runs_has_imaginary_modes: Optional[PhononRunsImaginaryModes] = Field(
        None,
        description="Collection indicating whether the structures from the "
        "phonon runs have imaginary modes",
    )
    gruneisen_parameter: Optional[GruneisenParameter] = Field(
        None, description="Grueneisen parameter object"
    )
    gruneisen_band_structure: Optional[GruneisenPhononBandStructureSymmLine] = Field(
        None, description="Grueneisen phonon band structure symmetry line object"
    )
    derived_properties: Optional[GruneisenDerivedProperties] = Field(
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
                    "is_gamma_center", True
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
                    "is_gamma_center", True
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
        kpath_dict, kpath_concrete = PhononBSDOSDoc.get_kpath(
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
        GruneisenParameterDocument.get_gruneisen_weighted_bandstructure(
            gruneisen_band_symline_plotter=gp_bs_plot,
            save_fig=True,
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

    @staticmethod
    def get_gruneisen_weighted_bandstructure(
        gruneisen_band_symline_plotter: GruneisenPhononBSPlotter,
        save_fig: bool = True,
        **kwargs,
    ) -> None:
        """Save a phonon band structure weighted with Grueneisen parameters.

        Parameters
        ----------
        gruneisen_band_symline_plotter: GruneisenPhononBSPlotter
            pymatgen GruneisenPhononBSPlotter obj
        save_fig: bool
            bool to save plots
        kwargs: dict
            keyword arguments to adjust plotter

        Returns
        -------
        None
        """
        u = freq_units(kwargs.get("units", "THz"))
        ax = pretty_plot(12, 8)
        gruneisen_band_symline_plotter._make_ticks(ax)  # noqa: SLF001

        # plot y=0 line
        ax.axhline(0, linewidth=1, color="black")

        # Create custom colormap (default is red to blue)
        cmap = LinearSegmentedColormap.from_list(
            "mycmap", kwargs.get("mycmap", ["red", "blue"])
        )

        data = gruneisen_band_symline_plotter.bs_plot_data()

        # extract min and max Grüneisen parameter values
        max_gruneisen = np.array(data["gruneisen"]).max()
        min_gruneisen = np.array(data["gruneisen"]).min()

        # LogNormalize colormap based on the min and max Grüneisen parameter values
        norm = colors.SymLogNorm(
            vmin=min_gruneisen,
            vmax=max_gruneisen,
            linthresh=1e-2,
            linscale=1,
        )

        for (dists_inx, dists), (_, freqs) in zip(
            enumerate(data["distances"]), enumerate(data["frequency"]), strict=True
        ):
            for band_idx in range(gruneisen_band_symline_plotter.n_bands):
                ys = [freqs[band_idx][j] * u.factor for j in range(len(dists))]
                ys_gru = [
                    data["gruneisen"][dists_inx][band_idx][idx]
                    for idx in range(len(data["distances"][dists_inx]))
                ]
                sc = ax.scatter(
                    dists, ys, c=ys_gru, cmap=cmap, norm=norm, marker="o", s=1
                )

        # Main X and Y Labels
        ax.set_xlabel(r"$\mathrm{Wave\ Vector}$", fontsize=30)
        units = kwargs.get("units", "THz")
        ax.set_ylabel(f"Frequencies ({units})", fontsize=30)
        # X range (K)
        # last distance point
        x_max = data["distances"][-1][-1]
        ax.set_xlim(0, x_max)

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(r"$\gamma \ \mathrm{(logarithmized)}$", fontsize=30)
        plt.tight_layout()
        gruneisen_band_plot = kwargs.get("gruneisen_bs", "gruneisen_band.pdf")
        if save_fig:
            plt.savefig(fname=gruneisen_band_plot)
            plt.close()
        else:
            plt.close()
