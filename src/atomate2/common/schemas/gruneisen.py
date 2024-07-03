"""General schemas for Grueneisen parameter workflow outputs."""

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from emmet.core.structure import StructureMetadata
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from pydantic import BaseModel, Field
from pymatgen.phonon.gruneisen import (
    GruneisenParameter,
    GruneisenPhononBandStructureSymmLine,
)
from pymatgen.phonon.plotter import GruneisenPhononBSPlotter, freq_units
from pymatgen.util.plotting import pretty_plot

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

    @staticmethod
    def get_gruneisen_weighted_bandstructure(
        gruneisen_band_symline_plotter: GruneisenPhononBSPlotter,
        save_fig: bool = True,
        **kwargs,
    ) -> None:
        """Save a phonon band structure weighted with Grueneisen parameters.

        gruneisen_band_symline_plotter: pymatgen GruneisenPhononBSPlotter obj
        save_fig: bool to switch plot saving
        kwargs: kwargs to adjust plotter
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
            enumerate(data["distances"]), enumerate(data["frequency"])
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
