"""Jobs for Grüneisen-Parameter computations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import phonopy
from jobflow import Response, job
from phonopy.api_gruneisen import PhonopyGruneisen
from pymatgen.io.phonopy import get_gruneisen_ph_bs_symm_line, get_gruneisenparameter
from pymatgen.phonon.gruneisen import (
    GruneisenParameter,
    GruneisenPhononBandStructureSymmLine,
)
from pymatgen.phonon.plotter import GruneisenPhononBSPlotter, GruneisenPlotter

from atomate2.common.schemas.gruneisen import GruneisenParameterDocument

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure


@job
def shrink_expand_structure(structure: Structure, perc_vol: float) -> Response:
    """
    Create structures with expanded and reduced volumes.

    structure: optimized pymatgen structure obj
    perc_vol: percentage to shrink and expand the volume
    """
    plus_struct = structure.copy()
    minus_struct = structure.copy()

    plus_struct.scale_lattice(volume=structure.volume * (1 + perc_vol))
    minus_struct.scale_lattice(volume=structure.volume * (1 - perc_vol))

    return Response(output={"plus": plus_struct, "minus": minus_struct})


@job(
    output=GruneisenParameterDocument,
    data=[GruneisenParameter, GruneisenPhononBandStructureSymmLine],
)
def compute_gruneisen_param(
    phonopy_yaml_paths_dict: dict,
    phonon_imaginary_modes_info: dict,
    mesh: tuple = (20, 20, 20),
    structure: Structure = None,
    **plot_kwargs,
) -> GruneisenParameterDocument:
    """
    Compute Gruneisen parameters from phonon runs.

    Requires phonopy yaml files from ground, expanded and contracted structures

    phonopy_yaml_paths_dict: optimized pymatgen structure obj
    phonon_imaginary_modes_info: dict with bool indicating if structure
        has imaginary modes
    mesh: sampling mesh
    structure: pymatgen structure object at ground state
    plot_kwargs: kwargs for pymatgen Gruneisen plotters
    """
    ground = phonopy.load(
        Path(phonopy_yaml_paths_dict["ground"]) / "ground_phonopy.yaml"
    )
    plus = phonopy.load(Path(phonopy_yaml_paths_dict["plus"]) / "plus_phonopy.yaml")
    minus = phonopy.load(Path(phonopy_yaml_paths_dict["minus"]) / "minus_phonopy.yaml")

    if plot_kwargs.get("units") is None:
        plot_kwargs["units"] = "thz"
    elif plot_kwargs.get("img_format") is None:
        plot_kwargs["img_format"] = "pdf"

    gru = PhonopyGruneisen(phonon=ground, phonon_plus=plus, phonon_minus=minus)
    gru.set_mesh(mesh=mesh)
    gru._mesh.write_yaml(filename="gruneisen_mesh.yaml")  # noqa: SLF001
    gruneisen_parameter = get_gruneisenparameter(
        gruneisen_path="gruneisen_mesh.yaml", structure=structure
    )
    gp_plot = GruneisenPlotter(gruneisen=gruneisen_parameter)
    gp_plot.save_plot(
        filename="gruneisen_mesh.pdf",
        units=plot_kwargs.get("units"),
        img_format=plot_kwargs.get("img_format"),
    )
    ground.auto_band_structure()
    gru.set_band_structure(bands=ground.band_structure.qpoints)
    gru._band_structure.write_yaml(filename="gruneisen_band.yaml")  # noqa: SLF001
    gruneisen_band_structure = get_gruneisen_ph_bs_symm_line(
        gruneisen_path="gruneisen_band.yaml", structure=structure
    )
    gp_bs_plot = GruneisenPhononBSPlotter(bs=gruneisen_band_structure)
    gp_bs_plot.save_plot_gs(filename="gruneisen_band.pdf", img_format="pdf")

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
        thermal_conductivity_slack = gruneisen_parameter.thermal_conductivity_slack()
    except ValueError:
        thermal_conductivity_slack = None

    derived_properties = {
        "average_gruneisen": average_gruneisen,
        "thermal_conductivity_slack": thermal_conductivity_slack,
    }

    return GruneisenParameterDocument(
        gruneisen_parameter_inputs=gruneisen_parameter_inputs,
        phonon_runs_has_imaginary_modes=phonon_imaginary_modes_info,
        gruneisen_parameter=gruneisen_parameter,
        gruneisen_band_structure=gruneisen_band_structure,
        derived_properties=derived_properties,
    )
