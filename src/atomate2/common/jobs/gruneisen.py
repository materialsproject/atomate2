"""Jobs for Grüneisen-Parameter computations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import phonopy
from jobflow import Response, job
from phonopy.api_gruneisen import PhonopyGruneisen
from pymatgen.io.phonopy import get_gruneisen_ph_bs_symm_line, get_gruneisenparameter
from pymatgen.phonon.plotter import GruneisenPhononBSPlotter, GruneisenPlotter

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


@job
def compute_gruneisen_param(
    phonopy_yaml_paths_dict: dict,
    mesh: tuple = (20, 20, 20),
    structure: Structure = None,
) -> dict:
    """
    Compute Gruneisen parameters from phonon runs.

    Requires phonopy yaml files from ground, expanded and contracted structures

    phonopy_yaml_paths_dict: optimized pymatgen structure obj
    mesh: sampling mesh
    structure: pymatgen structure object at ground state
    """
    ground = phonopy.load(Path(phonopy_yaml_paths_dict["ground"]) / "phonopy.yaml")
    plus = phonopy.load(Path(phonopy_yaml_paths_dict["plus"]) / "phonopy.yaml")
    minus = phonopy.load(Path(phonopy_yaml_paths_dict["minus"]) / "phonopy.yaml")

    gru = PhonopyGruneisen(phonon=ground, phonon_plus=plus, phonon_minus=minus)
    gru.set_mesh(mesh=mesh)
    gru.write_yaml_mesh()
    gru_mesh_file = Path("gruneisen.yaml")
    gru_mesh_file.rename("gruneisen_mesh.yaml")
    gruneisen_parameter = get_gruneisenparameter(
        gruneisen_path="gruneisen_mesh.yaml", structure=structure
    )
    gp_plot = GruneisenPlotter(gruneisen=gruneisen_parameter)
    gp_plot.save_plot(filename="gruneisen_mesh.pdf", units="thz", img_format="pdf")
    ground.auto_band_structure()
    gru.set_band_structure(bands=ground.band_structure.qpoints)
    gru.write_yaml_band_structure()
    gru_band_file = Path("gruneisen.yaml")
    gru_band_file.rename("gruneisen_band.yaml")
    gruneisen_band_structure = get_gruneisen_ph_bs_symm_line(
        gruneisen_path="gruneisen_band.yaml", structure=structure
    )
    gp_bs_plot = GruneisenPhononBSPlotter(bs=gruneisen_band_structure)
    gp_bs_plot.save_plot_gs(filename="gruneisen_band.pdf", img_format="pdf")
    return {"mesh": gruneisen_parameter, "band": gruneisen_band_structure}
