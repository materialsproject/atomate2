"""Jobs for Grueneisen-Parameter computations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import phonopy
from jobflow import Flow, Response, job
from phonopy.api_gruneisen import PhonopyGruneisen
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from pymatgen.io.phonopy import get_gruneisen_ph_bs_symm_line, get_gruneisenparameter
from pymatgen.phonon.gruneisen import (
    GruneisenParameter,
    GruneisenPhononBandStructureSymmLine,
)
from pymatgen.phonon.plotter import GruneisenPhononBSPlotter, GruneisenPlotter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from atomate2 import SETTINGS
from atomate2.common.schemas.gruneisen import GruneisenParameterDocument
from atomate2.common.schemas.phonons import PhononBSDOSDoc

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure

    from atomate2.forcefields.jobs import ForceFieldStaticMaker
    from atomate2.vasp.jobs.base import BaseVaspMaker


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


@job(data=[PhononBSDOSDoc])
def run_phonon_jobs(
    opt_struct: dict,
    phonon_maker: BaseVaspMaker | ForceFieldStaticMaker = None,
    symprec: float = SETTINGS.SYMPREC,
) -> Response:
    """Run all phonon jobs if the symmetry stayed the same.

    opt_struct: dict including all optimized structures
        with the keys ground, plus, minus
    phonon_maker: BaseVaspMaker | ForceFieldStaticMaker
    symprec: symmetry precision
    """
    symmetry = []
    for st in opt_struct:
        sga = SpacegroupAnalyzer(opt_struct[st], symprec=symprec)
        symmetry.append(sga.get_space_group_number())
    set_symmetry = list(set(symmetry))
    if len(set_symmetry) == 1:
        jobs = []
        phonon_yaml_dirs = dict.fromkeys(("ground", "plus", "minus"), None)
        phonon_imaginary_modes = dict.fromkeys(("ground", "plus", "minus"), None)
        for st in opt_struct:
            # phonon run for all 3 optimized structures (ground state, expanded, shrunk)
            phonon_job = phonon_maker.make(structure=opt_struct[st])
            phonon_job.append_name(f" {st}")
            # change default phonopy.yaml file name to ensure workflow can be
            # run without having to create folders, thus
            # prevent overwriting and easier to identify yaml file belong
            # to corresponding phonon run
            phonon_job.jobs[-1].function_kwargs.update(
                filename_phonopy_yaml=f"{st}_phonopy.yaml",
                filename_band_yaml=f"{st}_phonon_band_structure.yaml",
                filename_dos_yaml=f"{st}_phonon_dos.yaml",
                filename_bs=f"{st}_phonon_band_structure.pdf",
                filename_dos=f"{st}_phonon_dos.pdf",
            )
            jobs.append(phonon_job)
            # store each phonon run task doc
            phonon_yaml_dirs[st] = phonon_job.output.jobdirs.taskdoc_run_job_dir
            phonon_imaginary_modes[st] = phonon_job.output.has_imaginary_modes

        return Response(
            replace=Flow(jobs),
            output={
                "phonon_yaml": phonon_yaml_dirs,
                "imaginary_modes": phonon_imaginary_modes,
            },
        )
    return Response(output={"space_groups": set_symmetry}, stop_jobflow=True)


@job(
    output_schema=GruneisenParameterDocument,
    data=[GruneisenParameter, GruneisenPhononBandStructureSymmLine],
)
def compute_gruneisen_param(
    code: str,
    phonopy_yaml_paths_dict: dict,
    phonon_imaginary_modes_info: dict,
    kpath_scheme: str,
    symprec: float,
    mesh: tuple = (20, 20, 20),
    structure: Structure = None,
    **compute_gruneisen_param_kwargs,
) -> GruneisenParameterDocument:
    """
    Compute Grueneisen parameters from phonon runs.

    Requires phonopy yaml files from ground, expanded and contracted structures

    phonopy_yaml_paths_dict: phonopy yaml files path for ground, expanded and
        contracted structure phonon runs
    phonon_imaginary_modes_info: dict with bool indicating if structure
        has imaginary modes
    mesh: sampling mesh
    structure: pymatgen structure object at ground state
    compute_gruneisen_param_kwargs: kwargs for phonopy Grueneisen
        api and pymatgen plotters
    """
    ground = phonopy.load(
        Path(phonopy_yaml_paths_dict["ground"]) / "ground_phonopy.yaml"
    )
    plus = phonopy.load(Path(phonopy_yaml_paths_dict["plus"]) / "plus_phonopy.yaml")
    minus = phonopy.load(Path(phonopy_yaml_paths_dict["minus"]) / "minus_phonopy.yaml")

    gru = PhonopyGruneisen(phonon=ground, phonon_plus=plus, phonon_minus=minus)
    gru.set_mesh(
        mesh=mesh,
        shift=compute_gruneisen_param_kwargs.get("shift", None),
        is_gamma_center=compute_gruneisen_param_kwargs.get("is_gamma_center", False),
        is_time_reversal=compute_gruneisen_param_kwargs.get("is_time_reversal", True),
        is_mesh_symmetry=compute_gruneisen_param_kwargs.get("is_mesh_symmetry", True),
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
        structure=structure,
        kpath_scheme=kpath_scheme,
        symprec=symprec,
    )

    qpoints, connections = get_band_qpoints_and_path_connections(
        kpath_concrete, npoints=compute_gruneisen_param_kwargs.get("npoints_band", 101)
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
        thermal_conductivity_slack = gruneisen_parameter.thermal_conductivity_slack()
    except ValueError:
        thermal_conductivity_slack = None

    derived_properties = {
        "average_gruneisen": average_gruneisen,
        "thermal_conductivity_slack": thermal_conductivity_slack,
    }

    return GruneisenParameterDocument.from_structure(
        meta_structure=structure,
        code=code,
        gruneisen_parameter_inputs=gruneisen_parameter_inputs,
        phonon_runs_has_imaginary_modes=phonon_imaginary_modes_info,
        gruneisen_parameter=gruneisen_parameter,
        gruneisen_band_structure=gruneisen_band_structure,
        derived_properties=derived_properties,
    )
