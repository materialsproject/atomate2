from atomate2.aims.sets.base import AimsInputSet
import pytest
import numpy as np

control_in_str = '#===============================================================================\n# Created using the Atomic Simulation Environment (ASE)\n\n# Thu Oct  5 12:27:49 2023\n\n#===============================================================================\nxc                                 pbe\nk_grid                             2 2 2\ncompute_forces                     .true.\n#===============================================================================\n\n################################################################################\n#\n#  FHI-aims code project\n#  VB, Fritz-Haber Institut, 2009\n#\n#  Suggested "light" defaults for Si atom (to be pasted into control.in file)\n#  Be sure to double-check any results obtained with these settings for post-processing,\n#  e.g., with the "tight" defaults and larger basis sets.\n#\n#  2020/09/08 Added f function to "light" after reinspection of Delta test outcomes.\n#             This was done for all of Al-Cl and is a tricky decision since it makes\n#             "light" calculations measurably more expensive for these elements.\n#             Nevertheless, outcomes for P, S, Cl (and to some extent, Si) appear\n#             to justify this choice.\n#\n################################################################################\n  species        Si\n#     global species definitions\n    nucleus             14\n    mass                28.0855\n#\n    l_hartree           4\n#\n    cut_pot             3.5          1.5  1.0\n    basis_dep_cutoff    1e-4\n#\n    radial_base         42 5.0\n    radial_multiplier   1\n    angular_grids       specified\n      division   0.5866   50\n      division   0.9616  110\n      division   1.2249  194\n      division   1.3795  302\n#      division   1.4810  434\n#      division   1.5529  590\n#      division   1.6284  770\n#      division   1.7077  974\n#      division   2.4068 1202\n#      outer_grid   974\n      outer_grid 302\n################################################################################\n#\n#  Definition of "minimal" basis\n#\n################################################################################\n#     valence basis states\n    valence      3  s   2.\n    valence      3  p   2.\n#     ion occupancy\n    ion_occ      3  s   1.\n    ion_occ      3  p   1.\n################################################################################\n#\n#  Suggested additional basis functions. For production calculations, \n#  uncomment them one after another (the most important basis functions are\n#  listed first).\n#\n#  Constructed for dimers: 1.75 A, 2.0 A, 2.25 A, 2.75 A, 3.75 A\n#\n################################################################################\n#  "First tier" - improvements: -571.96 meV to -37.03 meV\n     hydro 3 d 4.2\n     hydro 2 p 1.4\n     hydro 4 f 6.2\n     ionic 3 s auto\n#  "Second tier" - improvements: -16.76 meV to -3.03 meV\n#     hydro 3 d 9\n#     hydro 5 g 9.4\n#     hydro 4 p 4\n#     hydro 1 s 0.65\n#  "Third tier" - improvements: -3.89 meV to -0.60 meV\n#     ionic 3 d auto\n#     hydro 3 s 2.6\n#     hydro 4 f 8.4\n#     hydro 3 d 3.4\n#     hydro 3 p 7.8\n#  "Fourth tier" - improvements: -0.33 meV to -0.11 meV\n#     hydro 2 p 1.6\n#     hydro 5 g 10.8\n#     hydro 5 f 11.2\n#     hydro 3 d 1\n#     hydro 4 s 4.5\n#  Further basis functions that fell out of the optimization - noise\n#  level... < -0.08 meV\n#     hydro 4 d 6.6\n#     hydro 5 g 16.4\n#     hydro 4 d 9\n'
control_in_str_rel = '#===============================================================================\n# Created using the Atomic Simulation Environment (ASE)\n\n# Thu Oct  5 12:33:50 2023\n\n#===============================================================================\nxc                                 pbe\nk_grid                             2 2 2\nrelax_geometry                     trm 1e-3\ncompute_forces                     .true.\n#===============================================================================\n\n################################################################################\n#\n#  FHI-aims code project\n#  VB, Fritz-Haber Institut, 2009\n#\n#  Suggested "light" defaults for Si atom (to be pasted into control.in file)\n#  Be sure to double-check any results obtained with these settings for post-processing,\n#  e.g., with the "tight" defaults and larger basis sets.\n#\n#  2020/09/08 Added f function to "light" after reinspection of Delta test outcomes.\n#             This was done for all of Al-Cl and is a tricky decision since it makes\n#             "light" calculations measurably more expensive for these elements.\n#             Nevertheless, outcomes for P, S, Cl (and to some extent, Si) appear\n#             to justify this choice.\n#\n################################################################################\n  species        Si\n#     global species definitions\n    nucleus             14\n    mass                28.0855\n#\n    l_hartree           4\n#\n    cut_pot             3.5          1.5  1.0\n    basis_dep_cutoff    1e-4\n#\n    radial_base         42 5.0\n    radial_multiplier   1\n    angular_grids       specified\n      division   0.5866   50\n      division   0.9616  110\n      division   1.2249  194\n      division   1.3795  302\n#      division   1.4810  434\n#      division   1.5529  590\n#      division   1.6284  770\n#      division   1.7077  974\n#      division   2.4068 1202\n#      outer_grid   974\n      outer_grid 302\n################################################################################\n#\n#  Definition of "minimal" basis\n#\n################################################################################\n#     valence basis states\n    valence      3  s   2.\n    valence      3  p   2.\n#     ion occupancy\n    ion_occ      3  s   1.\n    ion_occ      3  p   1.\n################################################################################\n#\n#  Suggested additional basis functions. For production calculations, \n#  uncomment them one after another (the most important basis functions are\n#  listed first).\n#\n#  Constructed for dimers: 1.75 A, 2.0 A, 2.25 A, 2.75 A, 3.75 A\n#\n################################################################################\n#  "First tier" - improvements: -571.96 meV to -37.03 meV\n     hydro 3 d 4.2\n     hydro 2 p 1.4\n     hydro 4 f 6.2\n     ionic 3 s auto\n#  "Second tier" - improvements: -16.76 meV to -3.03 meV\n#     hydro 3 d 9\n#     hydro 5 g 9.4\n#     hydro 4 p 4\n#     hydro 1 s 0.65\n#  "Third tier" - improvements: -3.89 meV to -0.60 meV\n#     ionic 3 d auto\n#     hydro 3 s 2.6\n#     hydro 4 f 8.4\n#     hydro 3 d 3.4\n#     hydro 3 p 7.8\n#  "Fourth tier" - improvements: -0.33 meV to -0.11 meV\n#     hydro 2 p 1.6\n#     hydro 5 g 10.8\n#     hydro 5 f 11.2\n#     hydro 3 d 1\n#     hydro 4 s 4.5\n#  Further basis functions that fell out of the optimization - noise\n#  level... < -0.08 meV\n#     hydro 4 d 6.6\n#     hydro 5 g 16.4\n#     hydro 4 d 9\n'
geometry_in_str = "#===============================================================================\n# Created using the Atomic Simulation Environment (ASE)\n\n# Thu Oct  5 12:27:49 2023\n\n#=======================================================\nlattice_vector 0.0000000000000000 2.7149999999999999 2.7149999999999999 \nlattice_vector 2.7149999999999999 0.0000000000000000 2.7149999999999999 \nlattice_vector 2.7149999999999999 2.7149999999999999 0.0000000000000000 \natom_frac 0.0000000000000000 0.0000000000000000 -0.0000000000000000 Si\natom_frac 0.2500000000000000 0.2500000000000000 0.2500000000000000 Si\n"
geometry_in_str_new = "#===============================================================================\n# Created using the Atomic Simulation Environment (ASE)\n\n# Thu Oct  5 12:27:49 2023\n\n#=======================================================\nlattice_vector 0.0000000000000000 2.7149999999999999 2.7149999999999999 \nlattice_vector 2.7149999999999999 0.0000000000000000 2.7149999999999999 \nlattice_vector 2.7149999999999999 2.7149999999999999 0.0000000000000000 \natom_frac -0.0100000000000000 0.0000000000000000 -0.0000000000000000 Si\natom_frac 0.2500000000000000 0.2500000000000000 0.2500000000000000 Si\n"


def test_input_set(Si, species_dir):
    parameters_json_str = (
        "{"
        + f'\n  "xc": "pbe",\n  "species_dir": "{species_dir}",\n  "k_grid": [\n    2,\n    2,\n    2\n  ]\n'
        + "}"
    )
    parameters_json_str_rel = (
        "{"
        + f'\n  "xc": "pbe",\n  "species_dir": "{species_dir}",\n  "k_grid": [\n    2,\n    2,\n    2\n  ],\n  "relax_geometry": "trm 1e-3"\n'
        + "}"
    )

    parameters = {"xc": "pbe", "species_dir": str(species_dir), "k_grid": [2, 2, 2]}
    properties = ("energy", "free_energy", "forces")

    in_set = AimsInputSet(parameters, Si, properties)
    assert geometry_in_str[175:] == in_set.geometry_in.get_str()[175:]
    assert control_in_str[175:] == in_set.control_in.get_str()[175:]
    print(parameters_json_str, "\n\n\n", in_set.parameters_json)
    assert parameters_json_str == in_set.parameters_json

    in_set_copy = in_set.deepcopy()
    assert geometry_in_str[175:] == in_set_copy.geometry_in.get_str()[175:]
    assert control_in_str[175:] == in_set_copy.control_in.get_str()[175:]
    assert parameters_json_str == in_set_copy.parameters_json

    in_set.set_parameters(**parameters, relax_geometry="trm 1e-3")
    assert control_in_str_rel[175:] == in_set.control_in.get_str()[175:]
    assert parameters_json_str_rel == in_set.parameters_json
    assert control_in_str[175:] == in_set_copy.control_in.get_str()[175:]
    assert parameters_json_str == in_set_copy.parameters_json

    in_set.remove_parameters(keys=["relax_geometry"])
    assert control_in_str[175:] == in_set.control_in.get_str()[175:]
    assert parameters_json_str == in_set.parameters_json

    in_set.remove_parameters(keys=["relax_geometry"], strict=False)
    assert control_in_str[175:] == in_set.control_in.get_str()[175:]
    assert parameters_json_str == in_set.parameters_json

    with pytest.raises(ValueError):
        in_set.remove_parameters(keys=["relax_geometry"], strict=True)

    new_atoms = Si.copy()
    new_atoms.set_scaled_positions(np.array([[-0.01, 0, 0], [0.25, 0.25, 0.25]]))
    in_set.set_atoms(new_atoms)
    assert geometry_in_str_new[175:] == in_set.geometry_in.get_str()[175:]
    assert geometry_in_str[175:] == in_set_copy.geometry_in.get_str()[175:]
