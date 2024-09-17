#!/usr/bin/env python3

import pathlib

import numpy as np

from atomate2.jdftx.io.jdftxinfile import JDFTXInfile


# read file example
p = pathlib.Path(__file__)
filename = p.parents[0] / pathlib.Path("input-simple1.in")
jin1 = JDFTXInfile.from_file(filename)
print(jin1)
# jin1.write_file('test-write.in')
print("===============================================================")



# strict dictionary initialization example
lattice = np.array(
    [
        [22.6767599999999980, 0, 0],
        [0, 22.6767599999999980, 0],
        [0, 0, 22.6767599999999980],
    ]
)
matrix_elements = ["R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"]
# flatten lattice matrix so matrix rows are concantenated
lattice_dict = dict(zip(matrix_elements, np.ndarray.flatten(lattice, order="C")))

atoms = ["O", "H", "H"]
coords = [
    [0.480785272057, 0.492689037759, 0.47474985113],
    [0.544367024943, 0.540886680214, 0.475393875341],
    [0.474847703, 0.466424282027, 0.549856273529],
]
temp = list(zip(atoms, coords))
ion_dict = [
    {
        "species-id": atoms[i],
        "x0": coords[i][0],
        "x1": coords[i][1],
        "x2": coords[i][2],
        "moveScale": 1,
    }
    for i in range(len(atoms))
]
water_tagdict = {
    "lattice": lattice_dict,
    "ion": ion_dict,
    "latt-move-scale": {"s0": 0, "s1": 0, "s2": 0},
    "kpoint-folding": {"n0": 1, "n1": 1, "n2": 1},
    "kpoint": [{"k0": 0, "k1": 0, "k2": 0, "weight": 1}],
    "ion-species": "GBRV/$ID_ONCV_PBE.upf",
    "elec-cutoff": {"Ecut": 20, "EcutRho": 100},
    "wavefunction": {"lcao": True},
    "electronic-minimize": {"nIterations": 100, "energyDiffThreshold": 1e-9},
    "spintype": "no-spin",
    "elec-n-bands": 20,
    "elec-ex-corr": "gga-PBE",
    "dump-name": "jdft.$VAR",
    #"dump": {"freq": "End", "var": "State"},  # TODO add support for dump lists
    "initial-magnetic-moments": "C 0.1 0.2 O 0.6",
}
jin2 = JDFTXInfile.from_dict(water_tagdict)
print(jin2)
print("===============================================================")


# initialization from lists over 2 dictionaries with conversion between dict/list representations
lattice = np.array(
    [
        [22.6767599999999980, 0, 0],
        [0, 22.6767599999999980, 0],
        [0, 0, 22.6767599999999980],
    ]
)

atoms = np.array([["O", "H", "H"]]).T
coords = [
    [0.480785272057, 0.492689037759, 0.47474985113],
    [0.544367024943, 0.540886680214, 0.475393875341],
    [0.474847703, 0.466424282027, 0.549856273529],
]
sd = np.array([[0, 1, 1]]).T
ion_list = np.concatenate((atoms, coords, sd), axis=1)
struc_dict = {
    "lattice": lattice,
    "ion": ion_list,
}
water_tagdict = {
    "latt-move-scale": [0, 0, 0],
    "kpoint-folding": [1, 1, 1],
    "kpoint": [{"k0": 0, "k1": 0, "k2": 0, "weight": 1}],
    "ion-species": "GBRV/$ID_ONCV_PBE.upf",
    "elec-cutoff": [20, 100],
    "wavefunction": {"lcao": True},
    "electronic-minimize": {"nIterations": 100, "energyDiffThreshold": 1e-9},
    "spintype": "no-spin",
    "elec-n-bands": 20,
    "elec-ex-corr": "gga-PBE",
    "dump-name": "jdft.$VAR",
    #"dump": {"freq": "End", "var": "State"},
}
jin3A = JDFTXInfile.from_dict(water_tagdict)
jin3B = JDFTXInfile.from_dict(struc_dict)
jin3 = jin3A + jin3B
print(jin3)
print("-------------------------")
jin3_list = JDFTXInfile.get_list_representation(jin3)
jin3_dict = JDFTXInfile.get_dict_representation(jin3)
# just type jin3_list or jin3_dict in console to see how the dictionary changes
