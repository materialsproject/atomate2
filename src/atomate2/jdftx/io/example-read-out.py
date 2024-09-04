#!/usr/bin/env python3

from JDFTXOutfile import JDFTXOutfile
from pathlib import Path

path = Path(__file__)
filename = path.parents[0] / Path("latticeminimize.out")
jout = JDFTXOutfile.from_file(filename)
# print(jout)
# print(jout.structure)
# print(jout.Ecomponents)
# print(jout.is_gc)

dct = jout.to_dict()
dct = jout.electronic_output

print(jout.trajectory_positions)
jout.trajectory_lattice
jout.trajectory_forces
jout.trajectory_ecomponents 

print(dct)