"""Define the Units for FHI-aims calculations."""
from ase import units as ase_units
from numpy import pi

PI = pi
EV = ase_units.CODATA["2002"]["_e"]  # = 1.602176634e-19  # [J]
THZ = 1e12
AMU = ase_units.CODATA["2002"]["_amu"]  # = 1.66053906660e-27  # [kg]
AA = 1e-10

omega_to_THz = (EV / AA**2 / AMU) ** 0.5 / THZ / 2 / PI
