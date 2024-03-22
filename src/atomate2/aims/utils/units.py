"""Define the Units for FHI-aims calculations."""

from numpy import pi

PI = pi
EV = 1.602176634e-19  # [J] CODATA 2002
THZ = 1e12
AMU = 1.66053906660e-27  # [kg] CODATA 2002
AA = 1e-10

omegaToTHz = (EV / AA**2 / AMU) ** 0.5 / THZ / 2 / PI  # noqa: N816
ev_per_A3_to_kbar = EV / AA**3 / 1e8  # noqa: N816
