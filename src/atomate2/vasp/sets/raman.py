"""VASP input set for the Raman dielectric calculations."""

from __future__ import annotations

from dataclasses import dataclass

from atomate2.vasp.sets.core import StaticSetGenerator


@dataclass
class RamanDielectricSetGenerator(StaticSetGenerator):
    """
    LEPSILON static run for structures displaced along normal modes.

    Based on the phonondb INCAR-nac used in atomate-CRD.

    Parameters
    ----------
    kpoints_length
        Rk of the VASP fully automatic Gamma-centred mesh. RamanMaker sets it
        from the band gap. None uses the atomate2 reciprocal-density mesh.
    lpead
        LPEAD tag.
    ediff
        Electronic convergence criterion.
    **kwargs
        Other keyword arguments passed to :obj:`StaticSetGenerator`.
    """

    lepsilon: bool = True
    kpoints_length: float | None = 40.0
    lpead: bool = False
    ediff: float = 1e-8
    auto_ismear: bool = False
    auto_ispin: bool = False

    @property
    def incar_updates(self) -> dict:
        """Get updates to the INCAR for a Raman dielectric job."""
        return {
            "PREC": "Accurate",
            "IBRION": -1,
            "NSW": 0,
            "ISIF": 2,
            "NELMIN": 5,
            "EDIFF": self.ediff,
            "ISMEAR": 0,
            "SIGMA": 0.01,
            "ALGO": "Normal",
            "LREAL": False,
            "ADDGRID": True,
            "LEPSILON": True,
            "LPEAD": self.lpead,
            "ISPIN": 1,
            "LWAVE": False,
            "LCHARG": False,
            "LAECHG": False,
            "LVHAR": False,
            "LVTOT": False,
            "LELF": False,
        }

    @property
    def kpoints_updates(self) -> dict:
        """Get updates to the KPOINTS for a Raman dielectric job."""
        return {} if self.kpoints_length is None else {"length": self.kpoints_length}
