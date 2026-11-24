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
    ispin
        ISPIN tag; 2 for magnetic materials (MAGMOM from the magmom site property).
    **kwargs
        Other keyword arguments passed to :obj:`StaticSetGenerator`.
    """

    lepsilon: bool = True
    kpoints_length: int | None = 40
    lpead: bool = False
    ediff: float = 1e-8
    ispin: int = 1
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
            "ISPIN": self.ispin,
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
        if self.kpoints_length is None:
            return {}
        return {"length": round(self.kpoints_length)}
