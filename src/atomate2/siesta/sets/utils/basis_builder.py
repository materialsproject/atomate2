"""Helper functions for building custom PAO.Basis blocks.

This module provides dataclasses and helper functions to programmatically
create `%block PAO.Basis` specifications for SIESTA calculations.

Key Features:
- Species variant support (O_surface, O_bulk, O_ghost, etc.)
- Complete PAO shell parameter control
- Automatic FDF block generation
- Validation of nzeta vs rc length
- Support for all SIESTA PAO flags

Usage:
    from atomate2.siesta.sets.utils.basis_builder import create_pao_basis

    basis_spec = {
        "O_surface": {
            "shells": [
                {"n": 2, "l": 0, "nzeta": 2, "rc": [6.0, 0.0]},
                {"n": 2, "l": 1, "nzeta": 2, "rc": [7.0, 0.0], "polarization": True},
            ]
        },
        "O_bulk": {
            "shells": [
                {"n": 2, "l": 0, "nzeta": 2, "rc": [4.5, 0.0]},
                {"n": 2, "l": 1, "nzeta": 2, "rc": [5.5, 0.0]},
            ]
        }
    }

    pao_basis_block = create_pao_basis(basis_spec)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["PAOBasisSpecies", "PAOShell", "create_pao_basis"]


@dataclass
class PAOShell:
    """
    Represents a single PAO orbital shell.

    Attributes
    ----------
    l : int
        Angular momentum quantum number (0=s, 1=p, 2=d, 3=f)
    nzeta : int
        Number of zeta functions (1=SZ, 2=DZ, 3=TZ, etc.)
    rc : list[float]
        Cutoff radii in Bohr (length must match nzeta)
    n : int, optional
        Principal quantum number (e.g., 2 for 2p, 3 for 3d)
    polarization : bool
        Whether this is a polarization orbital (default: False)
    nzeta_pol : int, optional
        Number of polarization functions (default: 1)
    split_norm_flag : bool
        Enable split-norm method (default: False)
    split_norm : float, optional
        Split norm value (0.0-1.0)
    soft_conf_flag : bool
        Enable soft confinement (default: False)
    v0_soft : float, optional
        Soft confinement potential (Ry)
    ri_soft : float, optional
        Inner radius for soft confinement (Bohr)
    filteret_flag : bool
        Enable filteret (default: False)
    filteret_cutoff : float, optional
        Filteret cutoff (Ry)
    charge_conf_flag : bool
        Enable charged confinement (default: False)
    z_charge : float, optional
        Charge for confinement
    screen : float, optional
        Screening parameter
    delta : float, optional
        Delta parameter
    contraction : str, optional
        Contraction scheme

    Examples
    --------
    Simple DZ shell for 2p:
    >>> shell = PAOShell(n=2, l=1, nzeta=2, rc=[5.0, 3.5])

    Polarization shell:
    >>> shell = PAOShell(l=2, nzeta=1, rc=[4.0], polarization=True)

    Shell with soft confinement:
    >>> shell = PAOShell(
    ...     n=2,
    ...     l=0,
    ...     nzeta=2,
    ...     rc=[6.0, 0.0],
    ...     soft_conf_flag=True,
    ...     v0_soft=40.0,
    ...     ri_soft=0.9,
    ... )
    """

    l: int  # noqa: E741  # Standard physics notation for angular momentum quantum number
    nzeta: int
    rc: list[float]
    n: int | None = None
    polarization: bool = False
    nzeta_pol: int | None = None
    split_norm_flag: bool = False
    split_norm: float | None = None
    soft_conf_flag: bool = False
    v0_soft: float | None = None
    ri_soft: float | None = None
    filteret_flag: bool = False
    filteret_cutoff: float | None = None
    charge_conf_flag: bool = False
    z_charge: float | None = None
    screen: float | None = None
    delta: float | None = None
    contraction: str | None = None

    def __post_init__(self):
        """Validate shell parameters."""
        # Validate l
        if self.l < 0 or self.l > 3:
            raise ValueError(f"Angular momentum l={self.l} must be 0-3 (s, p, d, f)")

        # Validate nzeta
        if self.nzeta < 1:
            raise ValueError(f"nzeta={self.nzeta} must be >= 1")

        # Validate rc length matches nzeta
        if len(self.rc) != self.nzeta:
            raise ValueError(
                f"rc length ({len(self.rc)}) must match nzeta ({self.nzeta})"
            )

        # Validate split_norm if provided
        if self.split_norm is not None:
            if not (0.0 <= self.split_norm <= 1.0):
                raise ValueError(
                    f"split_norm={self.split_norm} must be between 0.0 and 1.0"
                )

        # Auto-set nzeta_pol for polarization orbitals
        if self.polarization and self.nzeta_pol is None:
            self.nzeta_pol = 1

    @property
    def l_symbol(self) -> str:
        """Return angular momentum symbol (s, p, d, f)."""
        symbols = {0: "s", 1: "p", 2: "d", 3: "f"}
        return symbols[self.l]

    def to_fdf_lines(self) -> list[str]:
        """
        Generate FDF lines for this shell.

        Returns
        -------
        list[str]
            Lines to include in %block PAO.Basis

        Examples
        --------
        >>> shell = PAOShell(n=2, l=1, nzeta=2, rc=[5.0, 3.5])
        >>> shell.to_fdf_lines()
        ['  n=2  1  2', '    5.0  3.5']
        """
        lines = []

        # First line: n=N l nzeta [flags]
        parts = []
        if self.n is not None:
            parts.append(f"n={self.n}")
        parts.append(str(self.l))
        parts.append(str(self.nzeta))

        # Add optional flags
        if self.polarization:
            parts.append("P")
            if self.nzeta_pol is not None and self.nzeta_pol != 1:
                parts.append(str(self.nzeta_pol))

        if self.split_norm_flag:
            parts.append("S")
            if self.split_norm is not None:
                parts.append(str(self.split_norm))

        if self.soft_conf_flag:
            parts.append("E")
            if self.v0_soft is not None:
                parts.append(str(self.v0_soft))
            if self.ri_soft is not None:
                parts.append(str(self.ri_soft))

        if self.filteret_flag:
            parts.append("F")
            if self.filteret_cutoff is not None:
                parts.append(str(self.filteret_cutoff))

        if self.charge_conf_flag:
            parts.append("Q")
            if self.z_charge is not None:
                parts.append(str(self.z_charge))

        if self.screen is not None:
            parts.append("T")
            parts.append(str(self.screen))

        if self.delta is not None:
            parts.append("D")
            parts.append(str(self.delta))

        if self.contraction is not None:
            parts.append("C")
            parts.append(self.contraction)

        lines.append("  " + "  ".join(parts))

        # Second line: cutoff radii
        rc_str = "  ".join(str(r) for r in self.rc)
        lines.append("    " + rc_str)

        return lines


@dataclass
class PAOBasisSpecies:
    """
    Represents a complete PAO basis for one species.

    Attributes
    ----------
    label : str
        Species label (e.g., "Si", "O_surface", "O_ghost")
    shells : list[PAOShell]
        List of orbital shells
    basis_type : str, optional
        Basis type (default: "split")
    ionic_charge : float, optional
        Ionic charge for this species

    Examples
    --------
    >>> shells = [
    ...     PAOShell(n=2, l=0, nzeta=2, rc=[6.0, 0.0]),
    ...     PAOShell(n=2, l=1, nzeta=2, rc=[7.0, 0.0], polarization=True),
    ... ]
    >>> species = PAOBasisSpecies(label="O_surface", shells=shells)
    """

    label: str
    shells: list[PAOShell] = field(default_factory=list)
    basis_type: str | None = None
    ionic_charge: float | None = None

    def to_fdf_lines(self) -> list[str]:
        """
        Generate FDF lines for this species.

        Returns
        -------
        list[str]
            Lines to include in %block PAO.Basis

        Examples
        --------
        >>> shells = [PAOShell(n=2, l=0, nzeta=2, rc=[6.0, 0.0])]
        >>> species = PAOBasisSpecies(label="O", shells=shells)
        >>> species.to_fdf_lines()
        ['O  1', '  n=2  0  2', '    6.0  0.0']
        """
        lines = []

        # First line: Label NumberOfShells [flags]
        header = f"{self.label}  {len(self.shells)}"

        if self.basis_type is not None:
            header += f"  {self.basis_type}"

        if self.ionic_charge is not None:
            header += f"  {self.ionic_charge}"

        lines.append(header)

        # Add all shell definitions
        for shell in self.shells:
            lines.extend(shell.to_fdf_lines())

        return lines


def create_pao_basis(basis_spec: dict[str, dict[str, Any]]) -> list[str]:
    """
    Create a %block PAO.Basis from a high-level specification.

    Parameters
    ----------
    basis_spec : dict[str, dict[str, Any]]
        Dictionary mapping species labels to their basis specifications.

        Format:
        {
            "species_label": {
                "shells": [
                    {"l": 0, "nzeta": 2, "rc": [6.0, 0.0], "n": 2, ...},
                    ...
                ],
                "basis_type": "split",  # optional
                "ionic_charge": 0.0,    # optional
            }
        }

    Returns
    -------
    list[str]
        Lines for %block PAO.Basis (to be used in user_params)

    Examples
    --------
    Simple example with two species:

    >>> basis_spec = {
    ...     "O_surface": {
    ...         "shells": [
    ...             {"n": 2, "l": 0, "nzeta": 2, "rc": [6.0, 0.0]},
    ...             {
    ...                 "n": 2,
    ...                 "l": 1,
    ...                 "nzeta": 2,
    ...                 "rc": [7.0, 0.0],
    ...                 "polarization": True,
    ...             },
    ...         ]
    ...     },
    ...     "O_bulk": {
    ...         "shells": [
    ...             {"n": 2, "l": 0, "nzeta": 2, "rc": [4.5, 0.0]},
    ...             {"n": 2, "l": 1, "nzeta": 2, "rc": [5.5, 0.0]},
    ...         ]
    ...     },
    ... }
    >>> pao_basis = create_pao_basis(basis_spec)
    >>> # Use in RelaxMaker:
    >>> maker = RelaxMaker(user_params={"%block PAO.Basis": pao_basis})

    Complex example with all features:

    >>> basis_spec = {
    ...     "Fe": {
    ...         "shells": [
    ...             {
    ...                 "n": 3,
    ...                 "l": 2,
    ...                 "nzeta": 2,
    ...                 "rc": [5.0, 0.0],
    ...                 "polarization": True,
    ...                 "split_norm_flag": True,
    ...                 "split_norm": 0.15,
    ...             },
    ...             {"n": 4, "l": 0, "nzeta": 2, "rc": [6.0, 0.0]},
    ...         ]
    ...     },
    ...     "O_surface": {
    ...         "shells": [
    ...             {
    ...                 "n": 2,
    ...                 "l": 0,
    ...                 "nzeta": 2,
    ...                 "rc": [6.0, 0.0],
    ...                 "soft_conf_flag": True,
    ...                 "v0_soft": 40.0,
    ...                 "ri_soft": 0.9,
    ...             },
    ...         ],
    ...         "basis_type": "split",
    ...     },
    ... }
    >>> pao_basis = create_pao_basis(basis_spec)
    """
    all_lines = []

    # Process each species
    for label, spec in basis_spec.items():
        # Convert shell dictionaries to PAOShell objects
        shells = []
        for shell_dict in spec["shells"]:
            if isinstance(shell_dict, PAOShell):
                shell = shell_dict
            else:
                # Extract all possible parameters
                shell = PAOShell(
                    l=shell_dict["l"],
                    nzeta=shell_dict["nzeta"],
                    rc=shell_dict["rc"],
                    n=shell_dict.get("n"),
                    polarization=shell_dict.get("polarization", False),
                    nzeta_pol=shell_dict.get("nzeta_pol"),
                    split_norm_flag=shell_dict.get("split_norm_flag", False),
                    split_norm=shell_dict.get("split_norm"),
                    soft_conf_flag=shell_dict.get("soft_conf_flag", False),
                    v0_soft=shell_dict.get("v0_soft"),
                    ri_soft=shell_dict.get("ri_soft"),
                    filteret_flag=shell_dict.get("filteret_flag", False),
                    filteret_cutoff=shell_dict.get("filteret_cutoff"),
                    charge_conf_flag=shell_dict.get("charge_conf_flag", False),
                    z_charge=shell_dict.get("z_charge"),
                    screen=shell_dict.get("screen"),
                    delta=shell_dict.get("delta"),
                    contraction=shell_dict.get("contraction"),
                )
            shells.append(shell)

        # Create species object
        species = PAOBasisSpecies(
            label=label,
            shells=shells,
            basis_type=spec.get("basis_type"),
            ionic_charge=spec.get("ionic_charge"),
        )

        # Add to output
        all_lines.extend(species.to_fdf_lines())

    return all_lines
