"""(Work)flows for Siesta"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

from jobflow import Flow
from pymatgen.core import Molecule, Structure

from atomate2.siesta.flows.base import BaseSiestaFlowMaker
from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
from atomate2.siesta.powerups import update_user_siesta_settings
from atomate2.siesta.sets.core import SiestaInputGenerator

if TYPE_CHECKING:
    from atomate2.siesta.jobs.base import BaseSiestaMaker


@dataclass
class DifferentBasisSCFFlowMaker(BaseSiestaFlowMaker):
    """
    A maker to run SCF calculations with different basis sizes.

    This class supports multiple strategies for basis set parameters:
    - "standard": Fixed parameters (pao.energy.shift=0.01, pao.split.norm=0.15) for all basis sets
    - "advanced": Customized parameters per basis set size (SZ, DZ, TZ get different values)
    - "legacy": Uses the legacy StaticMaker.scf(basis_set_size) interface

    Parameters
    ----------
    name : str
        Name for the workflow
    static_maker : BaseSiestaMaker
        The static maker to use for calculations
    strategy : str
        Parameter strategy to use: "standard", "advanced", or "legacy"
    """

    name: str = "Different basis scf"
    static_maker: BaseSiestaMaker = field(default_factory=lambda: StaticMaker())
    strategy: str = "standard"

    def _get_basis_sizes(self) -> list[str]:
        """Get the list of basis sizes to compute."""
        return [
            "SZ",
            "MINIMAL",
            "SZP",
            "SZSP",
            "SZ1P",
            "SZP1",
            "DZ",
            "DZP",
            "DZSP",
            "DZP1",
            "DZ1P",
            "STANDARD",
            "DZDP",
            "DZP2",
            "DZ2P",
            "TZ",
            "TZP",
            "TZSP",
            "TZP1",
            "TZ1P",
            "TZDP",
            "TZP2",
            "TZ2P",
            "TZTP",
            "TZP3",
            "TZ3P",
        ]

    def _get_basis_params(self, basis: str) -> dict[str, str | float]:
        """
        Get basis-specific parameters based on the strategy.

        Parameters
        ----------
        basis : str
            Basis set name

        Returns
        -------
        dict[str, str | float]
            Dictionary of basis parameters
        """
        if self.strategy == "standard":
            return {"PAO.EnergyShift": "0.01 Ry", "PAO.SplitNorm": 0.15}

        elif self.strategy == "advanced":
            basis_params = {
                # Single-Zeta
                "SZ": {"PAO.EnergyShift": "0.02 Ry", "PAO.SplitNorm": 0.15},
                "MINIMAL": {"PAO.EnergyShift": "0.02 Ry", "PAO.SplitNorm": 0.15},
                "SZP": {"PAO.EnergyShift": "0.02 Ry", "PAO.SplitNorm": 0.15},
                "SZSP": {"PAO.EnergyShift": "0.02 Ry", "PAO.SplitNorm": 0.15},
                "SZ1P": {"PAO.EnergyShift": "0.02 Ry", "PAO.SplitNorm": 0.15},
                "SZP1": {"PAO.EnergyShift": "0.02 Ry", "PAO.SplitNorm": 0.15},
                # Double-Zeta
                "DZ": {"PAO.EnergyShift": "0.01 Ry", "PAO.SplitNorm": 0.20},
                "DZP": {"PAO.EnergyShift": "0.01 Ry", "PAO.SplitNorm": 0.20},
                "DZSP": {"PAO.EnergyShift": "0.01 Ry", "PAO.SplitNorm": 0.20},
                "DZP1": {"PAO.EnergyShift": "0.01 Ry", "PAO.SplitNorm": 0.20},
                "DZ1P": {"PAO.EnergyShift": "0.01 Ry", "PAO.SplitNorm": 0.20},
                "STANDARD": {"PAO.EnergyShift": "0.01 Ry", "PAO.SplitNorm": 0.20},
                "DZDP": {"PAO.EnergyShift": "0.01 Ry", "PAO.SplitNorm": 0.20},
                "DZP2": {"PAO.EnergyShift": "0.01 Ry", "PAO.SplitNorm": 0.20},
                "DZ2P": {"PAO.EnergyShift": "0.01 Ry", "PAO.SplitNorm": 0.20},
                # Triple-Zeta
                "TZ": {"PAO.EnergyShift": "0.005 Ry", "PAO.SplitNorm": 0.25},
                "TZP": {"PAO.EnergyShift": "0.005 Ry", "PAO.SplitNorm": 0.25},
                "TZSP": {"PAO.EnergyShift": "0.005 Ry", "PAO.SplitNorm": 0.25},
                "TZP1": {"PAO.EnergyShift": "0.005 Ry", "PAO.SplitNorm": 0.25},
                "TZ1P": {"PAO.EnergyShift": "0.005 Ry", "PAO.SplitNorm": 0.25},
                "TZDP": {"PAO.EnergyShift": "0.005 Ry", "PAO.SplitNorm": 0.25},
                "TZP2": {"PAO.EnergyShift": "0.005 Ry", "PAO.SplitNorm": 0.25},
                "TZ2P": {"PAO.EnergyShift": "0.005 Ry", "PAO.SplitNorm": 0.25},
                "TZTP": {"PAO.EnergyShift": "0.005 Ry", "PAO.SplitNorm": 0.25},
                "TZP3": {"PAO.EnergyShift": "0.005 Ry", "PAO.SplitNorm": 0.25},
                "TZ3P": {"PAO.EnergyShift": "0.005 Ry", "PAO.SplitNorm": 0.25},
            }
            return cast(
                dict[str, str | float],
                basis_params.get(
                    basis, {"PAO.EnergyShift": "0.01 Ry", "PAO.SplitNorm": 0.15}
                ),
            )

        # For legacy strategy, params are not used
        return {}

    def make(
        self,
        structure: Structure | Molecule,
        prev_dir: str | Path | None = None,
    ) -> Flow:
        """
        Create a flow with SCF calculations for different basis sizes.

        Parameters
        ----------
        structure : Structure | Molecule
            The structure to calculate
        prev_dir : str | Path | None
            Previous calculation directory

        Returns
        -------
        Flow
            A jobflow Flow containing all basis size calculations
        """
        from atomate2.siesta.utils.common import print_docstring_in_box

        print_docstring_in_box(self.__doc__, title=self.__class__.__name__)

        jobs = []
        allowed_basis_size = self._get_basis_sizes()

        for basis in allowed_basis_size:
            if self.strategy == "legacy":
                # Use legacy interface
                scf_maker_basis_job = self.static_maker.scf(
                    basis_set_size=f"{basis}"
                ).make(structure, prev_dir=None)
            else:
                # Use modern interface with parameter updates
                maker = self.static_maker.scf()
                self.propagate_custodian_to_maker(maker)
                basis_params = self._get_basis_params(basis)
                siesta_updates = {
                    "PAO.BasisSize": basis,
                    "PAO.BasisType": "split",
                    **basis_params,
                }
                maker = update_user_siesta_settings(maker, siesta_updates)
                maker = update_user_siesta_settings(maker, {"a2s_kpts": [3, 3, 3]})
                scf_maker_basis_job = maker.make(structure, prev_dir=None)

            scf_maker_basis_job.name += f"-{basis}"
            jobs.append(scf_maker_basis_job)

        return Flow(jobs, name=self.name)


# Backward compatibility aliases
# These classes are deprecated. Use DifferentBasisSCF with strategy parameter instead.


def DifferentBasisSCFAdvance(*args, **kwargs):
    """
    Deprecated: Use DifferentBasisSCF(strategy="advanced") instead.

    This function provides backward compatibility for code using the old
    DifferentBasisSCFAdvance class.
    """
    import warnings

    warnings.warn(
        "DifferentBasisSCFAdvance is deprecated. Use DifferentBasisSCF(strategy='advanced') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    kwargs["strategy"] = "advanced"
    return DifferentBasisSCFFlowMaker(*args, **kwargs)


def DifferentBasisSCFOld(*args, **kwargs):
    """
    Deprecated: Use DifferentBasisSCF(strategy="legacy") instead.

    This function provides backward compatibility for code using the old
    DifferentBasisSCFOld class.
    """
    import warnings

    warnings.warn(
        "DifferentBasisSCFOld is deprecated. Use DifferentBasisSCF(strategy='legacy') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    kwargs["strategy"] = "legacy"
    return DifferentBasisSCFFlowMaker(*args, **kwargs)


@dataclass
class DifferentBasisFlowMaker(BaseSiestaFlowMaker):
    """Double relaxation maker for SIESTA.

    A maker to perform a double relaxation in SIESTA (first with light,
    and then with tight species_defaults).

    Parameters
    ----------
    name : str
        A name for the flow
    relax_maker1: .BaseAimsMaker
        A maker that generates the first relaxation
    relax_maker2: .BaseAimsMaker
        A maker that generates the second relaxation
    """

    name: str = "Different basis"
    relax_maker_sz: BaseSiestaMaker = field(
        default_factory=lambda: RelaxMaker(
            input_set_generator=SiestaInputGenerator(basis_set_size="SZ")
        )
    )

    relax_maker_szp: BaseSiestaMaker = field(
        default_factory=lambda: RelaxMaker(
            input_set_generator=SiestaInputGenerator(basis_set_size="SZP")
        )
    )

    relax_maker_dz: BaseSiestaMaker = field(
        default_factory=lambda: RelaxMaker(
            input_set_generator=SiestaInputGenerator(basis_set_size="DZ")
        )
    )

    relax_maker_dzp: BaseSiestaMaker = field(
        default_factory=lambda: RelaxMaker(
            input_set_generator=SiestaInputGenerator(basis_set_size="DZP")
        )
    )

    relax_maker_tz: BaseSiestaMaker = field(
        default_factory=lambda: RelaxMaker(
            input_set_generator=SiestaInputGenerator(basis_set_size="TZ")
        )
    )

    relax_maker_tzp: BaseSiestaMaker = field(
        default_factory=lambda: RelaxMaker(
            input_set_generator=SiestaInputGenerator(basis_set_size="TZP")
        )
    )

    def make(
        self,
        structure: Structure | Molecule,
        prev_dir: str | Path | None = None,
    ) -> Flow:
        """Create a flow with two chained relaxations.

        Parameters
        ----------
        structure : Structure or Molecule
            The structure to relax.
        prev_dir : str or Path or None
            A previous SIESTA calculation directory to copy output files from.
        """
        from atomate2.siesta.utils.common import print_docstring_in_box

        print_docstring_in_box(self.__doc__, title=self.__class__.__name__)

        relax_sz = self.relax_maker_sz.make(structure, prev_dir=prev_dir)
        relax_sz.name += "-SZ"

        relax_szp = self.relax_maker_szp.make(structure, prev_dir=prev_dir)
        relax_szp.name += "-SZP"

        relax_dz = self.relax_maker_dz.make(structure, prev_dir=prev_dir)
        relax_dz.name += "-DZ"

        relax_dzp = self.relax_maker_dzp.make(structure, prev_dir=prev_dir)
        relax_dzp.name += "-DZP"

        relax_tz = self.relax_maker_tz.make(structure, prev_dir=prev_dir)
        relax_tz.name += "-TZ"

        relax_tzp = self.relax_maker_tzp.make(structure, prev_dir=prev_dir)
        relax_tzp.name += "-TZP"

        return Flow(
            [relax_sz, relax_szp, relax_dz, relax_dzp, relax_tz, relax_tzp],
            name=self.name,
        )


@dataclass
class DifferentBasisRelaxFlowMaker(BaseSiestaFlowMaker):
    """Double relaxation maker for SIESTA with different basis sets and fixed/variable cell.

    A maker to perform relaxations in SIESTA with different basis sets (SZ, DZ, etc.)
    and optionally fixed or variable cell relaxation.
    """

    name: str = "Different basis relaxation"

    relax_maker_sz_fixed: BaseSiestaMaker = field(
        default_factory=lambda: RelaxMaker.fixed_cell_relaxation(basis_set_size="SZ")
    )

    relax_maker_sz_variable: BaseSiestaMaker = field(
        default_factory=lambda: RelaxMaker.variable_cell_relaxation(basis_set_size="SZ")
    )

    relax_maker_dz_fixed: BaseSiestaMaker = field(
        default_factory=lambda: RelaxMaker.fixed_cell_relaxation(basis_set_size="DZ")
    )

    relax_maker_dz_variable: BaseSiestaMaker = field(
        default_factory=lambda: RelaxMaker.variable_cell_relaxation(basis_set_size="DZ")
    )

    def make(
        self,
        structure: Structure | Molecule,
        prev_dir: str | Path | None = None,
    ) -> Flow:
        """Create a flow with multiple chained relaxations."""
        from atomate2.siesta.utils.common import print_docstring_in_box

        print_docstring_in_box(self.__doc__, title=self.__class__.__name__)

        # First, perform fixed-cell relaxation with SZ basis
        relax_sz_fixed = self.relax_maker_sz_fixed.make(structure, prev_dir=prev_dir)
        relax_sz_fixed.name += "-SZ-fixed-cell"

        # Then, perform variable-cell relaxation with SZ basis
        relax_sz_variable = self.relax_maker_sz_variable.make(
            structure, prev_dir=prev_dir
        )
        relax_sz_variable.name += "-SZ-variable-cell"

        # Perform fixed-cell relaxation with DZ basis
        relax_dz_fixed = self.relax_maker_dz_fixed.make(structure, prev_dir=prev_dir)
        relax_dz_fixed.name += "-DZ-fixed-cell"

        # Perform variable-cell relaxation with DZ basis
        relax_dz_variable = self.relax_maker_dz_variable.make(
            structure, prev_dir=prev_dir
        )
        relax_dz_variable.name += "-DZ-variable-cell"

        # Return the flow with all relaxations in sequence
        return Flow(
            [relax_sz_fixed, relax_sz_variable, relax_dz_fixed, relax_dz_variable],
            name=self.name,
        )
