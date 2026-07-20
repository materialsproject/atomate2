"""Makers for defect calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomate2.siesta.jobs.core import RelaxMaker, StaticMaker
from atomate2.siesta.sets.core import RelaxSetGenerator, StaticSetGenerator

if TYPE_CHECKING:
    from atomate2.siesta.sets.base import SiestaInputGenerator

logger = logging.getLogger(__name__)


@dataclass
class DefectStaticMaker(StaticMaker):
    """
    SIESTA Static calculation for defect supercells.

    Specialized StaticMaker for defect calculations with sensible defaults:
    - Higher k-point density for accurate defect energies
    - Tighter SCF convergence for charged systems
    - Electronic density mixing optimized for defects

    This is essentially a wrapper around StaticMaker with defect-specific
    parameter recommendations.

    Parameters
    ----------
    calc_type : str
        The type key for the calculation (default: "defect_scf")
    name : str
        The job name (default: "Defect Static Calculation")
    input_set_generator : SiestaInputGenerator
        The InputGenerator for the calculation (default: StaticSetGenerator)

    Examples
    --------
    >>> from atomate2.siesta.flows.defects.makers import DefectStaticMaker
    >>> from pymatgen.core import Structure
    >>> defect_structure = Structure.from_file("defect_supercell.cif")
    >>> maker = DefectStaticMaker.defect_scf()
    >>> job = maker.make(defect_structure)
    """

    input_set_generator: SiestaInputGenerator = field(
        default_factory=StaticSetGenerator
    )
    calc_type: str = "defect_scf"
    name: str = "Defect Static Calculation"

    @classmethod
    def defect_scf(cls, *args, **kwargs) -> DefectStaticMaker:
        """
        Create a defect SCF maker with recommended settings.

        Recommended parameters for defect calculations:
        - DM.MixingWeight: 0.1 (conservative mixing for charged systems)
        - DM.NumberPulay: 8 (improved convergence)
        - DM.Tolerance: 1e-5 (tighter tolerance)

        Parameters
        ----------
        *args
            Positional arguments for StaticSetGenerator
        **kwargs
            Keyword arguments for maker and input generator

        Returns
        -------
        DefectStaticMaker
            Configured maker for defect SCF calculations
        """
        logger.info("DefectStaticMaker.defect_scf()")

        # Separate maker kwargs from input generator kwargs
        maker_kwargs = {}
        input_gen_kwargs = {}
        maker_fields = {
            "use_custodian",
            "custodian_handlers",
            "custodian_max_errors",
            "write_input_set_kwargs",
            "copy_siesta_kwargs",
            "run_siesta_kwargs",
            "task_document_kwargs",
            "stop_children_kwargs",
            "write_additional_data",
            "store_output_data",
            "dry_run",
            "dry_run_output_dir",
            "dry_run_format",
            "dry_run_label",
        }

        for key, value in kwargs.items():
            if key in maker_fields:
                maker_kwargs[key] = value
            else:
                input_gen_kwargs[key] = value

        # Add defect-specific defaults if not provided
        if "user_params" not in input_gen_kwargs:
            input_gen_kwargs["user_params"] = {}

        defect_defaults = {
            "Spin": "polarized",  # Defects commonly have unpaired electrons
            "SCF.Mixer.Weight": 0.1,  # Conservative mixing
            "SCF.Mixer.History": 8,  # Better convergence (Pulay history)
            # Tighter tolerance (legacy alias for SCF.DM.Tolerance)
            "DM.Tolerance": 1e-5,
        }

        # Only add defaults that aren't already specified
        for key, value in defect_defaults.items():
            if key not in input_gen_kwargs["user_params"]:
                input_gen_kwargs["user_params"][key] = value

        return cls(
            input_set_generator=StaticSetGenerator(*args, **input_gen_kwargs),
            name="Defect Static Calculation",
            **maker_kwargs,
        )


@dataclass
class DefectRelaxMaker(RelaxMaker):
    """
    SIESTA Relaxation calculation for defect supercells.

    Specialized RelaxMaker for defect calculations with sensible defaults:
    - Fixed-cell relaxation (defects created in fixed supercells)
    - Tighter force tolerance for accurate defect geometries
    - Conservative mixing for charged systems

    This is essentially a wrapper around RelaxMaker with defect-specific
    parameter recommendations.

    Parameters
    ----------
    calc_type : str
        The type key for the calculation (default: "defect_relax")
    name : str
        The job name (default: "Defect Relaxation")
    input_set_generator : SiestaInputGenerator
        The InputGenerator for the calculation (default: RelaxSetGenerator)

    Examples
    --------
    >>> from atomate2.siesta.flows.defects.makers import DefectRelaxMaker
    >>> from pymatgen.core import Structure
    >>> defect_structure = Structure.from_file("defect_supercell.cif")
    >>> maker = DefectRelaxMaker.defect_relax()
    >>> job = maker.make(defect_structure)
    """

    input_set_generator: SiestaInputGenerator = field(default_factory=RelaxSetGenerator)
    calc_type: str = "defect_relax"
    name: str = "Defect Relaxation"

    @classmethod
    def defect_relax(cls, *args, **kwargs) -> DefectRelaxMaker:
        """
        Create a defect relaxation maker with recommended settings.

        Recommended parameters for defect relaxation:
        - Fixed-cell relaxation (relax_cell=False)
        - MD.MaxForceTol: 0.02 eV/Ang (tighter than default)
        - DM.MixingWeight: 0.1 (conservative mixing for charged systems)
        - DM.NumberPulay: 8 (improved convergence)

        Parameters
        ----------
        *args
            Positional arguments for RelaxSetGenerator
        **kwargs
            Keyword arguments for maker and input generator

        Returns
        -------
        DefectRelaxMaker
            Configured maker for defect relaxation calculations
        """
        logger.info("DefectRelaxMaker.defect_relax()")

        # Separate maker kwargs from input generator kwargs
        maker_kwargs = {}
        input_gen_kwargs = {}
        maker_fields = {
            "use_custodian",
            "custodian_handlers",
            "custodian_max_errors",
            "write_input_set_kwargs",
            "copy_siesta_kwargs",
            "run_siesta_kwargs",
            "task_document_kwargs",
            "stop_children_kwargs",
            "write_additional_data",
            "store_output_data",
            "dry_run",
            "dry_run_output_dir",
            "dry_run_format",
            "dry_run_label",
        }

        for key, value in kwargs.items():
            if key in maker_fields:
                maker_kwargs[key] = value
            else:
                input_gen_kwargs[key] = value

        # Add defect-specific defaults if not provided
        if "user_params" not in input_gen_kwargs:
            input_gen_kwargs["user_params"] = {}

        defect_defaults = {
            "Spin": "polarized",  # Defects commonly have unpaired electrons
            "MD.MaxForceTol": "0.02 eV/Ang",  # Tighter than default 0.04
            "SCF.Mixer.Weight": 0.1,  # Conservative mixing
            "SCF.Mixer.History": 8,  # Better convergence (Pulay history)
            # Tighter tolerance (legacy alias for SCF.DM.Tolerance)
            "DM.Tolerance": 1e-5,
        }

        # Only add defaults that aren't already specified
        for key, value in defect_defaults.items():
            if key not in input_gen_kwargs["user_params"]:
                input_gen_kwargs["user_params"][key] = value

        return cls(
            input_set_generator=RelaxSetGenerator(
                *args,
                relax_cell=False,
                **input_gen_kwargs,  # Fixed-cell!
            ),
            name="Defect Relaxation",
            **maker_kwargs,
        )
