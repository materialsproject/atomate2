"""Utils for using a force field (aka an interatomic potential)."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from ase.io import Trajectory as AseTrajectory
from ase.units import Bohr
from ase.units import GPa as _GPa_to_eV_per_A3
from monty.json import MontyDecoder
from pymatgen.core.trajectory import Trajectory as PmgTrajectory

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any

    from ase.calculators.calculator import Calculator

    from atomate2.ase.schemas import AseResult

_FORCEFIELD_DATA_OBJECTS = [PmgTrajectory, AseTrajectory, "ionic_steps"]


class MLFF(Enum):  # TODO inherit from StrEnum when 3.11+
    """Names of ML force fields."""

    MACE = "MACE"  # This is MACE-MP-0 (medium), deprecated
    MACE_MP_0 = "MACE-MP-0"
    MACE_MPA_0 = "MACE-MPA-0"
    MACE_MP_0B3 = "MACE-MP-0b3"
    GAP = "GAP"
    M3GNet = "M3GNet"
    CHGNet = "CHGNet"
    Forcefield = "Forcefield"  # default placeholder option
    NEP = "NEP"
    Nequip = "Nequip"
    SevenNet = "SevenNet"
    MATPES_R2SCAN = "MatPES-r2SCAN"
    MATPES_PBE = "MatPES-PBE"

    @classmethod
    def _missing_(cls, value: Any) -> Any:
        """Allow input of str(MLFF) as valid enum."""
        if isinstance(value, str):
            value = value.split("MLFF.")[-1]
        for member in cls:
            if member.name == value:
                return member
        return None


_DEFAULT_CALCULATOR_KWARGS = {
    MLFF.CHGNet: {"stress_weight": _GPa_to_eV_per_A3},
    MLFF.M3GNet: {"stress_weight": _GPa_to_eV_per_A3},
    MLFF.NEP: {"model_filename": "nep.txt"},
    MLFF.GAP: {"args_str": "IP GAP", "param_filename": "gap.xml"},
    MLFF.MACE: {"model": "medium"},
    MLFF.MACE_MP_0: {"model": "medium"},
    MLFF.MACE_MPA_0: {"model": "medium-mpa-0"},
    MLFF.MACE_MP_0B3: {"model": "medium-0b3"},
    MLFF.MATPES_PBE: {
        "architecture": "TensorNet",
        "version": "2025.1",
        "stress_unit": "eV/A3",
    },
    MLFF.MATPES_R2SCAN: {
        "architecture": "TensorNet",
        "version": "2025.1",
        "stress_unit": "eV/A3",
    },
}


def _get_formatted_ff_name(force_field_name: str | MLFF) -> str:
    """
    Get the standardized force field name.

    Parameters
    ----------
    force_field_name : str or .MLFF
        The name of the force field

    Returns
    -------
    str : the name of the forcefield from MLFF
    """
    if isinstance(force_field_name, str):
        # ensure `force_field_name` uses enum format
        if force_field_name in MLFF.__members__:
            force_field_name = MLFF[force_field_name]
        elif force_field_name in [v.value for v in MLFF]:
            force_field_name = MLFF(force_field_name)
    force_field_name = str(force_field_name)
    if force_field_name in {"MLFF.MACE", "MACE"}:
        warnings.warn(
            "Because the default MP-trained MACE model is constantly evolving, "
            "we no longer recommend using `MACE` or `MLFF.MACE` to specify "
            "a MACE model. For reproducibility purposes, specifying `MACE` "
            "will still default to MACE-MP-0 (medium), which is identical to "
            "specifying `MLFF.MACE_MP_0`.",
            category=UserWarning,
            stacklevel=2,
        )
    return force_field_name


@dataclass
class ForceFieldMixin:
    """Mix-in class for force-fields.

    Attributes
    ----------
    force_field_name : str or MLFF
        Name of the forcefield which will be
        correctly deserialized/standardized if the forcefield is
        a known `MLFF`.
    calculator_kwargs : dict = field(default_factory=dict)
        Keyword arguments that will get passed to the ASE calculator.
    task_document_kwargs: dict = field(default_factory=dict)
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()
        or another final document schema.
    """

    force_field_name: str | MLFF = MLFF.Forcefield
    calculator_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Ensure that force_field_name is correctly assigned."""
        if hasattr(super(), "__post_init__"):
            super().__post_init__()  # type: ignore[misc]

        self.force_field_name = _get_formatted_ff_name(self.force_field_name)

        # Pad calculator_kwargs with default values, but permit user to override them
        self.calculator_kwargs = {
            **_DEFAULT_CALCULATOR_KWARGS.get(
                MLFF(self.force_field_name.split("MLFF.")[-1]), {}
            ),
            **self.calculator_kwargs,
        }

        if not self.task_document_kwargs.get("force_field_name"):
            self.task_document_kwargs["force_field_name"] = str(self.force_field_name)

    def _run_ase_safe(self, *args, **kwargs) -> AseResult:
        if not hasattr(self, "run_ase"):
            raise NotImplementedError(
                "You must implement a `run_ase` method to use this method."
            )
        with revert_default_dtype():
            return self.run_ase(*args, **kwargs)

    @property
    def calculator(self) -> Calculator:
        """ASE calculator, can be overwritten by user."""
        return ase_calculator(
            str(self.force_field_name),  # make mypy happy
            **self.calculator_kwargs,
        )


def ase_calculator(
    calculator_meta: str | MLFF | dict, **kwargs: Any
) -> Calculator | None:
    """
    Create an ASE calculator from a given set of metadata.

    Parameters
    ----------
    calculator_meta : str or dict
        If a str, should be one of `atomate2.forcefields.MLFF`.
        If a dict, should be decodable by `monty.json.MontyDecoder`.
        For example, one can also call the CHGNet calculator as follows
        ```
            calculator_meta = {
                "@module": "chgnet.model.dynamics",
                "@callable": "CHGNetCalculator"
            }
        ```
    args : optional args to pass to a calculator
    kwargs : optional kwargs to pass to a calculator

    Returns
    -------
    ASE .Calculator
    """
    calculator = None

    if (
        isinstance(calculator_meta, str) and calculator_meta in map(str, MLFF)
    ) or isinstance(calculator_meta, MLFF):
        calculator_name = MLFF(calculator_meta)

        if calculator_name == MLFF.CHGNet:
            from chgnet.model.dynamics import CHGNetCalculator

            calculator = CHGNetCalculator(**kwargs)

        elif calculator_name in (MLFF.M3GNet, MLFF.MATPES_R2SCAN, MLFF.MATPES_PBE):
            import matgl
            from matgl.ext.ase import PESCalculator

            if calculator_name == MLFF.M3GNet:
                path = kwargs.get("path", "M3GNet-MP-2021.2.8-PES")
            elif calculator_name in (MLFF.MATPES_R2SCAN, MLFF.MATPES_PBE):
                architecture = kwargs.pop("architecture", "TensorNet")
                matpes_version = kwargs.pop("version", "2025.1")
                path = f"{architecture}-{calculator_name.value}-v{matpes_version}-PES"

            potential = matgl.load_model(path)
            calculator = PESCalculator(potential, **kwargs)

        elif calculator_name in map(
            MLFF, ("MACE", "MACE-MP-0", "MACE-MPA-0", "MACE-MP-0b3")
        ):
            from mace.calculators import MACECalculator, mace_mp

            model = kwargs.get("model")
            if isinstance(model, str | Path) and Path(model).exists():
                model_path = model
                device = kwargs.pop("device", None) or "cpu"
                if "device" in kwargs:
                    del kwargs["device"]
                calculator = MACECalculator(
                    model_paths=model_path,
                    device=device,
                    **kwargs,
                )

                if kwargs.get("dispersion", False):
                    # See https://github.com/materialsproject/atomate2/issues/1262
                    # Specifying an explicit model path unsets the dispersio
                    # Reset it here.
                    import torch
                    from ase.calculators.mixing import SumCalculator
                    from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

                    default_d3_kwargs = {
                        "damping": "bj",
                        "xc": "pbe",
                        "cutoff": 40.0 * Bohr,
                        "dtype": kwargs.get("default_dtype", torch.get_default_dtype()),
                    }
                    for k, v in default_d3_kwargs.items():
                        if k not in kwargs:
                            kwargs[k] = v

                    d3_calc = TorchDFTD3Calculator(device=device, **kwargs)
                    calculator = SumCalculator([calculator, d3_calc])
            else:
                calculator = mace_mp(**kwargs)

        elif calculator_name == MLFF.GAP:
            from quippy.potential import Potential

            calculator = Potential(**kwargs)

        elif calculator_name == MLFF.NEP:
            from calorine.calculators import CPUNEP

            calculator = CPUNEP(**kwargs)

        elif calculator_name == MLFF.Nequip:
            from nequip.ase import NequIPCalculator

            calculator = NequIPCalculator.from_deployed_model(**kwargs)

        elif calculator_name == MLFF.SevenNet:
            from sevenn.sevennet_calculator import SevenNetCalculator

            calculator = SevenNetCalculator(**{"model": "7net-0"} | kwargs)

    elif isinstance(calculator_meta, dict):
        calc_cls = MontyDecoder().process_decoded(calculator_meta)
        calculator = calc_cls(**kwargs)

    if calculator is None:
        raise ValueError(f"Could not create ASE calculator for {calculator_meta}.")

    return calculator


@contextmanager
def revert_default_dtype() -> Generator[None, None, None]:
    """Context manager for torch.default_dtype.

    Reverts it to whatever torch.get_default_dtype() was when entering the context.

    Originally added for use with MACE(Relax|Static)Maker.
    https://github.com/ACEsuit/mace/issues/328
    """
    import torch

    orig = torch.get_default_dtype()
    yield
    torch.set_default_dtype(orig)
