"""Utils for using a force field (aka an interatomic potential)."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING

from ase.units import Bohr
from monty.json import MontyDecoder
from typing_extensions import assert_never, deprecated

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from typing import Any

    try:
        from torch import dtype as torch_dtype
    except ImportError:
        torch_dtype = str

    from ase.calculators.calculator import Calculator

    from atomate2.ase.schemas import AseResult

_FORCEFIELD_DATA_OBJECTS = ["trajectory", "ionic_steps"]


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
    DeepMD = "DeepMD"
    Allegro = "Allegro"
    FAIRChem = "FAIRChem"
    MatterSim = "MatterSim"
    UPET = "UPET"

    @classmethod
    def _missing_(cls, value: Any) -> Any:
        """Allow input of str(MLFF) as valid enum."""
        if isinstance(value, str):
            value = value.split("MLFF.")[-1]
        for member in cls:
            if member.name == value:
                return member
        return None


_DEFAULT_CALCULATOR_KWARGS: dict[MLFF, Any] = {
    MLFF.CHGNet: {"stress_unit": "eV/A3"},
    MLFF.FAIRChem: {
        "predict_unit": {"model_name": "uma-s-1p1"},
        "task_name": "omat",
    },
    MLFF.GAP: {"args_str": "IP GAP", "param_filename": "gap.xml"},
    MLFF.M3GNet: {"stress_unit": "eV/A3"},
    MLFF.MACE: {"model": "medium"},
    MLFF.MACE_MP_0: {"model": "medium"},
    MLFF.MACE_MP_0B3: {"model": "medium-0b3"},
    MLFF.MACE_MPA_0: {"model": "medium-mpa-0"},
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
    MLFF.NEP: {"model_filename": "nep.txt"},
    MLFF.SevenNet: {"model": "7net-0"},
    MLFF.UPET: {
        "model": "pet-mad-s",
        "version": "1.5.0",
    },
}


def _get_standardized_mlff(force_field_name: str | MLFF) -> MLFF:
    """Get the standardized force field name.

    Parameters
    ----------
    force_field_name : str or .MLFF
        The name of the force field
        For str, accept both with and without the `MLFF.` prefix.

    Returns
    -------
    MLFF: the name of the forcefield
    """
    if isinstance(force_field_name, str):
        # ensure `force_field_name` uses enum format
        if force_field_name.startswith("MLFF."):
            force_field_name = force_field_name.split("MLFF.")[-1]

        if force_field_name in MLFF.__members__:
            force_field_name = MLFF[force_field_name]
        elif force_field_name in [v.value for v in MLFF]:
            force_field_name = MLFF(force_field_name)
        else:
            raise ValueError(
                f"force_field_name={force_field_name} is not a valid MLFF name."
            )

    if force_field_name == MLFF.MACE:
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


@deprecated("Use _get_standardized_mlff instead.")
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
    force_field_name = _get_standardized_mlff(force_field_name)
    return str(force_field_name)


@dataclass
class ForceFieldMixin:
    """Mix-in class for force-fields.

    Attributes
    ----------
    force_field_name : str or MLFF
        Name of the forcefield which will be
        correctly deserialized/standardized if the forcefield is
        a known `MLFF`.
    calculator_meta : MLFF or dict
        Actual metadata to instantiate the ASE calculator.
    calculator_kwargs : dict = field(default_factory=dict)
        Keyword arguments that will get passed to the ASE calculator.
    task_document_kwargs: dict = field(default_factory=dict)
        Additional keyword args passed to :obj:`.ForceFieldTaskDocument()
        or another final document schema.
    """

    force_field_name: str | MLFF | dict = MLFF.Forcefield
    calculator_meta: MLFF | dict = field(init=False)
    calculator_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Ensure that force_field_name is correctly assigned."""
        if hasattr(super(), "__post_init__"):
            super().__post_init__()  # type: ignore[misc]

        if isinstance(self.force_field_name, dict):
            mlff = MLFF.Forcefield  # Fallback to placeholder
            self.calculator_meta = self.force_field_name.copy()
        else:
            mlff = _get_standardized_mlff(self.force_field_name)
            self.calculator_meta = mlff

        self.force_field_name: str = str(mlff)  # Narrow-down type for mypy

        # Pad calculator_kwargs with default values, but permit user to override them
        self.calculator_kwargs = {
            **_DEFAULT_CALCULATOR_KWARGS.get(mlff, {}),
            **self.calculator_kwargs,
        }

        if not self.task_document_kwargs.get("force_field_name"):
            self.task_document_kwargs["force_field_name"] = self.force_field_name

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
            self.calculator_meta,
            **self.calculator_kwargs,
        )

    @property
    def mlff(self) -> MLFF:
        """The MLFF enum corresponding to the force field name."""
        return MLFF(str(self.force_field_name).split("MLFF.")[-1])

    @cached_property
    def ase_calculator_name(self) -> str:
        """The name of the ASE calculator for schemas."""
        if isinstance(self.calculator_meta, MLFF):
            return str(self.force_field_name)
        if isinstance(self.calculator_meta, dict):
            calc_cls = _load_calc_cls(self.calculator_meta)
            return calc_cls.__name__
        assert_never(self.calculator_meta)


def ase_calculator(
    calculator_meta: str | MLFF | dict,
    default_dtype: str | torch_dtype | None = None,
    **kwargs: Any,
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
    default_dtype (str or pytorch dtype) : optional pytorch dtype to use if applicable
    kwargs : optional kwargs to pass to a calculator

    Returns
    -------
    ASE .Calculator
    """
    calculator = None

    if (
        isinstance(calculator_meta, str)
        and (calculator_meta in map(str, MLFF) or calculator_meta in MLFF)
    ) or isinstance(calculator_meta, MLFF):
        calculator_name = MLFF(calculator_meta)

        match calculator_name:
            # Simple APIs
            case (
                MLFF.DeepMD
                | MLFF.GAP
                | MLFF.MatterSim
                | MLFF.NEP
                | MLFF.SevenNet
                | MLFF.UPET
            ):
                import_str = {
                    MLFF.DeepMD: "deepmd.calculator.DP",
                    MLFF.GAP: "quippy.potential.Potential",
                    MLFF.MatterSim: "mattersim.forcefield.MatterSimCalculator",
                    MLFF.NEP: "calorine.calculators.CPUNEP",
                    MLFF.SevenNet: "sevenn.sevennet_calculator.SevenNetCalculator",
                    MLFF.UPET: "upet.calculator.UPETCalculator",
                }
                _mod, _cls = import_str[calculator_name].rsplit(".", 1)
                calculator = getattr(import_module(_mod), _cls, None)(**kwargs)

            case MLFF.CHGNet | MLFF.M3GNet | MLFF.MATPES_R2SCAN | MLFF.MATPES_PBE:
                if calculator_name == MLFF.CHGNet:
                    # Legacy interface to `chgnet` package
                    try:
                        from chgnet.model.dynamics import CHGNetCalculator

                        return CHGNetCalculator(**kwargs)
                    except ImportError:
                        pass

                import matgl

                match calculator_name:
                    case MLFF.M3GNet:
                        path = kwargs.get("path", "M3GNet-MP-2021.2.8-PES")
                        matgl.config.BACKEND = "DGL"
                    case MLFF.CHGNet:
                        path = kwargs.get("path", "CHGNet-MPtrj-2023.12.1-2.7M-PES")
                        matgl.config.BACKEND = "DGL"

                        warnings.warn(
                            "The CHGNet functionality in atomate2 has been migrated "
                            "from the `chgnet` package to `matgl` to ensure continuing "
                            "support. If you want to use the `chgnet` package, "
                            "`pip install chgnet`",
                            stacklevel=2,
                        )
                    case MLFF.MATPES_R2SCAN | MLFF.MATPES_PBE:
                        path = (
                            f"{kwargs.pop('architecture', 'TensorNet')}"
                            f"-{calculator_name.value}"
                            f"-v{kwargs.pop('version', '2025.1')}"
                            "-PES"
                        )
                        matgl.config.BACKEND = "PYG"

                if default_dtype is not None:
                    matgl.set_default_dtype(default_dtype)

                matgl_calc = getattr(
                    import_module(f"matgl.ext._ase_{matgl.config.BACKEND.lower()}"),
                    "PESCalculator",
                    None,
                )
                calculator = matgl_calc(matgl.load_model(path), **kwargs)

            case MLFF.MACE | MLFF.MACE_MP_0 | MLFF.MACE_MPA_0 | MLFF.MACE_MP_0B3:
                from mace.calculators import MACECalculator, mace_mp

                model = kwargs.get("model")
                if isinstance(model, str | Path) and Path(model).exists():
                    model_path = model
                    device = kwargs.pop("device", None) or "cpu"
                    kwargs.pop("device", None)
                    calculator = MACECalculator(
                        model_paths=model_path,
                        device=device,
                        default_dtype=default_dtype or "",
                        **kwargs,
                    )

                    if kwargs.get("dispersion", False):
                        # See https://github.com/materialsproject/atomate2/issues/1262
                        # Specifying an explicit model path unsets the dispersio
                        # Reset it here.
                        import torch
                        from ase.calculators.mixing import SumCalculator
                        from torch_dftd.torch_dftd3_calculator import (
                            TorchDFTD3Calculator,
                        )

                        default_d3_kwargs = {
                            "damping": "bj",
                            "xc": "pbe",
                            "cutoff": 40.0 * Bohr,
                            "dtype": default_dtype or torch.get_default_dtype(),
                        }
                        kwargs.update(
                            {
                                k: v
                                for k, v in default_d3_kwargs.items()
                                if k not in kwargs
                            }
                        )

                        d3_calc = TorchDFTD3Calculator(device=device, **kwargs)
                        calculator = SumCalculator([calculator, d3_calc])
                else:
                    calculator = mace_mp(default_dtype=default_dtype or "", **kwargs)

            case MLFF.Nequip | MLFF.Allegro:
                from nequip.integrations.ase import NequIPCalculator

                calculator = getattr(
                    NequIPCalculator,
                    "from_compiled_model"
                    if hasattr(NequIPCalculator, "from_compiled_model")
                    else "from_deployed_model",
                )(**kwargs)

            case MLFF.FAIRChem:
                from fairchem.core import FAIRChemCalculator, pretrained_mlip

                predict_unit_kwargs = kwargs.pop(
                    "predict_unit",
                    _DEFAULT_CALCULATOR_KWARGS[MLFF.FAIRChem]["predict_unit"],
                )
                calculator = FAIRChemCalculator(
                    pretrained_mlip.get_predict_unit(**predict_unit_kwargs),
                    **{k: v for k, v in kwargs.items() if k != "predict_unit"},
                )

    elif isinstance(calculator_meta, dict):
        calc_cls = _load_calc_cls(calculator_meta)
        calculator = calc_cls(**kwargs)

    if calculator is None:
        raise ValueError(f"Could not create ASE calculator for {calculator_meta}.")

    return calculator


def _load_calc_cls(
    calculator_meta: dict,
) -> type[Calculator] | Callable[..., Calculator]:
    return MontyDecoder().process_decoded(calculator_meta)


@contextmanager
def revert_default_dtype() -> Generator[None]:
    """Context manager for torch.default_dtype.

    Reverts it to whatever torch.get_default_dtype() was when entering the context.

    Originally added for use with MACE(Relax|Static)Maker.
    https://github.com/ACEsuit/mace/issues/328
    """
    import torch

    orig = torch.get_default_dtype()
    yield
    torch.set_default_dtype(orig)


def _get_pkg_name(calculator_meta: MLFF | dict[str, Any]) -> str | None:
    """Get the package name for a given force field.

    Parameters
    ----------
    calculator_meta : MLFF or JSONable dict
        The calculator metadata used to load the calculator,
        or an MLFF enum.

    Returns
    -------
    str or None: The package name of the force field if it could be identified,
        None otherwise.
    """
    if isinstance(calculator_meta, MLFF):
        # map force field name to its package name
        match calculator_meta:
            case MLFF.Allegro | MLFF.Nequip:
                ff_pkg = "nequip"
            case MLFF.CHGNet:
                # Check if CHGNet is installed
                try:
                    ff_pkg = next(pkg for pkg in ("chgnet", "matgl") if find_spec(pkg))
                except StopIteration:
                    ff_pkg = None
            case MLFF.M3GNet | MLFF.MATPES_PBE | MLFF.MATPES_R2SCAN:
                ff_pkg = "matgl"
            case MLFF.DeepMD:
                ff_pkg = "deepmd-kit"
            case MLFF.FAIRChem:
                ff_pkg = "fairchem.core"
            case MLFF.GAP:
                ff_pkg = "quippy-ase"
            case MLFF.MACE | MLFF.MACE_MP_0 | MLFF.MACE_MPA_0 | MLFF.MACE_MP_0B3:
                ff_pkg = "mace-torch"
            case MLFF.MatterSim:
                ff_pkg = "mattersim"
            case MLFF.NEP:
                ff_pkg = "calorine"
            case MLFF.SevenNet:
                ff_pkg = "sevenn"
            case MLFF.UPET:
                ff_pkg = "upet"
            case _:
                ff_pkg = None
        return ff_pkg
    if isinstance(calculator_meta, dict):
        calc_cls = _load_calc_cls(calculator_meta)
        return calc_cls.__module__.split(".", 1)[0]
    assert_never(calculator_meta)


def _get_pkg_version(calculator_meta: MLFF | dict[str, Any]) -> str | None:
    """Try to establish the imported version of a forcefield python package."""
    if isinstance(pkg_name := _get_pkg_name(calculator_meta), str):
        try:
            return version(pkg_name)
        except PackageNotFoundError:
            try:
                return getattr(import_module(pkg_name), "__version__", None)
            except ImportError:
                pass
    return None
