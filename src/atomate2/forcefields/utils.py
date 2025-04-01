"""Utils for using a force field (aka an interatomic potential)."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from monty.json import MontyDecoder

from atomate2.forcefields import MLFF

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any

    from ase.calculators.calculator import Calculator


def ase_calculator(calculator_meta: str | dict, **kwargs: Any) -> Calculator | None:
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

    if isinstance(calculator_meta, str | MLFF) and calculator_meta in map(str, MLFF):
        calculator_name = MLFF[calculator_meta.split("MLFF.")[-1]]

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
                device = kwargs.get("device") or "cpu"
                if "device" in kwargs:
                    del kwargs["device"]
                calculator = MACECalculator(
                    model_paths=model_path,
                    device=device,
                    **kwargs,
                )
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
