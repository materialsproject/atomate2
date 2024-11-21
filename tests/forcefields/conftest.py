from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from typing import Any


def pytest_runtest_setup(item: Any) -> None:
    # MACE changes the default dtype, ensure consistent dtype here
    torch.set_default_dtype(torch.float32)
    # For consistent performance across hardware, explicitly set device to CPU
    torch.set_default_device("cpu")
