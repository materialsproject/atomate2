import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from atomate2.settings import Atomate2Settings


def test_empty_and_invalid_config_file(clean_dir):
    # set path to load settings from though ATOMATE2_CONFIG_FILE env variable
    config_file_path = Path.cwd() / "test-atomate2-config.yaml"
    os.environ["ATOMATE2_CONFIG_FILE"] = str(config_file_path)

    settings = Atomate2Settings()
    assert str(config_file_path) == settings.CONFIG_FILE
    assert settings.SYMPREC == 0.1
    assert settings.BANDGAP_TOL == 1e-4
    assert settings.VASP_RUN_BADER is False
    assert settings.VASP_RUN_DDEC6 is False
    assert settings.DDEC6_ATOMIC_DENSITIES_DIR is None

    # test warning if config file is empty
    config_file_path.touch()
    with pytest.warns(
        UserWarning, match="Using atomate2 config file at .+ but it's empty"
    ):
        Atomate2Settings()

    # test error if the file exists and contains invalid YAML
    with open(config_file_path, "w") as file:
        file.write("invalid yaml")

    with pytest.raises(SyntaxError, match="atomate2 config file at"):
        Atomate2Settings()

    # test error if the file exists and contains invalid settings
    with open(config_file_path, "w") as file:
        file.write("VASP_CMD: 42")

    with pytest.raises(
        ValidationError,
        match="1 validation error for Atomate2Settings\nVASP_CMD\n  "
        "Input should be a valid string ",
    ):
        Atomate2Settings()

    with open(config_file_path, "w") as file:
        file.write("BANDGAP_TOL: invalid")

    with pytest.raises(
        ValidationError,
        match="1 validation error for Atomate2Settings\nBANDGAP_TOL\n  "
        "Input should be a valid number",
    ):
        Atomate2Settings()
