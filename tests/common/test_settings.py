import pytest
from pydantic import ValidationError


def test_empty_and_invalid_config_file(clean_dir):
    import os
    from pathlib import Path

    from atomate2.settings import Atomate2Settings

    # set path to load settings from though ATOMATE2_CONFIG_FILE env variable
    config_file_path = Path.cwd() / "test-atomate2-config.yaml"
    os.environ["ATOMATE2_CONFIG_FILE"] = str(config_file_path)

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
