from typing import Any

import pytest
from pymatgen.io.vasp import Kpoints

from atomate2.cp2k import powerups
from atomate2.cp2k.flows.core import DoubleRelaxMaker
from atomate2.cp2k.jobs.core import RelaxMaker


@pytest.mark.parametrize(
    "powerup,attribute,settings",
    [
        ("update_user_input_settings", "user_input_settings", {"max_scf": 1}),
        (
            "update_user_kpoints_settings",
            "user_kpoints_settings",
            {"reciprocal_density": 100},
        ),
    ],
)
def test_update_user_settings(
    powerup: str, attribute: str, settings: dict[str, Any]
) -> None:
    """Test basic functionality of user settings updates."""
    powerup_func = getattr(powerups, powerup)
    test_cases = [
        ("maker", RelaxMaker()),
        ("job", RelaxMaker().make()),
        ("flow_maker", DoubleRelaxMaker()),
        ("flow", DoubleRelaxMaker().make(1)),
    ]

    for case_name, obj in test_cases:
        updated = powerup_func(obj, settings)
        if case_name == "maker":
            assert getattr(updated.input_set_generator, attribute) == settings
        elif case_name == "job":
            assert (
                getattr(updated.function.__self__.input_set_generator, attribute)
                == settings
            )
        elif case_name == "flow_maker":
            for maker in (updated.relax_maker1, updated.relax_maker2):
                assert getattr(maker.input_set_generator, attribute) == settings
        else:  # flow
            for job in updated.jobs:
                assert (
                    getattr(job.function.__self__.input_set_generator, attribute)
                    == settings
                )


def test_update_user_input_settings_nested() -> None:
    """Test nested dictionary updates for user input settings."""
    nested_settings = {
        "FORCE_EVAL": {"DFT": {"SCF": {"MAX_SCF": 100}, "QS": {"METHOD": "GPW"}}}
    }

    flow = DoubleRelaxMaker().make(1)
    updated_flow = powerups.update_user_input_settings(flow, nested_settings)
    input_gen = updated_flow.jobs[0].function.__self__.input_set_generator

    assert input_gen.user_input_settings["FORCE_EVAL"]["DFT"]["SCF"]["MAX_SCF"] == 100
    assert input_gen.user_input_settings["FORCE_EVAL"]["DFT"]["QS"]["METHOD"] == "GPW"


@pytest.mark.parametrize(
    "kpoints_input",
    [
        {"reciprocal_density": 100},
        Kpoints.automatic(100),
        {"kpoints_mode": "line", "kpts": [[0, 0, 0], [0.5, 0.5, 0.5]]},
    ],
)
def test_update_user_kpoints_settings_input_types(
    kpoints_input: dict[str, Any] | Kpoints,
) -> None:
    """Test different input types for kpoints settings updates."""
    flow = DoubleRelaxMaker().make(1)
    updated_flow = powerups.update_user_kpoints_settings(flow, kpoints_input)
    settings = updated_flow.jobs[
        0
    ].function.__self__.input_set_generator.user_kpoints_settings

    if isinstance(kpoints_input, dict):
        assert all(settings[k] == v for k, v in kpoints_input.items())
    else:
        assert isinstance(settings, Kpoints)


@pytest.mark.parametrize(
    "metadata",
    [
        {"mp-id": "mp-xxx"},
        {"mp-id": "mp-149", "composition": "Si2", "tags": ["semiconductor"]},
    ],
)
def test_add_metadata_to_flow(metadata: dict[str, Any]) -> None:
    """Test adding metadata to flow."""
    flow = DoubleRelaxMaker().make(1)
    updated_flow = powerups.add_metadata_to_flow(flow, metadata)

    for job in updated_flow.jobs:
        assert (
            job.function.__self__.task_document_kwargs["additional_fields"] == metadata
        )


@pytest.mark.parametrize(
    "handlers",
    [(), ("Handler1", "Handler2"), ("CustomHandler",)],
)
def test_update_cp2k_custodian_handlers(handlers: tuple[str, ...]) -> None:
    """Test custodian handlers updates."""
    flow = DoubleRelaxMaker().make(1)
    updated_flow = powerups.update_cp2k_custodian_handlers(flow, handlers)

    for job in updated_flow.jobs:
        assert job.function.__self__.run_cp2k_kwargs["handlers"] == handlers


def test_name_and_class_filters() -> None:
    """Test name and class filters for powerups."""

    settings = {"max_scf": 200}
    flow = DoubleRelaxMaker().make(1)

    # Test name filter
    name_filtered = powerups.update_user_input_settings(
        flow, settings, name_filter="relax 1"
    )
    job1, job2 = name_filtered.jobs

    settings1 = job1.function.__self__.input_set_generator.user_input_settings
    assert settings1["max_scf"] == 200
    settings2 = job2.function.__self__.input_set_generator.user_input_settings
    assert "max_scf" not in settings2

    class CustomMaker(RelaxMaker):
        pass

    # Test class filter
    class_filtered = powerups.update_user_input_settings(
        flow, settings, class_filter=CustomMaker
    )
    for job in class_filtered.jobs:
        settings = job.function.__self__.input_set_generator.user_input_settings
        assert "max_scf" not in settings
