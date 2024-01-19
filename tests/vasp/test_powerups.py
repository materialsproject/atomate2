import pytest

from atomate2.vasp import powerups
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.core import RelaxMaker


@pytest.mark.parametrize(
    "powerup,attribute,settings",
    [
        ("update_user_incar_settings", "user_incar_settings", {"NCORE": 10}),
        (
            "update_user_kpoints_settings",
            "user_kpoints_settings",
            {"reciprocal_density": 100},
        ),
        ("update_user_potcar_settings", "user_potcar_settings", {"Gd": "Gd_3"}),
        ("update_user_potcar_functional", "user_potcar_functional", "LDA"),
        ("use_auto_ispin", "auto_ispin", True),
    ],
)
def test_update_user_settings(powerup, attribute, settings):
    powerup_func = getattr(powerups, powerup)

    # test job maker
    rm = RelaxMaker()
    rm = powerup_func(rm, settings)
    assert getattr(rm.input_set_generator, attribute) == settings

    # test job
    job = RelaxMaker().make(1)
    job = powerup_func(job, settings)
    assert getattr(job.function.__self__.input_set_generator, attribute) == settings

    # test flow maker
    drm = DoubleRelaxMaker()
    drm = powerup_func(drm, settings)
    assert getattr(drm.relax_maker1.input_set_generator, attribute) == settings
    assert getattr(drm.relax_maker2.input_set_generator, attribute) == settings

    # test flow
    drm = DoubleRelaxMaker()
    flow = drm.make(1)
    flow = powerup_func(flow, settings)
    assert (
        getattr(flow.jobs[0].function.__self__.input_set_generator, attribute)
        == settings
    )
    assert (
        getattr(flow.jobs[1].function.__self__.input_set_generator, attribute)
        == settings
    )

    # test name filter
    drm = DoubleRelaxMaker()
    flow = drm.make(1)
    flow = powerup_func(flow, settings, name_filter="relax 1")
    assert (
        getattr(flow.jobs[0].function.__self__.input_set_generator, attribute)
        == settings
    )
    assert (
        getattr(flow.jobs[1].function.__self__.input_set_generator, attribute)
        != settings
    )


@pytest.mark.parametrize(
    "powerup,settings",
    [
        ("add_metadata_to_flow", {"mp-id": "mp-xxx"}),
        ("add_metadata_to_flow", {"mp-id": "mp-161", "composition": "NaCl"}),
    ],
)
def test_add_metadata_to_flow(powerup, settings):
    powerup_func = getattr(powerups, powerup)

    # test flow
    drm = DoubleRelaxMaker()
    flow = drm.make(1)
    flow = powerup_func(flow, settings)
    assert (
        flow.jobs[0].function.__self__.task_document_kwargs["additional_fields"]
        == settings
    )


@pytest.mark.parametrize(
    "powerup, settings",
    [("update_vasp_custodian_handlers", ())],
)
def test_update_vasp_custodian_handlers(powerup, settings):
    powerup_func = getattr(powerups, powerup)

    # test flow
    drm = DoubleRelaxMaker()
    flow = drm.make(1)
    flow = powerup_func(flow, settings)
    assert flow.jobs[0].function.__self__.run_vasp_kwargs["handlers"] == settings
