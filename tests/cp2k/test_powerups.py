import pytest


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
def test_update_user_settings(powerup, attribute, settings):
    from atomate2.cp2k import powerups
    from atomate2.cp2k.flows.core import DoubleRelaxMaker
    from atomate2.cp2k.jobs.core import RelaxMaker

    powerup_func = getattr(powerups, powerup)

    # test job maker
    rm = RelaxMaker()
    rm = powerup_func(rm, settings)
    assert getattr(rm.input_set_generator, attribute) == settings

    # test job
    job = RelaxMaker().make()
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
        ("add_metadata_to_flow", {"mp-id": "mp-170", "composition": "NaCl"}),
    ],
)
def test_add_metadata_to_flow(powerup, settings):
    from atomate2.cp2k import powerups
    from atomate2.cp2k.flows.core import DoubleRelaxMaker

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
    [("update_cp2k_custodian_handlers", ())],
)
def test_update_cp2k_custodian_handlers(powerup, settings):
    from atomate2.cp2k import powerups
    from atomate2.cp2k.flows.core import DoubleRelaxMaker

    powerup_func = getattr(powerups, powerup)

    # test flow
    drm = DoubleRelaxMaker()
    flow = drm.make(1)
    flow = powerup_func(flow, settings)
    assert flow.jobs[0].function.__self__.run_cp2k_kwargs["handlers"] == settings
