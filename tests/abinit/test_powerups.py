import pytest


@pytest.mark.parametrize(
    "powerup,attribute,settings",
    [
        ("update_user_abinit_settings", "user_abinit_settings", {"ecut": 10}),
        (
            "update_user_kpoints_settings",
            "user_kpoints_settings",
            {"reciprocal_density": 100},
        ),
        ("update_factory_kwargs", "factory_kwargs", {"kppa": 100}),
    ],
)
def test_update_user_settings(powerup, attribute, settings):
    from atomate2.abinit import powerups
    from atomate2.abinit.flows.core import RelaxFlowMaker
    from atomate2.abinit.jobs.core import RelaxMaker

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
    drm = RelaxFlowMaker()
    drm = powerup_func(drm, settings)
    for relax_maker_i in drm.relaxation_makers:
        assert getattr(relax_maker_i.input_set_generator, attribute) == settings

    # test flow
    drm = RelaxFlowMaker()
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
    drm = RelaxFlowMaker()
    flow = drm.make(1)

    flow = powerup_func(
        flow, settings, name_filter="Relaxation calculation (ions only)"
    )
    assert (
        getattr(flow.jobs[0].function.__self__.input_set_generator, attribute)
        == settings
    )
    assert (
        getattr(flow.jobs[1].function.__self__.input_set_generator, attribute)
        != settings
    )


def test_update_generator_attributes():
    from atomate2.abinit.flows.core import RelaxFlowMaker
    from atomate2.abinit.jobs.core import RelaxMaker
    from atomate2.abinit.powerups import update_generator_attributes

    settings = {"pseudos": "ONCVPSP-PBE-SR-PDv0.3:standard"}

    # test job maker
    rm = RelaxMaker()
    rm = update_generator_attributes(rm, settings)
    for attribute, value in settings.items():
        assert getattr(rm.input_set_generator, attribute) == value

    # test job
    job = RelaxMaker().make(1)
    job = update_generator_attributes(job, settings)
    for attribute, value in settings.items():
        assert getattr(job.function.__self__.input_set_generator, attribute) == value

    # test flow maker
    drm = RelaxFlowMaker()
    drm = update_generator_attributes(drm, settings)
    for attribute, value in settings.items():
        for relax_maker_i in drm.relaxation_makers:
            assert getattr(relax_maker_i.input_set_generator, attribute) == value

    # test flow
    drm = RelaxFlowMaker()
    flow = drm.make(1)
    flow = update_generator_attributes(flow, settings)
    for attribute, value in settings.items():
        assert (
            getattr(flow.jobs[0].function.__self__.input_set_generator, attribute)
            == value
        )
        assert (
            getattr(flow.jobs[1].function.__self__.input_set_generator, attribute)
            == value
        )

    # test name filter
    drm = RelaxFlowMaker()
    flow = drm.make(1)
    flow = update_generator_attributes(
        flow, settings, name_filter="Relaxation calculation (ions only)"
    )
    for attribute, value in settings.items():
        assert (
            getattr(flow.jobs[0].function.__self__.input_set_generator, attribute)
            == value
        )
        assert (
            getattr(flow.jobs[1].function.__self__.input_set_generator, attribute)
            != value
        )
