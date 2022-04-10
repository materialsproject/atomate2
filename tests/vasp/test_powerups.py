import pytest


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
    from atomate2.vasp import powerups
    from atomate2.vasp.flows.core import DoubleRelaxMaker
    from atomate2.vasp.jobs.core import RelaxMaker

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


def test_use_auto_ispin():
    from atomate2.vasp.flows.core import DoubleRelaxMaker
    from atomate2.vasp.jobs.core import RelaxMaker
    from atomate2.vasp.powerups import use_auto_ispin

    # test job maker
    rm = RelaxMaker()
    rm = use_auto_ispin(rm)
    assert rm.input_set_generator.auto_ispin is True

    # test job
    job = RelaxMaker().make(1)
    job = use_auto_ispin(job)
    assert job.function.__self__.input_set_generator.auto_ispin is True

    # test flow maker
    drm = DoubleRelaxMaker()
    drm = use_auto_ispin(drm)
    assert drm.relax_maker1.input_set_generator.auto_ispin is True
    assert drm.relax_maker2.input_set_generator.auto_ispin is True

    # test flow
    drm = DoubleRelaxMaker()
    flow = drm.make(1)
    flow = use_auto_ispin(flow)
    assert flow.jobs[0].function.__self__.input_set_generator.auto_ispin is True
    assert flow.jobs[1].function.__self__.input_set_generator.auto_ispin is True

    # test name filter
    drm = DoubleRelaxMaker()
    flow = drm.make(1)
    flow = use_auto_ispin(flow, name_filter="relax 1")
    assert flow.jobs[0].function.__self__.input_set_generator.auto_ispin is True
    assert flow.jobs[1].function.__self__.input_set_generator.auto_ispin is False
