import pytest

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
def test_update_user_settings(powerup, attribute, settings):
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
