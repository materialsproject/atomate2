from atomate2.vampire.vampire_caller import VampireCaller
from atomate2.vampire.schemas.vampire_output import VampireOutput
from pymatgen.analysis.magnetism.heisenberg import HeisenbergModel
from jobflow import job

@job(name="run vampire")
def run_vampire(
    heisenberg_model: HeisenbergModel,
    mc_settings: dict | None = None,
) -> VampireOutput:
    """Run Vampire Monte-Carlo to estimate the critical temperature.

    This wraps the (vendored) ``VampireCaller``, which shells out to the external
    ``vampire-serial`` binary. A clear error is raised if the binary is not found
    on PATH.

    Parameters
    ----------
    heisenberg_model : HeisenbergModel
        The fitted Heisenberg model from :func:`heisenberg_mapping`.
    mc_settings : dict or None
        Keyword arguments for VampireCaller, e.g. ``mc_box_size``,
        ``equil_timesteps``, ``mc_timesteps``, ``avg``.

    Returns
    -------
    VampireOutput
        The Vampire Monte-Carlo result, exposing ``critical_temp``.
    """
    mc_settings = mc_settings or {}
    vampire_caller = VampireCaller(hm=heisenberg_model, **mc_settings)
    return vampire_caller.output