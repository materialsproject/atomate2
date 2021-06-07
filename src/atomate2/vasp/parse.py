"""Functions for parsing calculation outputs."""

from __future__ import annotations

import logging
import typing

from atomate2.settings import settings

if typing.TYPE_CHECKING:
    from pathlib import Path
    from typing import Union


logger = logging.getLogger(__name__)


def parse_vasp_outputs(
    calc_dir: Path = None,
    defuse_unsuccessful: Union[bool, str] = settings.VASP_DEFUSE_UNSUCCESSFUL,
    vasp_drone_kwargs: dict = None,
):
    """
    Parse VASP outputs and generate a :obj:`.Response`.

    Parameters
    ----------
    calc_dir
        The calculation directory to parse. Defaults to parsing the current directory.
    defuse_unsuccessful
        This is a three-way toggle on what to do if your job looks OK, but is actually
        unconverged (either electronic or ionic).
        - `True`: Mark job as completed, but defuse children.
        - `False`: Do nothing, continue with workflow as normal.
        - `"fizzle"`: Throw an error (mark this job as fizzled).
    vasp_drone_kwargs
        Additional keyword arguments to pass to the VaspDrone.
    """
    from pathlib import Path

    from jobflow import Response

    from atomate2.vasp.drones import VaspDrone

    if calc_dir is None:
        calc_dir = Path.cwd()

    # parse the VASP directory
    logger.info(f"Parsing directory: {calc_dir}")

    vasp_drone_kwargs = {} if vasp_drone_kwargs is None else vasp_drone_kwargs
    drone = VaspDrone(**vasp_drone_kwargs)
    task_doc = drone.assimilate(calc_dir)

    stop_children = False
    if task_doc.state != "successful":
        if isinstance(defuse_unsuccessful, bool):
            stop_children = defuse_unsuccessful
        elif defuse_unsuccessful == "fizzle":
            raise RuntimeError(
                "Job was not successful (perhaps your job did not converge within the "
                "limit of electronic/ionic iterations)!"
            )
        else:
            raise RuntimeError(
                f"Unknown option for defuse_unsuccessful: {defuse_unsuccessful}"
            )

    return Response(
        stored_data={"task_id": task_doc.get("task_id", None)},
        stop_children=stop_children,
        output=task_doc,
    )
