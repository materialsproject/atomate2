"""Test ASE NEB jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pytest
from ase.calculators.emt import EMT
from emmet.core.neb import BarrierAnalysis, NebResult
from jobflow import run_locally
from pymatgen.core import Structure

from atomate2.ase.jobs import EmtRelaxMaker
from atomate2.ase.neb import AseNebFromEndpointsMaker, EmtNebFromImagesMaker
from atomate2.common.jobs.neb import _get_images_from_endpoints

if TYPE_CHECKING:
    from atomate2.ase.jobs import AseRelaxMaker


def initial_endpoints():
    structure = Structure(
        4.1 * np.eye(3),
        ["Al", "Al", "Al", "Al"],
        [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
    ) * (2, 2, 2)

    _, n_idx, _, d = structure.get_neighbor_list(
        structure.lattice.a / 2, sites=[structure[0]]
    )
    min_idx = np.argmin(d)
    nn_idx = n_idx[min_idx]

    endpoints = [structure.copy() for _ in range(2)]
    endpoints[0].remove_sites([0])
    endpoints[1].remove_sites([nn_idx])
    return endpoints


@dataclass
class EmtNebFromEndpointsMaker(AseNebFromEndpointsMaker):
    name: str = "ASE EMT NEB from images maker"

    endpoint_relax_maker: AseRelaxMaker = field(
        default_factory=EmtRelaxMaker,
    )

    @property
    def calculator(self):
        return EMT(**self.calculator_kwargs)


def test_neb_from_images():
    images = _get_images_from_endpoints(initial_endpoints(), 1, autosort_tol=0.5)
    assert len(images) == 3
    job = EmtNebFromImagesMaker().make(images)
    response = run_locally(job)
    output = response[job.uuid][1].output
    assert isinstance(output, NebResult)
    assert isinstance(output.barrier_analysis, BarrierAnalysis)

    assert all(
        getattr(output, f"{direction}_barrier") == pytest.approx(0.3823904696311531)
        for direction in ("forward", "reverse")
    )

    assert all(
        output.energies[i] == pytest.approx(ref_energy)
        for i, ref_energy in enumerate(
            [1.1882799797196175, 1.5706704493507706, 1.1882799797196228]
        )
    )

    assert len(output.images) == 3
    assert all(
        images[i] == init_image for i, init_image in enumerate(output.initial_images)
    )
    assert output.state.value == "successful"


def test_neb_from_endpoints(memory_jobstore):
    job = EmtNebFromEndpointsMaker().make(initial_endpoints(), 1, autosort_tol=0.5)
    response = run_locally(job, store=memory_jobstore)

    # `job` is a replacement containing the workflow
    replacement_job_names = {job.name for job in response[job.uuid][1].replace.jobs}
    assert all(
        name in replacement_job_names
        for name in (
            "EMT relaxation endpoint 1",
            "EMT relaxation endpoint 2",
            "AseNebFromImagesMaker.make",
        )
    )

    output = NebResult(
        **memory_jobstore.query_one({"name": "AseNebFromImagesMaker.make"})["output"]
    )

    assert isinstance(output.barrier_analysis, BarrierAnalysis)

    assert all(
        getattr(output, f"{direction}_barrier")
        == pytest.approx(0.40925355998834156, rel=1e-6)
        for direction in ("forward", "reverse")
    )

    assert all(
        output.energies[i] == pytest.approx(ref_energy)
        for i, ref_energy in enumerate(
            [0.8217810288156584, 1.231034588804, 0.8217810288156824]
        )
    )

    assert len(output.images) == 3
    assert output.state.value == "successful"
