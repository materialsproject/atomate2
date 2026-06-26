"""Test the elastic workflow for TorchSim"""
# ruff: noqa: E402

from __future__ import annotations

import pytest

from atomate2.torchsim.schema import TorchSimModelType

ts = pytest.importorskip("torch_sim")

from jobflow import run_locally
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from atomate2.common.schemas.elastic import ElasticDocument
from atomate2.torchsim.core import TorchSimOptimizeMaker
from atomate2.torchsim.flows.elastic import ElasticMaker

from ..conftest import _SKIP_MACE  # noqa: TID252


@pytest.mark.skipif(_SKIP_MACE, reason="mace_torch is not installed")
@pytest.mark.parametrize("socket", [True, False])
def test_elastic_wf_with_mace(clean_dir, si_structure, test_dir, socket: bool):
    si_prim = SpacegroupAnalyzer(si_structure).get_primitive_standard_structure()
    model_path = f"{test_dir}/forcefields/mace/MACE.model"

    bulk_relax_maker = TorchSimOptimizeMaker(
        optimizer=ts.Optimizer.fire,
        model_type=TorchSimModelType.MACE,
        model_path=model_path,
        init_kwargs={"cell_filter": ts.CellFilter.frechet, "compute_stress": True},
        convergence_fn_kwargs={"force_tol": 0.00001},
    )
    elastic_relax_maker = TorchSimOptimizeMaker(
        optimizer=ts.Optimizer.fire,
        model_type=TorchSimModelType.MACE,
        model_path=model_path,
        convergence_fn_kwargs={"force_tol": 0.00001},
    )

    maker = ElasticMaker(
        bulk_relax_maker=bulk_relax_maker,
        elastic_relax_maker=elastic_relax_maker,
        socket=socket,
    )

    flow = maker.make(si_prim)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    elastic_output = responses[flow[-1].uuid][1].output
    assert isinstance(elastic_output, ElasticDocument)
    assert elastic_output.derived_properties.k_voigt == pytest.approx(
        9.7005429, abs=0.01
    )
    assert elastic_output.derived_properties.g_voigt == pytest.approx(
        0.002005039, abs=0.01
    )
    assert elastic_output.chemsys == "Si"
