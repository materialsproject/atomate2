from pathlib import Path

import numpy as np
import pytest
from jobflow import JobStore, run_locally
from maggma.stores import MemoryStore
from pymatgen.core import Molecule

from atomate2.qchem.flows.core import FrequencyOptFlatteningMaker, FrequencyOptMaker
from atomate2.qchem.jobs.core import FreqMaker, OptMaker
from atomate2.qchem.sets.core import FreqSetGenerator, OptSetGenerator

fake_run_qchem_kwargs = {}


def test_frequency_opt_maker(mock_qchem, clean_dir, qchem_test_dir, h2o_molecule):
    ref_paths = {
        "Geometry Optimization": Path(qchem_test_dir)
        / "ffopt"
        / "geometry_optimization",
        "Frequency Analysis": Path(qchem_test_dir) / "ffopt" / "frequency_analysis_1",
    }
    mock_qchem(ref_paths, fake_run_qchem_kwargs)

    flow = FrequencyOptMaker().make(h2o_molecule)
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    output = {job.name: responses[job.uuid][1].output for job in flow}

    ref_total_energy = -76.346601
    assert output["Geometry Optimization"].output.final_energy == pytest.approx(
        ref_total_energy, rel=1e-6
    )

    assert output["Frequency Analysis"].output.final_energy == pytest.approx(
        ref_total_energy, rel=1e-6
    )
    ref_freq = [1587.39, 3864.9, 3969.87]
    assert all(
        freq == pytest.approx(ref_freq[i], abs=1e-2)
        for i, freq in enumerate(output["Frequency Analysis"].output.frequencies)
    )
    assert (
        output["Geometry Optimization"].output.optimized_molecule
        == output["Frequency Analysis"].output.initial_molecule
    )
    assert output["Frequency Analysis"].output.optimized_molecule is None


def _out_of_plane_height(molecule: Molecule) -> float:
    """Distance of the first atom from the plane of the next three atoms."""
    center, *others = (site.coords for site in molecule)
    normal = np.cross(others[1] - others[0], others[2] - others[0])
    return abs(np.dot(center - others[0], normal / np.linalg.norm(normal)))


def test_frequency_opt_flattening_maker(mock_qchem, clean_dir, qchem_test_dir):
    # planar NH3 is the transition state of the NH3 inversion. By symmetry the first
    # optimization stays planar and the frequency analysis finds one imaginary mode,
    # so the workflow must perturb along that mode and reoptimize to pyramidal NH3.
    planar_nh3 = Molecule(
        species=["N", "H", "H", "H"],
        coords=[
            [0.0, 0.0, 0.0],
            [1.012, 0.0, 0.0],
            [-0.506, 0.8764292328, 0.0],
            [-0.506, -0.8764292328, 0.0],
        ],
    )
    ref_dir = Path(qchem_test_dir) / "ffopt_planar_nh3"
    ref_paths = {
        name: ref_dir / name.lower().replace(" ", "_")
        for name in (
            "Geometry Optimization 1",
            "Frequency Analysis 1",
            "Geometry Optimization 2",
            "Frequency Analysis 2",
        )
    }
    mock_qchem(ref_paths, fake_run_qchem_kwargs)

    # the reference calculations use def2-SVPD to keep the test data small, and were
    # generated with the original atomate2 algorithm: displace by the raw mode vector
    maker = FrequencyOptFlatteningMaker(
        opt_maker=OptMaker(input_set_generator=OptSetGenerator(basis_set="def2-svpd")),
        freq_maker=FreqMaker(
            input_set_generator=FreqSetGenerator(basis_set="def2-svpd")
        ),
        scale=1.0,
        max_ffopt_runs=5,
        imag_freq_tolerance=0.0,
        energy_diff_cutoff=None,
        linked=False,
    )
    flow = maker.make(planar_nh3)
    store = JobStore(MemoryStore(), additional_stores={"data": MemoryStore()})
    responses = run_locally(flow, store=store, create_folders=True, ensure_success=True)

    # get the QChem jobs of the dynamic flow in the order they ran
    jobs = []
    for resp in responses.values():
        if replace_flow := getattr(resp[1], "replace", None):
            jobs += [
                job
                for job in replace_flow.jobs
                if job.name not in (maker.name, "store_inputs")
            ]
    assert [job.name for job in jobs] == list(ref_paths)
    opt1, freq1, opt2, freq2 = (responses[job.uuid][1].output for job in jobs)

    # the numbered job names are not used as the task type
    assert [doc.task_type for doc in (opt1, freq1, opt2, freq2)] == [
        "Geometry Optimization",
        "Frequency Analysis",
    ] * 2

    # the first cycle stays at the planar transition state
    assert opt1.output.final_energy == pytest.approx(-56.479811309, abs=1e-8)
    assert _out_of_plane_height(opt1.output.optimized_molecule) == pytest.approx(
        0.0, abs=1e-3
    )
    assert freq1.output.frequencies[:3] == pytest.approx(
        [-743.32, 1540.92, 1543.11], abs=1e-2
    )
    assert sum(freq < 0 for freq in freq1.output.frequencies) == 1

    # the second cycle reaches the pyramidal minimum
    assert opt2.output.final_energy == pytest.approx(-56.486273422, abs=1e-8)
    assert opt2.output.final_energy < opt1.output.final_energy
    assert _out_of_plane_height(opt2.output.optimized_molecule) == pytest.approx(
        0.375, abs=1e-3
    )
    assert freq2.output.frequencies[:3] == pytest.approx(
        [976.56, 1615.17, 1617.33], abs=1e-2
    )
    assert all(freq > 0 for freq in freq2.output.frequencies)

    # each frequency analysis runs at the optimized geometry and copies files from
    # the optimization directory
    for job, opt, freq in ((jobs[1], opt1, freq1), (jobs[3], opt2, freq2)):
        assert opt.output.optimized_molecule == freq.output.initial_molecule
        assert freq.output.optimized_molecule is None
        assert job.function_kwargs["prev_dir"] == opt.dir_name

    # the flow output should be the final frequency analysis task document
    flow_output = flow.output.resolve(store)
    assert flow_output is not None
    assert flow_output.output.final_energy == pytest.approx(
        freq2.output.final_energy, abs=1e-8
    )
    assert flow_output.output.frequencies == pytest.approx(
        freq2.output.frequencies, abs=1e-2
    )


def _planar_nh3() -> Molecule:
    return Molecule(
        species=["N", "H", "H", "H"],
        coords=[
            [0.0, 0.0, 0.0],
            [1.012, 0.0, 0.0],
            [-0.506, 0.8764292328, 0.0],
            [-0.506, -0.8764292328, 0.0],
        ],
    )


def _qchem_jobs(responses, maker):
    """QChem jobs of a dynamic frequency flattening flow, in the order they ran."""
    jobs = []
    for resp in responses.values():
        if replace_flow := getattr(resp[1], "replace", None):
            jobs += [
                job
                for job in replace_flow.jobs
                if job.name not in (maker.name, "store_inputs")
            ]
    return jobs


def _svpd_ffopt_maker(**kwargs) -> FrequencyOptFlatteningMaker:
    return FrequencyOptFlatteningMaker(
        opt_maker=OptMaker(input_set_generator=OptSetGenerator(basis_set="def2-svpd")),
        freq_maker=FreqMaker(
            input_set_generator=FreqSetGenerator(basis_set="def2-svpd")
        ),
        **kwargs,
    )


def test_frequency_opt_flattening_maker_imag_freq_tolerance(
    mock_qchem, clean_dir, qchem_test_dir
):
    # the planar NH3 transition state has a single imaginary mode at -743 cm^-1,
    # which a large enough tolerance accepts after the first iteration
    ref_dir = Path(qchem_test_dir) / "ffopt_planar_nh3"
    ref_paths = {
        name: ref_dir / name.lower().replace(" ", "_")
        for name in ("Geometry Optimization 1", "Frequency Analysis 1")
    }
    mock_qchem(ref_paths, fake_run_qchem_kwargs)

    maker = _svpd_ffopt_maker(imag_freq_tolerance=800.0)
    flow = maker.make(_planar_nh3())
    store = JobStore(MemoryStore(), additional_stores={"data": MemoryStore()})
    responses = run_locally(flow, store=store, create_folders=True, ensure_success=True)

    jobs = _qchem_jobs(responses, maker)
    assert [job.name for job in jobs] == list(ref_paths)

    flow_output = flow.output.resolve(store)
    assert flow_output.output.frequencies[0] == pytest.approx(-743.32, abs=1e-2)


def test_frequency_opt_flattening_maker_adaptive_perturbation(
    mock_qchem, clean_dir, qchem_test_dir
):
    pytest.importorskip("openbabel")
    ref_dir = Path(qchem_test_dir) / "ffopt_planar_nh3"
    ref_paths = {
        name: ref_dir / name.lower().replace(" ", "_")
        for name in (
            "Geometry Optimization 1",
            "Frequency Analysis 1",
            "Geometry Optimization 2",
            "Frequency Analysis 2",
        )
    }
    # the adaptive displacement differs from the one used for the reference input
    mock_qchem(ref_paths, {"Geometry Optimization 2": {"input_exclude": ["molecule"]}})

    maker = _svpd_ffopt_maker(scale=0.3, linked=False, adaptive_perturbation=True)
    flow = maker.make(_planar_nh3())
    store = JobStore(MemoryStore(), additional_stores={"data": MemoryStore()})
    responses = run_locally(flow, store=store, create_folders=True, ensure_success=True)

    jobs = _qchem_jobs(responses, maker)
    assert [job.name for job in jobs] == list(ref_paths)
    opt1 = responses[jobs[0].uuid][1].output

    # the second optimization starts from the first optimized geometry displaced so
    # that the most displaced atom moves by exactly `scale`
    start = jobs[2].function_args[0]
    shifts = np.linalg.norm(
        start.cart_coords - opt1.output.optimized_molecule.cart_coords, axis=1
    )
    assert shifts.max() == pytest.approx(0.3, abs=1e-6)
    assert _out_of_plane_height(start) > 0

    flow_output = flow.output.resolve(store)
    assert all(freq > 0 for freq in flow_output.output.frequencies)


@pytest.mark.parametrize(
    ("maker_kwargs", "lowest_freq", "frequencies", "energies", "expected"),
    [
        # without a tolerance any imaginary frequency triggers another iteration
        (
            {"imag_freq_tolerance": 0.0, "energy_diff_cutoff": None},
            -5.0,
            [-5.0, 100.0],
            [-1.0, -1.0],
            False,
        ),
        # the default tolerance is custodian's 15 cm^-1
        ({}, -14.9, [-14.9, 100.0], [], True),
        # a single small imaginary frequency is tolerated
        ({"imag_freq_tolerance": 15.0}, -14.9, [-14.9, 100.0], [], True),
        # but not a larger one
        ({"imag_freq_tolerance": 15.0}, -15.1, [-15.1, 100.0], [], False),
        # nor two small ones
        ({"imag_freq_tolerance": 15.0}, -10.0, [-10.0, -5.0, 100.0], [], False),
        # the energy check needs two optimizations
        ({"energy_diff_cutoff": 1e-6}, -50.0, [-50.0], [-10.0], False),
        ({"energy_diff_cutoff": 1e-6}, -50.0, [-50.0], [-10.0, -10.0000005], True),
        ({"energy_diff_cutoff": 1e-6}, -50.0, [-50.0], [-10.0, -10.00001], False),
    ],
)
def test_frequency_opt_flattening_convergence_criteria(
    maker_kwargs, lowest_freq, frequencies, energies, expected
):
    maker = FrequencyOptFlatteningMaker(**maker_kwargs)
    converged = maker._is_converged(lowest_freq, frequencies, energies)  # noqa: SLF001
    assert converged is expected


def test_perturb_preserving_connectivity():
    pytest.importorskip("openbabel")
    from atomate2.qchem.flows.core import perturb_preserving_connectivity

    nh3 = _planar_nh3()
    # out-of-plane umbrella mode: the hydrogens move more than the nitrogen
    umbrella = [[0.0, 0.0, -0.3], [0.0, 0.0, 0.1], [0.0, 0.0, 0.1], [0.0, 0.0, 0.1]]
    perturbed = perturb_preserving_connectivity(nh3, umbrella, max_scale=0.2)
    shifts = np.linalg.norm(perturbed.cart_coords - nh3.cart_coords, axis=1)
    assert shifts.max() == pytest.approx(0.2)
    assert shifts == pytest.approx([0.2, 0.2 / 3, 0.2 / 3, 0.2 / 3])
    assert nh3.cart_coords[0] == pytest.approx([0.0, 0.0, 0.0])  # not modified

    # pulling one hydrogen far away would break its bond, so the displacement is
    # reduced until the bonding is unchanged
    stretch = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    perturbed = perturb_preserving_connectivity(
        nh3, stretch, max_scale=2.0, min_scale=0.1, scale_grid=19
    )
    shift = np.linalg.norm(perturbed.cart_coords - nh3.cart_coords, axis=1).max()
    assert 0.1 <= shift < 2.0

    with pytest.raises(ValueError, match="without changing its bonding"):
        perturb_preserving_connectivity(
            nh3, stretch, max_scale=3.0, min_scale=2.0, scale_grid=2
        )

    # a zero mode leaves the molecule unchanged
    zero = [[0.0, 0.0, 0.0]] * 4
    assert perturb_preserving_connectivity(nh3, zero, 0.3).cart_coords == (
        pytest.approx(nh3.cart_coords)
    )


def test_frequency_opt_flattening_maker_linked(mock_qchem, clean_dir, qchem_test_dir):
    ref_dir = Path(qchem_test_dir) / "ffopt_planar_nh3"
    ref_paths = {
        name: ref_dir / name.lower().replace(" ", "_")
        for name in (
            "Geometry Optimization 1",
            "Frequency Analysis 1",
            "Geometry Optimization 2",
            "Frequency Analysis 2",
        )
    }
    # the linked optimization restarts from the unperturbed geometry and reads the
    # Hessian, unlike the (perturbed) reference calculation
    mock_qchem(
        ref_paths,
        {"Geometry Optimization 2": {"input_exclude": ["molecule", "rem"]}},
    )

    maker = _svpd_ffopt_maker(linked=True)
    flow = maker.make(_planar_nh3())
    store = JobStore(MemoryStore(), additional_stores={"data": MemoryStore()})
    responses = run_locally(flow, store=store, create_folders=True, ensure_success=True)

    jobs = _qchem_jobs(responses, maker)
    assert [job.name for job in jobs] == list(ref_paths)
    opt1, freq1 = (responses[job.uuid][1].output for job in jobs[:2])

    # the second optimization starts from the first optimized geometry, copies the
    # Hessian of the first frequency analysis and reads it as its initial Hessian
    opt2_job = jobs[2]
    assert opt2_job.function_args[0] == opt1.output.optimized_molecule
    assert opt2_job.function_kwargs["prev_dir"] == freq1.dir_name
    opt2_maker = opt2_job.function.__self__
    rem = opt2_maker.input_set_generator.overwrite_inputs["rem"]
    assert rem["geom_opt_hessian"] == "read"
    assert "132.0" in opt2_maker.copy_qchem_kwargs["additional_qchem_files"]
    opt2_dir = Path(responses[opt2_job.uuid][1].output.dir_name.split(":")[-1])
    assert any(opt2_dir.glob("132.0*"))

    # the first optimization is unchanged
    assert "rem" not in (maker.opt_maker.input_set_generator.overwrite_inputs or {})

    flow_output = flow.output.resolve(store)
    assert all(freq > 0 for freq in flow_output.output.frequencies)


def test_frequency_opt_flattening_linked_opt_maker():
    opt_maker = OptMaker(
        input_set_generator=OptSetGenerator(
            overwrite_inputs={"rem": {"thresh": "14"}, "smx": {"solvent": "water"}},
            new_geom_opt={"maxiter": "200"},
        ),
        copy_qchem_kwargs={"additional_qchem_files": ("53.0",)},
    )
    maker = FrequencyOptFlatteningMaker(opt_maker=opt_maker, linked=True)
    linked = maker._linked_opt_maker()  # noqa: SLF001

    generator = linked.input_set_generator
    assert generator.overwrite_inputs == {
        "rem": {"thresh": "14", "geom_opt_hessian": "read"},
        "smx": {"solvent": "water"},
    }
    assert generator.new_geom_opt == {"maxiter": "200", "initial_hessian": "read"}
    assert linked.copy_qchem_kwargs["additional_qchem_files"] == ("53.0", "132.0")

    # the original opt maker is not modified
    assert opt_maker.input_set_generator.overwrite_inputs["rem"] == {"thresh": "14"}
    assert opt_maker.input_set_generator.new_geom_opt == {"maxiter": "200"}
    assert opt_maker.copy_qchem_kwargs == {"additional_qchem_files": ("53.0",)}


def test_frequency_opt_flattening_maker_incompatible_options():
    with pytest.raises(ValueError, match="cannot both be enabled"):
        FrequencyOptFlatteningMaker(linked=True, adaptive_perturbation=True)
    with pytest.raises(ValueError, match="enable linked"):
        FrequencyOptFlatteningMaker(linked=False, freq_before_opt=True)


def test_frequency_opt_flattening_maker_defaults():
    # the defaults follow custodian's linked frequency flattener
    maker = FrequencyOptFlatteningMaker()
    assert maker.linked
    assert not maker.adaptive_perturbation
    assert maker.imag_freq_tolerance == 15.0
    assert maker.energy_diff_cutoff == 1e-6
    assert maker.max_ffopt_runs == 10
    assert maker.scale == 0.3


@pytest.mark.parametrize("linked", [True, False])
def test_frequency_opt_flattening_maker_custodian_like(linked):
    opt_maker = OptMaker(input_set_generator=OptSetGenerator(basis_set="def2-svpd"))
    maker = FrequencyOptFlatteningMaker.custodian_like(
        linked=linked, opt_maker=opt_maker, max_ffopt_runs=3
    )
    assert maker.linked is linked
    assert maker.adaptive_perturbation is not linked
    assert maker.backtrack
    assert (maker.scale, maker.min_scale, maker.scale_grid) == (0.3, 0.1, 10)
    assert (maker.imag_freq_tolerance, maker.energy_diff_cutoff) == (15.0, 1e-6)
    # keyword arguments override the custodian settings
    assert maker.opt_maker is opt_maker
    assert maker.max_ffopt_runs == 3


def test_frequency_opt_flattening_maker_freq_before_opt(
    mock_qchem, clean_dir, qchem_test_dir
):
    ref_dir = Path(qchem_test_dir) / "ffopt_planar_nh3"
    names = (
        "Geometry Optimization 1",
        "Frequency Analysis 1",
        "Geometry Optimization 2",
        "Frequency Analysis 2",
    )
    ref_paths = {name: ref_dir / name.lower().replace(" ", "_") for name in names}
    # the initial frequency analysis of the (planar) input reuses the data of the
    # first one, and the linked optimizations read the Hessian, unlike the reference
    ref_paths["Frequency Analysis 0"] = ref_dir / "frequency_analysis_1"
    mock_qchem(
        ref_paths,
        {
            "Frequency Analysis 0": {"input_exclude": ["molecule"]},
            "Geometry Optimization 1": {"input_exclude": ["rem"]},
            "Geometry Optimization 2": {"input_exclude": ["molecule", "rem"]},
        },
    )

    maker = _svpd_ffopt_maker(freq_before_opt=True)
    flow = maker.make(_planar_nh3())
    store = JobStore(MemoryStore(), additional_stores={"data": MemoryStore()})
    responses = run_locally(flow, store=store, create_folders=True, ensure_success=True)

    jobs = _qchem_jobs(responses, maker)
    assert [job.name for job in jobs] == ["Frequency Analysis 0", *names]

    # the first optimization reads the Hessian of the initial frequency analysis
    freq0 = responses[jobs[0].uuid][1].output
    opt1_job = jobs[1]
    assert opt1_job.function_kwargs["prev_dir"] == freq0.dir_name
    rem = opt1_job.function.__self__.input_set_generator.overwrite_inputs["rem"]
    assert rem["geom_opt_hessian"] == "read"

    flow_output = flow.output.resolve(store)
    assert all(freq > 0 for freq in flow_output.output.frequencies)


def _iteration(num_neg_freqs, energy, parent=None, children=(), mode=None):
    return {
        "molecule": _planar_nh3(),
        "mode": mode or [[0.0, 0.0, 1.0]] + [[0.0, 0.0, -0.3]] * 3,
        "num_neg_freqs": num_neg_freqs,
        "energy": energy,
        "parent": parent,
        "children": list(children),
    }


def test_plan_backtracking_perturbation():
    from atomate2.qchem.flows.core import plan_backtracking_perturbation

    # the first iteration is displaced along its own mode
    history = [_iteration(1, -1.0)]
    assert plan_backtracking_perturbation(history) == (0, False)
    assert history[0]["children"] == [1]

    # fewer imaginary frequencies than the parent: continue from the new iteration
    history = [_iteration(2, -1.0, children=[1]), _iteration(1, -1.1, parent=0)]
    assert plan_backtracking_perturbation(history) == (1, False)
    assert history[1]["children"] == [2]

    # no improvement after the positive displacement: try the negative one
    history = [_iteration(1, -1.0, children=[1]), _iteration(1, -1.1, parent=0)]
    assert plan_backtracking_perturbation(history) == (0, True)
    assert history[0]["children"] == [1, 2]

    # both directions tried: continue from (a copy of) the lower-energy result
    history = [
        _iteration(1, -1.0, children=[1, 2]),
        _iteration(1, -1.2, parent=0),
        _iteration(1, -1.1, parent=0),
    ]
    assert plan_backtracking_perturbation(history) == (3, False)
    assert history[3]["energy"] == -1.2
    assert history[3]["children"] == [4]

    # neither direction lowers the energy or changes the mode: give up
    history = [
        _iteration(1, -1.0, children=[1, 2]),
        _iteration(1, -0.9, parent=0),
        _iteration(1, -0.8, parent=0),
    ]
    with pytest.raises(ValueError, match="Neither displacement direction"):
        plan_backtracking_perturbation(history)

    # nor can two imaginary frequencies be resolved this way
    history = [
        _iteration(2, -1.0, children=[1, 2]),
        _iteration(2, -1.2, parent=0),
        _iteration(2, -1.1, parent=0),
    ]
    with pytest.raises(ValueError, match="more than one imaginary"):
        plan_backtracking_perturbation(history)


def test_perturb_preserving_connectivity_reversed():
    pytest.importorskip("openbabel")
    from atomate2.qchem.flows.core import perturb_preserving_connectivity

    nh3 = _planar_nh3()
    umbrella = [[0.0, 0.0, -0.3], [0.0, 0.0, 0.1], [0.0, 0.0, 0.1], [0.0, 0.0, 0.1]]
    forward = perturb_preserving_connectivity(nh3, umbrella, 0.2)
    backward = perturb_preserving_connectivity(
        nh3, umbrella, 0.2, reversed_direction=True
    )
    assert forward.cart_coords[0][2] == pytest.approx(-0.2)
    assert backward.cart_coords[0][2] == pytest.approx(0.2)
