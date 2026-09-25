import pytest
from jobflow import Flow
from pymatgen.core.structure import Structure

from atomate2.vasp.flows.cte import CTEMaker
from atomate2.vasp.flows.elastic import ElasticMaker
from atomate2.vasp.flows.pheasy import PhononMaker


def test_cte_maker_vasp_flow(si_structure: Structure):
    """The relaxed structure goes to both flows, and their outputs to compute_cte."""
    maker = CTEMaker(temperatures=[100, 300], mesh=(10, 10, 10))
    flow = maker.make(si_structure)
    relax, phonon_flow, elastic_flow, cte = flow.jobs
    assert isinstance(phonon_flow, Flow)
    assert isinstance(elastic_flow, Flow)

    # both flows start from the relaxed structure
    relaxed = relax.output.structure
    for sub_flow in (phonon_flow, elastic_flow):
        structure = sub_flow.jobs[0].function_args[0]
        assert structure.uuid == relaxed.uuid
        assert structure.attributes == relaxed.attributes

    kwargs = cte.function_kwargs
    assert kwargs["phonon_output"].uuid == phonon_flow.output.uuid
    assert kwargs["elastic_tensor"].uuid == elastic_flow.output.uuid
    assert kwargs["elastic_structure"].uuid == elastic_flow.output.uuid
    assert kwargs["anhar_fit_methods"] == ("one-shot",)
    assert kwargs["temperatures"] == [100, 300]
    assert kwargs["mesh"] == (10, 10, 10)
    assert kwargs["symprec"] == maker.phonon_maker.symprec
    assert kwargs["min_frequency"] == maker.min_frequency
    assert maker.phonon_maker.min_length == 12.0

    assert kwargs["elastic_tensor"].attributes == (
        ("a", "elastic_tensor"),
        ("a", "raw"),
    )

    # the elastic fit gets the stress of the relaxation, as in the elastic flow
    fit = next(job for job in elastic_flow.jobs if job.name == "fit_elastic_tensor")
    stress = fit.function_kwargs["equilibrium_stress"]
    assert stress.uuid == relax.output.uuid
    assert stress.attributes == (("a", "output"), ("a", "stress"))


@pytest.mark.parametrize(
    ("phonon_kwargs", "match"),
    [
        ({"bulk_relax_maker": None, "cal_anhar_fcs": False}, "cal_anhar_fcs=True"),
        (
            {
                "bulk_relax_maker": None,
                "cal_anhar_fcs": True,
                "use_symmetrized_structure": "primitive",
            },
            "use_symmetrized_structure=None",
        ),
        ({"cal_anhar_fcs": True}, "The phonon maker needs bulk_relax_maker=None"),
    ],
)
def test_cte_maker_checks_phonon_maker(phonon_kwargs, match):
    phonon_maker = PhononMaker(**phonon_kwargs)
    with pytest.raises(ValueError, match=match):
        CTEMaker(phonon_maker=phonon_maker)


def test_cte_maker_checks_elastic_maker():
    with pytest.raises(ValueError, match="The elastic maker needs bulk_relax_maker"):
        CTEMaker(elastic_maker=ElasticMaker())
