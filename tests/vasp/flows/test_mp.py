import pytest

from atomate2.vasp.flows.mp import MP2023RelaxMaker
from atomate2.vasp.jobs.mp import MPPreRelaxMaker, MPRelaxMaker


def test_MP2023RelaxMaker_default_values():
    job = MP2023RelaxMaker()

    assert isinstance(job.pre_relax_maker, MPPreRelaxMaker)
    assert job.pre_relax_maker.name == "MP PreRelax"

    assert isinstance(job.relax_maker, MPRelaxMaker)
    assert job.relax_maker.name == "MP Relax"

    assert job.name == "MP 2023 Relax"


@pytest.mark.parametrize(
    "name, pre_relax_maker, relax_maker",
    [
        ("Test", MPPreRelaxMaker(), MPRelaxMaker()),
        ("Relax", MPPreRelaxMaker(), MPRelaxMaker()),
    ],
)
def test_MP2023RelaxMaker_custom_values(name, pre_relax_maker, relax_maker):
    maker = MP2023RelaxMaker(
        name=name, pre_relax_maker=pre_relax_maker, relax_maker=relax_maker
    )
    assert maker.name == name
    assert maker.pre_relax_maker == pre_relax_maker
    assert maker.relax_maker == relax_maker


# @pytest.mark.parametrize("prev_vasp_dir", [None, "/dummy/dir", Path("/dummy/dir")])
# def test_make(si_structure, dummy_flow, prev_vasp_dir):
#     # Mock the make method to return a dummy flow
#     maker = MP2023RelaxMaker()
#     flow = maker.make(si_structure, prev_vasp_dir)

#     assert len(flow.jobs) == 2
#     assert flow.jobs[0].output == dummy_flow.output
#     assert flow.output == dummy_flow.output
#     assert flow.name == maker.name

#     MPPreRelaxMaker.make.assert_called_once_with(
#         si_structure, prev_vasp_dir=prev_vasp_dir
#     )
#     MPRelaxMaker.make.assert_called_once_with(
#         dummy_flow.output.structure, prev_vasp_dir=dummy_flow.output.dir_name
#     )
