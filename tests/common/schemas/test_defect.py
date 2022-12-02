def test_sort_pos_dist():
    """
    Test the sorting algorithm with a list of 2D positions.
    The algorithm should sort the list into a straight line depending on the direction of s1 and s2
    """
    import numpy as np

    from atomate2.common.schemas.defects import sort_pos_dist

    def abs_d(s1, s2):
        return np.linalg.norm(np.array(s1) - np.array(s2))

    points_on_line_2d = [(1, 1), (-2, -2), (0, 0), (2, 2), (-1, -1)]
    r, d = sort_pos_dist(points_on_line_2d, s1=(0, 0), s2=(1.5, 1.5), dist=abs_d)
    assert r == [(-2, -2), (-1, -1), (0, 0), (1, 1), (2, 2)]

    r, d = sort_pos_dist(points_on_line_2d, s1=(0, 0), s2=(-2.5, -2.5), dist=abs_d)
    assert r == [(2, 2), (1, 1), (0, 0), (-1, -1), (-2, -2)]


def test_FormationEnergyDiagramDocument(test_dir):
    import pytest
    from monty.serialization import loadfn

    from atomate2.common.schemas.defects import FormationEnergyDiagramDocument

    test_json = test_dir / "schemas" / "formation_en.json"
    fe_doc = FormationEnergyDiagramDocument(**loadfn(test_json))
    assert fe_doc.vbm == pytest.approx(4.5715)
    fe_obj = fe_doc.as_FormationEnergyDiagram(pd_entries=fe_doc.pd_entries)
    fe_obj1 = fe_doc.as_FormationEnergyDiagram()
    assert set(fe_obj1.pd_entries) == set(fe_obj.pd_entries)
