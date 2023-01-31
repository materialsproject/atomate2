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
