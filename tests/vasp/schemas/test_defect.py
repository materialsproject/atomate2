import numpy as np

from atomate2.vasp.schemas.defect import CCDDocument, sort_pos_dist
from atomate2.vasp.schemas.task import TaskDocument


def test_sort_pos_dist():
    """
    Test the sorting algorithm with a list of 2D positions.
    The alorithm should sort the list into a straight line depending on the direction of s1 and s2
    """

    def abs_d(s1, s2):
        return np.linalg.norm(np.array(s1) - np.array(s2))

    points_on_line_2d = [(1, 1), (-2, -2), (0, 0), (2, 2), (-1, -1)]
    r, d = sort_pos_dist(points_on_line_2d, s1=(0, 0), s2=(1.5, 1.5), dist=abs_d)
    assert r == [(-2, -2), (-1, -1), (0, 0), (1, 1), (2, 2)]

    r, d = sort_pos_dist(points_on_line_2d, s1=(0, 0), s2=(-2.5, -2.5), dist=abs_d)
    assert r == [(2, 2), (1, 1), (0, 0), (-1, -1), (-2, -2)]


def test_CCDDocument(vasp_test_dir):
    """
    Test the CCDDocument schema
    """

    def is_strict_minimum(min_index, arr):
        min_val = arr[min_index]
        for i, val in enumerate(arr):
            if i != min_index:
                if val < min_val:
                    return False
        return True

    distored_0 = []
    distored_1 = []
    for i in range(5):
        static_dir0 = vasp_test_dir / "Si_CCD" / f"static_q=0_{i}" / "outputs"
        static_dir1 = vasp_test_dir / "Si_CCD" / f"static_q=1_{i}" / "outputs"
        distored_0.append(TaskDocument.from_directory(static_dir0))
        distored_1.append(TaskDocument.from_directory(static_dir1))
    relaxed_0 = TaskDocument.from_directory(
        vasp_test_dir / "Si_CCD" / "relax_q=0" / "outputs"
    )
    relaxed_1 = TaskDocument.from_directory(
        vasp_test_dir / "Si_CCD" / "relax_q=1" / "outputs"
    )
    s0 = relaxed_0.output.structure
    s1 = relaxed_1.output.structure

    ccd_doc = CCDDocument.from_distorted_calcs(distored_0, distored_1, s0, s1)
    assert is_strict_minimum(2, ccd_doc.energies1)
    assert is_strict_minimum(2, ccd_doc.energies2)

    # check that you can recreate the task document from the ccd document
    tasks = ccd_doc.get_taskdocs()
    len(tasks) == 2
