from collections import defaultdict

from emmet.core.tasks import TaskDoc

from atomate2.common.schemas.defects import CCDDocument


def test_ccd_document(vasp_test_dir):
    """
    Test the CCDDocument schema, this test needs to be placed here
    since we are using the VASP TaskDocuments for testing.
    """

    def is_strict_minimum(min_index, arr):
        min_val = arr[min_index]
        return all(
            not (idx != min_index and val < min_val) for idx, val in enumerate(arr)
        )

    static_tasks1: list[TaskDoc] = []
    static_tasks2: list[TaskDoc] = []
    static_dirs1: list[str] = []
    static_dirs2: list[str] = []
    for idx in range(5):
        sdir1 = vasp_test_dir / "Si_config_coord" / f"static_q1_{idx}" / "outputs"
        sdir2 = vasp_test_dir / "Si_config_coord" / f"static_q2_{idx}" / "outputs"
        static_tasks1.append(TaskDoc.from_directory(sdir1))
        static_tasks2.append(TaskDoc.from_directory(sdir2))
        static_dirs1.append(str(sdir1))
        static_dirs2.append(str(sdir2))

    inputs1 = [
        (task.output.structure, task.output.energy, sdir)
        for task, sdir in zip(static_tasks1, static_dirs1, strict=True)
    ]
    inputs2 = [
        (task.output.structure, task.output.energy, sdir)
        for task, sdir in zip(static_tasks2, static_dirs2, strict=True)
    ]

    input_dict = defaultdict(list)

    for s, e, sdir in inputs1:
        input_dict["structures1"].append(s)
        input_dict["energies1"].append(e)
        input_dict["static_dirs1"].append(sdir)
        input_dict["static_uuids1"].append(sdir)

    for s, e, sdir in inputs2:
        input_dict["structures2"].append(s)
        input_dict["energies2"].append(e)
        input_dict["static_dirs2"].append(sdir)
        input_dict["static_uuids2"].append(sdir)

    input_dict["relaxed_uuid1"] = static_dirs1[2]
    input_dict["relaxed_uuid2"] = static_dirs2[2]

    ccd_doc = CCDDocument.from_task_outputs(**input_dict)

    # create the CCD document
    # ccd_doc = CCDDocument.from_struct_en(static_tasks1, static_tasks2, s0, s1)
    # # check that the middle entry has the lowest energy
    assert is_strict_minimum(2, ccd_doc.energies1)
    assert is_strict_minimum(2, ccd_doc.energies2)
    # # check that you can recreate the task document from the ccd document
    tasks = ccd_doc.get_taskdocs()
    assert len(tasks[0]) == 5
    assert len(tasks[1]) == 5
