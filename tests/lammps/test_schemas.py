from atomate2.lammps.schemas.task import LammpsTaskDocument, StoreTrajectoryOption


def test_task_doc(ref_path):
    ref_output_files = ref_path / "nvt_test" / "outputs"

    task_doc = LammpsTaskDocument.from_directory(
        dir_name=ref_output_files,
        task_label="test_full_store",
        store_trajectory=StoreTrajectoryOption.FULL,
    )
    assert task_doc.task_label == "test_full_store"
    assert task_doc.composition is not None
    assert task_doc.state is not None
    assert task_doc.structure is not None
    assert isinstance(task_doc.raw_log_file, str)
    assert len(task_doc.trajectories[0]) == 1001
    assert task_doc.trajectories[0].frame_properties is not None
    assert len(list(task_doc.dump_files.keys())) == 1
    dump_key = next(iter(task_doc.dump_files.keys()))
    assert dump_key.endswith(".dump")
    assert isinstance(task_doc.dump_files[dump_key], str)

    task_doc = LammpsTaskDocument.from_directory(
        dir_name=ref_output_files,
        task_label="test_no_store",
        store_trajectory=StoreTrajectoryOption.NO,
    )
    assert task_doc.task_label == "test_no_store"
    assert task_doc.state is not None
    assert task_doc.structure is not None
    assert task_doc.trajectories is None
    assert len(list(task_doc.dump_files.keys())) == 1
    dump_key = next(iter(task_doc.dump_files.keys()))
    assert dump_key.endswith(".dump")
    assert isinstance(task_doc.dump_files[dump_key], str)

    task_doc = LammpsTaskDocument.from_directory(
        dir_name=ref_output_files,
        task_label="test_partial_store",
        store_trajectory=StoreTrajectoryOption.PARTIAL,
    )

    assert task_doc.task_label == "test_partial_store"
    assert task_doc.composition is not None
    assert task_doc.state is not None
    assert task_doc.structure is not None
    assert task_doc.trajectories is not None
    assert len(task_doc.trajectories[0]) == 1001
    assert task_doc.trajectories[0].frame_properties is None
