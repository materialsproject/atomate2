def test_task_details():
    from atomate2.openmm.schemas.task_details import TaskDetails

    task_details = TaskDetails(
        task_name="my_task",
        task_kwargs={
            "temperature": 100.0,
            "steps": 1000,
            "frequency": 10,
        },
        platform_kwargs={
            "platform": "CPU",
            "platform_properties": None,
        },
        total_steps=1000,
    )
