def test_state_reports(test_state_report_file):
    from atomate2.openmm.schemas.state_reports import StateReports
    import pandas as pd

    state_reporter = StateReports.from_state_file(test_state_report_file)

    state_report_df = pd.read_csv(filepath_or_buffer=test_state_report_file)

    num_records = state_report_df.shape[0]

    assert len(state_reporter.kinetic_energy) == num_records
    assert len(state_reporter.potential_energy) == num_records
    assert len(state_reporter.steps) == num_records
    assert len(state_reporter.temperature) == num_records
    assert len(state_reporter.total_energy) == num_records
