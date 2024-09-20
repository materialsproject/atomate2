import pytest

from atomate2.openmm.utils import download_opls_xml, increment_name


@pytest.mark.skip("annoying test")
def test_download_xml(tmp_path):
    pytest.importorskip("selenium")

    download_opls_xml("CCO", tmp_path / "CCO.xml")

    assert (tmp_path / "CCO.xml").exists()


def test_increment_file_name():
    test_cases = [
        ("report", "report2"),
        ("report123", "report124"),
        ("report.123", "report.124"),
        ("report-123", "report-124"),
        ("report-dcd", "report-dcd2"),
        ("report.123.dcd", "report.123.dcd2"),
    ]

    for file_name, expected_output in test_cases:
        result = increment_name(file_name)
        assert (
            result == expected_output
        ), f"Failed for case: {file_name}. Expected: {expected_output}, Got: {result}"
