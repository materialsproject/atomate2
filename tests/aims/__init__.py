"""Utils or testing"""

from glob import glob
import json
from pathlib import Path


def compare_files(test_name, work_dir, ref_dir):
    for file in glob(f"{work_dir / test_name}/*in"):
        assert (
            open(file, "rt").readlines()[4:]
            == open(f"{ref_dir / test_name / Path(file).name}", "rt").readlines()[4:]
        )
    ref = json.load(open(f"{ref_dir / test_name}/parameters.json"))
    ref.pop("species_dir", None)
    check = json.load(open(f"{work_dir / test_name}/parameters.json"))
    check.pop("species_dir", None)

    assert ref == check
