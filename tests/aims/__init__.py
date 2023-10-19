"""Utils or testing"""

import gzip
import json
from glob import glob
from pathlib import Path


def compare_files(test_name, work_dir, ref_dir):
    for file in glob(f"{work_dir / test_name}/*in"):
        with open(file) as test_file:
            test_lines = [
                line.strip()
                for line in test_file.readlines()[4:]
                if len(line.strip()) > 0
            ]

        with gzip.open(f"{ref_dir / test_name / Path(file).name}.gz", "rt") as ref_file:
            ref_lines = [
                line.strip()
                for line in ref_file.readlines()[4:]
                if len(line.strip()) > 0
            ]

        assert test_lines == ref_lines

    with open(f"{ref_dir / test_name}/parameters.json") as ref_file:
        ref = json.load(ref_file)
    ref.pop("species_dir", None)

    with open(f"{work_dir / test_name}/parameters.json") as check_file:
        check = json.load(check_file)
    check.pop("species_dir", None)

    assert ref == check
