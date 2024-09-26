from atomate2.jdftx.sets.core import IonicMinSetGenerator, LatticeMinSetGenerator
from atomate2.jdftx.jobs.core import IonicMinMaker, LatticeMinMaker
from pymatgen.core import Structure
from pathlib import Path
from jobflow import run_locally
from atomate2.jdftx.run import run_jdftx

# curr_dir = (Path(__file__).resolve().parent / f"../../../../tests/test_data/structures/Si.cif").resolve()
# si_structure = Structure.from_file(curr_dir)
# maker = LatticeMinMaker(input_set_generator=LatticeMinSetGenerator())
# job = maker.make(si_structure)
# responses = run_locally(job, create_folders=True, ensure_success=True)

run = run_jdftx()

