from atomate2.jdftx.run import run_jdftx

# curr_dir = (Path(__file__).resolve().parent / f"../../../../tests/test_data/structures/Si.cif").resolve()
# si_structure = Structure.from_file(curr_dir)
# maker = LatticeMinMaker(input_set_generator=LatticeMinSetGenerator())
# job = maker.make(si_structure)
# responses = run_locally(job, create_folders=True, ensure_success=True)

run = run_jdftx()
