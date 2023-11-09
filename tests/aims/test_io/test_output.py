# import json
# from pathlib import Path

# from monty.json import MontyDecoder

# from atomate2.aims.io.aims_output import AimsOutput

# outfile_dir = Path(__file__).parent / "aims_output_files"


# def test_output_si():
#     si = AimsOutput.from_outfile(f"{outfile_dir}/si.out")
#     with open(f"{outfile_dir}/si_out.json") as ref_file:
#         si_ref = json.load(ref_file, cls=MontyDecoder)

#     assert si_ref.metadata == si.metadata
#     assert si_ref.atoms_summary == si.atoms_summary

#     assert si_ref.n_images == si.n_images
#     for ii in range(si.n_images):
#         assert si_ref.get_results_for_image(ii) == si.get_results_for_image(ii)

#     assert si.as_dict() == si_ref.as_dict()


# def test_output_h2o():
#     h2o = AimsOutput.from_outfile(f"{outfile_dir}/h2o.out")
#     with open(f"{outfile_dir}/h2o_out.json") as ref_file:
#         h2o_ref = json.load(ref_file, cls=MontyDecoder)

#     assert h2o_ref.metadata == h2o.metadata
#     assert h2o_ref.atoms_summary == h2o.atoms_summary

#     assert h2o_ref.n_images == h2o.n_images
#     for ii in range(h2o.n_images):
#         assert h2o_ref.get_results_for_image(ii) == h2o.get_results_for_image(ii)

#     assert h2o.as_dict() == h2o_ref.as_dict()
