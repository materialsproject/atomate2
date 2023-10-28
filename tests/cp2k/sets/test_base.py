from atomate2.cp2k.sets.base import Cp2kInputSet, recursive_update


def test_cp2k_input_set(cp2k_test_dir, cp2k_test_inputs):
    for input_dir in cp2k_test_inputs:
        cis = Cp2kInputSet.from_directory(input_dir)
        assert cis.is_valid


def test_recursive_update():
    d = {"activate_hybrid": {"hybrid_functional": "HSE06"}}
    u = {"activate_hybrid": {"cutoff_radius": 8}}
    dnew = recursive_update(d, u)

    assert dnew["activate_hybrid"]["hybrid_functional"] == "HSE06"
    assert dnew["activate_hybrid"]["cutoff_radius"] == 8
