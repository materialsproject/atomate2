import pytest

from atomate2.forcefields import MLFF
from atomate2.forcefields.utils import ase_calculator


@pytest.mark.parametrize(("force_field"), ["CHGNet", "MACE"])
def test_ext_load(force_field: str):
    decode_dict = {
        "CHGNet": {"@module": "chgnet.model.dynamics", "@callable": "CHGNetCalculator"},
        "MACE": {"@module": "mace.calculators", "@callable": "mace_mp"},
    }[force_field]
    calc_from_decode = ase_calculator(decode_dict)
    calc_from_preset = ase_calculator(str(MLFF(force_field)))
    assert type(calc_from_decode) is type(calc_from_preset)
    assert calc_from_decode.name == calc_from_preset.name
    assert calc_from_decode.parameters == calc_from_preset.parameters == {}


def test_raises_error():
    with pytest.raises(ValueError, match="Could not create"):
        ase_calculator("not_a_calculator")


@pytest.mark.skip(reason="M3GNet requires DGL which is PyTorch 2.4 incompatible")
def test_m3gnet_pot():
    import matgl
    from matgl.ext.ase import PESCalculator

    kwargs_calc = {"path": "M3GNet-MP-2021.2.8-DIRECT-PES", "stress_weight": 2.0}
    kwargs_default = {"stress_weight": 2.0}

    m3gnet_calculator = ase_calculator(calculator_meta="MLFF.M3GNet", **kwargs_calc)

    # uses "M3GNet-MP-2021.2.8-PES" per default
    m3gnet_default = ase_calculator(calculator_meta="MLFF.M3GNet", **kwargs_default)

    potential = matgl.load_model("M3GNet-MP-2021.2.8-DIRECT-PES")
    m3gnet_pes_calc = PESCalculator(potential=potential, stress_weight=2.0)

    assert str(m3gnet_pes_calc.potential) == str(m3gnet_calculator.potential)
    # casting necessary because <class 'matgl.apps.pes.Potential'> can't be compared
    assert str(m3gnet_pes_calc.potential) != str(m3gnet_default.potential)
    assert m3gnet_pes_calc.stress_weight == m3gnet_calculator.stress_weight
    assert m3gnet_pes_calc.stress_weight == m3gnet_default.stress_weight
