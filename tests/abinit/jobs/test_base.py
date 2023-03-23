from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pytest
from abipy.abio.input_tags import SCF
from abipy.abio.inputs import AbinitInput

from atomate2.abinit.jobs.base import BaseAbinitMaker
from atomate2.abinit.sets.base import AbinitInputGenerator, as_pseudo_table


@dataclass
class SomeAISG1(AbinitInputGenerator):
    calc_type: str = "some_calc1"

    param1: int = 1
    param2: Optional[float] = None
    param3: List[int] = field(default_factory=list)
    param4: str = "test_string1"

    extra_abivars: dict = field(default_factory=lambda: {"extra1": 1, "extra2": 2})

    restart_from_deps: tuple = (f"{SCF}:WFK|DEN",)

    def get_abinit_input(
        self, structure=None, pseudos=None, prev_outputs=None, **kwargs
    ):
        return AbinitInput(
            structure=structure,
            pseudos=as_pseudo_table(AbinitInputGenerator.pseudos),
        )


@dataclass
class SomeAISG2(AbinitInputGenerator):
    calc_type: str = "some_calc2"

    param1: int = 2
    # param2: Optional[float] = None
    param3: List[int] = field(default_factory=list)
    param4: str = "test_string2"
    param5: Dict[int, str] = field(default_factory=lambda: {1: "1"})

    extra_abivars: dict = field(default_factory=dict)

    restart_from_deps: tuple = (f"{SCF}:WFK|DEN",)

    def get_abinit_input(
        self, structure=None, pseudos=None, prev_outputs=None, **kwargs
    ):
        return AbinitInput(
            structure=structure,
            pseudos=as_pseudo_table(AbinitInputGenerator.pseudos),
        )


@dataclass
class SomeMaker1(BaseAbinitMaker):
    name: str = "SomeMaker1 calculation"
    input_set_generator: SomeAISG1 = field(default_factory=lambda: SomeAISG1())


@dataclass
class SomeMaker2(BaseAbinitMaker):
    name: str = "SomeMaker2 calculation"
    input_set_generator: SomeAISG2 = field(default_factory=lambda: SomeAISG2())


def test_maker_from_params():
    with pytest.raises(
        TypeError,
        match=r"SomeMaker1.from_params\(\) got "
        r"an unexpected keyword argument 'param8'",
    ):
        SomeMaker1.from_params(param8=2)
    maker1_1 = SomeMaker1.from_params(param2=3.0, param3=[1, 2])
    assert isinstance(maker1_1.input_set_generator, SomeAISG1)
    assert maker1_1.input_set_generator.extra_abivars == {"extra1": 1, "extra2": 2}
    assert maker1_1.input_set_generator.param1 == 1
    assert maker1_1.input_set_generator.param2 == 3.0
    assert maker1_1.input_set_generator.param3 == [1, 2]
    assert maker1_1.name == "SomeMaker1 calculation"
    assert maker1_1.wall_time is None
    maker1_2 = SomeMaker1(name="myname", wall_time=5)
    assert isinstance(maker1_2.input_set_generator, SomeAISG1)
    assert maker1_2.name == "myname"
    assert maker1_2.wall_time == 5
    maker1_3 = SomeMaker1(input_set_generator=SomeAISG2())
    assert isinstance(maker1_3.input_set_generator, SomeAISG2)
    assert maker1_3.name == "SomeMaker1 calculation"
    maker1_4 = SomeMaker1.from_params(extra_abivars={"extra5": 5})
    assert maker1_4.input_set_generator.extra_abivars == {"extra5": 5}


def test_maker_from_prev_maker():
    maker1_1 = SomeMaker1.from_params(param2=3.0, param3=[1, 2])
    maker2_1 = SomeMaker2.from_prev_maker(maker1_1, param4="explicit_param4")
    assert maker2_1.input_set_generator.param1 == 1
    assert maker2_1.input_set_generator.param3 == [1, 2]
    assert maker2_1.input_set_generator.param4 == "explicit_param4"
    assert maker2_1.input_set_generator.param5 == {1: "1"}
    assert maker2_1.input_set_generator.extra_abivars == {"extra1": 1, "extra2": 2}
    maker2_2 = SomeMaker2.from_prev_maker(maker1_1, extra_abivars={"extra1": 5})
    assert maker2_2.input_set_generator.extra_abivars == {"extra1": 5, "extra2": 2}
    maker2_3 = SomeMaker2.from_prev_maker(
        maker1_1, extra_abivars={"extra2": None, "extra3": 3}
    )
    assert maker2_3.input_set_generator.extra_abivars == {"extra1": 1, "extra3": 3}
