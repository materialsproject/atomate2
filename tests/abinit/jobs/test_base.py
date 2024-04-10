from __future__ import annotations

from dataclasses import dataclass, field

from abipy.abio.input_tags import SCF
from abipy.abio.inputs import AbinitInput

from atomate2.abinit.jobs.base import BaseAbinitMaker
from atomate2.abinit.sets.base import AbinitInputGenerator, as_pseudo_table


@dataclass
class SomeAISG1(AbinitInputGenerator):
    calc_type: str = "some_calc1"

    param1: int = 1
    param2: float | None = None
    param3: list[int] = field(default_factory=list)
    param4: str = "test_string1"

    user_abinit_settings: dict = field(
        default_factory=lambda: {"extra1": 1, "extra2": 2}
    )

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
    param3: list[int] = field(default_factory=list)
    param4: str = "test_string2"
    param5: dict[int, str] = field(default_factory=lambda: {1: "1"})

    user_abinit_settings: dict = field(default_factory=dict)

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
    input_set_generator: SomeAISG1 = field(default_factory=SomeAISG1)


@dataclass
class SomeMaker2(BaseAbinitMaker):
    name: str = "SomeMaker2 calculation"
    input_set_generator: SomeAISG2 = field(default_factory=SomeAISG2)
