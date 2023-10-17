from pydantic import BaseModel, Field
from pymatgen.io.openmm.schema import SetContents
from pymatgen.io.openmm.sets import OpenMMSet
from atomate2.openmm.schemas.physical_state import PhysicalState


class CalculationInput(BaseModel):
    input_set: OpenMMSet = Field(None, description="Input set for the calculation")
    physical_state: PhysicalState = Field(None, description="Physical state for the calculation")
    contents: SetContents = Field(None, description="Contents of the set")

    @classmethod
    def from_input_set(cls, input_set):
        physical_state = PhysicalState.from_input_set(input_set)
        contents = input_set.inputs[input_set.contents_file].contents
        return CalculationInput(
            input_set=input_set,
            physical_state=physical_state,
            contents=contents,
        )