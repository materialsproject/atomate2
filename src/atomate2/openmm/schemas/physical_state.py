from pydantic import BaseModel, Field
from typing import Tuple


class PhysicalState(BaseModel):
    box_vectors: Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]] = Field(None, description="Box vectors for the calculation")
    temperature: float = Field(None, description="Temperature for the calculation")
    step_size: float = Field(None, description="Step size for the calculation")
    friction_coefficient: float = Field(None, description="Friction coefficient for the calculation")

    @classmethod
    def from_input_set(cls, input_set):
        integrator = input_set.inputs[input_set.integrator_file].get_integrator()
        state = input_set.inputs[input_set.state_file].get_state()
        vector_array = state.getPeriodicBoxVectors(asNumpy=True)._value
        box_vectors = tuple(tuple(vector) for vector in vector_array)
        temperature = integrator.getTemperature()._value  # kelvin
        step_size = integrator.getStepSize()._value  # picoseconds
        friction_coefficient = integrator.getFriction()._value  # 1/picoseconds

        return PhysicalState(
            box_vectors=box_vectors,
            temperature=temperature,
            step_size=step_size,
            friction_coefficient=friction_coefficient,
        )
