from pydantic import BaseModel, Field
from dataclasses import asdict

class TaskDetails(BaseModel):
    name: str = Field(None, description="Task name")
    kwargs: dict = Field(None, description="Task kwargs")
    steps: int = Field(None, description="Total steps")

    @classmethod
    def from_maker(cls, maker):
        maker = asdict(maker)
        return TaskDetails(
            name=maker.get("name"),
            kwargs=maker,
            steps=maker.get("steps", 0),
        )
