from pydantic.main import BaseModel


class AtomateModel(BaseModel):
    """Mixin class for base atomate pydantic model."""

    class Config:
        """Pydantic model configuration."""

        extra = "forbid"
