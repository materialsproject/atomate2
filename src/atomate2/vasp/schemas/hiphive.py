"""Schemas for hiphive documents."""

from pydantic import BaseModel


class LTCDoc(BaseModel):
    """Collection to store Lattice thermal conductivity."""
