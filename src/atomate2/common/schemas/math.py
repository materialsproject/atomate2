"""Schemas for matrices and vectors."""

from typing import Tuple

__all__ = ["Vector3D", "Vector6D", "Matrix3D", "MatrixVoigt"]


Vector3D = Tuple[float, float, float]
Vector3D.__doc__ = "Real space vector"  # type: ignore

Matrix3D = Tuple[Vector3D, Vector3D, Vector3D]
Matrix3D.__doc__ = "Real space Matrix"  # type: ignore

Vector6D = Tuple[float, float, float, float, float, float]
Vector6D.__doc__ = "6D Voigt matrix component"  # type: ignore

MatrixVoigt = Tuple[Vector6D, Vector6D, Vector6D, Vector6D, Vector6D, Vector6D]
Vector6D.__doc__ = "Voigt representation of a 3x3x3x3 tensor"  # type: ignore
