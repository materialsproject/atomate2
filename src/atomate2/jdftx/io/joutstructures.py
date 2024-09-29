"""Module for JOutStructures class.

This module contains the JOutStructures class for storing a series of
JOutStructure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pymatgen.io.jdftx.joutstructure import JOutStructure
from pymatgen.io.jdftx.joutstructure_helpers import (
    correct_iter_type,
    is_lowdin_start_line,
)

elec_min_start_flag: str = "-------- Electronic minimization -----------"


def get_start_idx(
    out_slice: list[str],
    out_slice_start_flag: str = elec_min_start_flag,
) -> int:
    """Return index of first line of first structure.

    Return the index of the first line of the first structure in the out_slice.

    Parameters
    ----------
    out_slice: list[str]
        A slice of a JDFTx out file (individual call of JDFTx)

    Returns
    -------
    i: int
        The index of the first line of the first structure in the out_slice
    """
    for i, line in enumerate(out_slice):
        if out_slice_start_flag in line:
            return i
    return i


def get_step_bounds(
    out_slice: list[str],
    out_slice_start_flag: str = elec_min_start_flag,
) -> list[list[int]]:
    """Return list of boundary indices for each structure in out_slice.

    Return a list of lists of integers where each sublist contains the start and end
    of an individual optimization step (or SCF cycle if no optimization).

    Parameters
    ----------
    out_slice: list[str]
        A slice of a JDFTx out file (individual call of JDFTx)

    Returns
    -------
    bounds_list: list[list[int, int]]
        A list of lists of integers where each sublist contains the start and end
        of an individual optimization step (or SCF cycle if no optimization)
    """
    bounds_list = []
    bounds = None
    end_started = False
    for i, line in enumerate(out_slice):
        if not end_started:
            if out_slice_start_flag in line:
                bounds = [i]
            elif (bounds is not None) and (is_lowdin_start_line(line)):
                end_started = True
        elif not len(line.strip()):
            bounds.append(i)
            bounds_list.append(bounds)
            bounds = None
            end_started = False
    return bounds_list


@dataclass
class JOutStructures:
    """Class for storing a series of JStructure objects.

    A class for storing a series of JStructure objects.
    """

    out_slice_start_flag = "-------- Electronic minimization -----------"
    iter_type: str = None
    geom_converged: bool = False
    geom_converged_reason: str = None
    elec_converged: bool = False
    elec_converged_reason: str = None
    _t_s: float = None
    slices: list[JOutStructure] = field(default_factory=list)

    @classmethod
    def from_out_slice(
        cls, out_slice: list[str], iter_type: str = "IonicMinimize"
    ) -> JOutStructures:
        """Return JStructures object.

        Create a JStructures object from a slice of an out file's text
        corresponding to a single JDFTx call.

        Parameters
        ----------
        out_slice: list[str]
            A slice of a JDFTx out file (individual call of JDFTx)
        """
        instance = cls()
        if iter_type not in ["IonicMinimize", "LatticeMinimize"]:
            iter_type = correct_iter_type(iter_type)
        instance.iter_type = iter_type
        start_idx = get_start_idx(out_slice)
        instance.set_joutstructure_list(out_slice[start_idx:])
        if instance.iter_type is None and len(instance) > 1:
            raise Warning(
                "iter type interpreted as single-point calculation, but \
                           multiple structures found"
            )
        return instance

    @property
    def t_s(self) -> float:
        """Return time of calculation.

        Return the total time in seconds for the calculation.

        Returns
        -------
        t_s: float
            The total time in seconds for the calculation
        """
        if self._t_s is not None:
            return self._t_s
        if len(self):
            if self.iter_type in ["single point", None]:
                self._t_s = self[-1].elecmindata[-1].t_s
            else:
                self._t_s = self[-1].t_s
        return self._t_s

    def get_joutstructure_list(self, out_slice: list[str]) -> list[JOutStructure]:
        """Return list of JOutStructure objects.

        Set relevant variables for the JStructures object by parsing the
        out_slice.

        Parameters
        ----------
        out_slice: list[str]
            A slice of a JDFTx out file (individual call of JDFTx)
        """
        out_bounds = get_step_bounds(out_slice)
        # out_list = [
        #     JOutStructure.from_text_slice(
        #         out_slice[bounds[0] : bounds[1]], iter_type=self.iter_type
        #     )
        #     for bounds in out_bounds
        # ]
        # for bounds in out_bounds:
        #     out_list.append(
        #         JOutStructure.from_text_slice(
        #             out_slice[bounds[0] : bounds[1]], iter_type=self.iter_type
        #         )
        #     )
        return [
            JOutStructure.from_text_slice(
                out_slice[bounds[0] : bounds[1]], iter_type=self.iter_type
            )
            for bounds in out_bounds
        ]

    def set_joutstructure_list(self, out_slice: list[str]) -> None:
        """Set list of JOutStructure objects to slices.

        Set the list of JOutStructure objects to the slices attribute.

        Parameters
        ----------
        out_slice: list[str]
            A slice of a JDFTx out file (individual call of JDFTx)
        """
        out_list = self.get_joutstructure_list(out_slice)
        for jos in out_list:
            self.slices.append(jos)

    def check_convergence(self) -> None:
        """Set convergence flags.

        Check if the geometry and electronic density of last structure in the
        list has converged.
        """
        jst = self.slices[-1]
        if jst.elecmindata.converged:
            self.elec_converged = True
            self.elec_converged_reason = jst.elecmindata.converged_reason
        if jst.geom_converged:
            self.geom_converged = True
            self.geom_converged_reason = jst.geom_converged_reason

    def __getatr__(self, name: str) -> Any:
        """Return attribute value.

        Return the value of an attribute.

        Parameters
        ----------
        name: str
            The name of the attribute

        Returns
        -------
        value
            The value of the attribute
        """
        if not hasattr(self, name):
            if not hasattr(self.slices[-1], name):
                raise AttributeError(f"{self.__class__.__name__} not found: {name}")
            return getattr(self.slices[-1], name)
        return getattr(self, name)

    def __getitem__(self, key: int | str) -> JOutStructure | Any:
        """Return item.

        Return the value of an item.

        Parameters
        ----------
        key: int | str
            The key of the item

        Returns
        -------
        val
            The value of the item
        """
        val = None
        if type(key) is int:
            val = self.getitem_int(key)
        if type(key) is str:
            val = self.getitem_str(key)
        return val

    def getitem_int(self, key: int) -> JOutStructure:
        """Return JOutStructure object.

        Return the JOutStructure object at the key index.

        Parameters
        ----------
        key: int
            The index of the JOutStructure object

        Returns
        -------
        joutstructure: JOutStructure
            The JOutStructure object at the key index
        """
        return self.slices[key]

    def getitem_str(self, key: str) -> Any:
        """Return attribute value.

        Return the value of an attribute.

        Parameters
        ----------
        key: str
            The name of the attribute

        Returns
        -------
        value
            The value of the attribute
        """
        return getattr(self, key)

    def __len__(self) -> int:
        """Return length of JOutStructures object.

        Returns the number of geometric optimization steps in the
        JOutStructures object.

        Returns
        -------
        length: int
            The number of geometric optimization steps in the JOutStructures
            object
        """
        return len(self.slices)
