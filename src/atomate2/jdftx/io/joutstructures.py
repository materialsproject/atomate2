from dataclasses import dataclass
from typing import Any

from atomate2.jdftx.io.joutstructure import JOutStructure, correct_iter_type, is_lowdin_start_line


def get_start_idx(out_slice: list[str], out_slice_start_flag: str = "-------- Electronic minimization -----------") -> int:
    """
    Returns the index of the first line of the first structure in the out_slice

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


def get_step_bounds(out_slice: list[str], out_slice_start_flag: str = "-------- Electronic minimization -----------") -> list[list[int, int]]:
    """
    Returns a list of lists of integers where each sublist contains the start and end
    of an individual optimization step (or SCF cycle if no optimization)

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
            elif bounds is not None:
                if is_lowdin_start_line(line):
                    end_started = True
        elif not len(line.strip()):
            bounds.append(i)
            bounds_list.append(bounds)
            bounds = None
            end_started = False
    return bounds_list





@dataclass
class JOutStructures(list[JOutStructure]):
    """
    A class for storing a series of JStructure objects
    """

    out_slice_start_flag = "-------- Electronic minimization -----------"
    iter_type: str = None
    geom_converged: bool = False
    geom_converged_reason: str = None
    elec_converged: bool = False
    elec_converged_reason: str = None
    _t_s: float = None
    

    @classmethod
    def from_out_slice(cls, out_slice: list[str], iter_type: str = "IonicMinimize"):
        """
        Create a JStructures object from a slice of an out file's text corresponding
        to a single JDFTx call

        Parameters
        ----------
        out_slice: list[str]
            A slice of a JDFTx out file (individual call of JDFTx)
        """
        instance = cls([])
        if not iter_type in ["IonicMinimize", "LatticeMinimize"]:
            iter_type = correct_iter_type(iter_type)
        instance.iter_type = iter_type
        start_idx = get_start_idx(out_slice)
        instance.set_JOutStructure_list(out_slice[start_idx:])
        if instance.iter_type is None and len(instance) > 1:
            raise Warning(
                "iter type interpreted as single-point calculation, but \
                           multiple structures found"
            )
        return instance

    @property
    def t_s(self) -> float:
        """
        Returns the total time in seconds for the calculation

        Returns
        -------
        t_s: float
            The total time in seconds for the calculation
        """
        if self._t_s is not None:
            return self._t_s
        if len(self):
            if self.iter_type in ["single point", None]:
                self._t_s = self[-1].elecMinData[-1].t_s
            else:
                self._t_s = self[-1].t_s
        return self._t_s
    

    def get_JOutStructure_list(self, out_slice: list[str]) -> list[JOutStructure]:
        """
        Set relevant variables for the JStructures object by parsing the out_slice

        Parameters
        ----------
        out_slice: list[str]
            A slice of a JDFTx out file (individual call of JDFTx)
        """
        out_bounds = get_step_bounds(out_slice)
        out_list = []
        print(self.iter_type)
        for bounds in out_bounds:
            out_list.append(
                JOutStructure.from_text_slice(
                    out_slice[bounds[0] : bounds[1]], iter_type=self.iter_type
                )
            )
        return out_list
    
    
    def set_JOutStructure_list(self, out_slice: list[str]) -> None:
        out_list = self.get_JOutStructure_list(out_slice)
        for jos in out_list:
            self.append(jos)


    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if self:
                return getattr(self[-1], name)
            raise AttributeError(f"'JStructures' object has no attribute '{name}'")


    


    

    def check_convergence(self) -> None:
        """
        Check if the geometry and electronic density of last structure in the list has converged
        """
        jst = self[-1]
        if jst.elecMinData.converged:
            self.elec_converged = True
            self.elec_converged_reason = jst.elecMinData.converged_reason
        if jst.geom_converged:
            self.geom_converged = True
            self.geom_converged_reason = jst.geom_converged_reason



# @dataclass
# class JOutStructures(list[JOutStructure], JOutStructure):
#     """
#     A class for storing a series of JStructure objects
#     """

#     out_slice_start_flag = "-------- Electronic minimization -----------"
#     iter_type: str = None
#     geom_converged: bool = False
#     geom_converged_reason: str = None
#     elec_converged: bool = False
#     elec_converged_reason: str = None
#     _t_s: float = None

#     def __init__(self, *args: Any, **kwargs: Any):
#         super().__init__(*args, **kwargs)
#         self._t_s = None

#     @classmethod
#     def from_out_slice(cls, out_slice: list[str], iter_type: str = "IonicMinimize"):
#         """
#         Create a JStructures object from a slice of an out file's text corresponding
#         to a single JDFTx call

#         Parameters
#         ----------
#         out_slice: list[str]
#             A slice of a JDFTx out file (individual call of JDFTx)
#         """
#         instance = cls()
#         if iter_type not in ["IonicMinimize", "LatticeMinimize"]:
#             iter_type = instance.correct_iter_type(iter_type)
#         instance.iter_type = iter_type
#         start_idx = instance.get_start_idx(out_slice)
#         instance.parse_out_slice(out_slice[start_idx:])
#         if instance.iter_type is None and len(instance) > 1:
#             raise Warning(
#                 "iter type interpreted as single-point calculation, but \
#                            multiple structures found"
#             )
#         return instance

#     @property
#     def t_s(self) -> float:
#         """
#         Returns the total time in seconds for the calculation

#         Returns
#         -------
#         t_s: float
#             The total time in seconds for the calculation
#         """
#         if self._t_s is not None:
#             return self._t_s
#         if len(self):
#             if self.iter_type in ["single point", None]:
#                 self._t_s = self[-1].elecMinData[-1].t_s
#             else:
#                 self._t_s = self[-1].t_s
#         return self._t_s

#     def __getattr__(self, name):
#         try:
#             return super().__getattr__(name)
#         except AttributeError:
#             if self:
#                 return getattr(self[-1], name)
#             raise AttributeError(f"'JStructures' object has no attribute '{name}'")


#     def get_start_idx(self, out_slice: list[str]) -> int:
#         """
#         Returns the index of the first line of the first structure in the out_slice

#         Parameters
#         ----------
#         out_slice: list[str]
#             A slice of a JDFTx out file (individual call of JDFTx)

#         Returns
#         -------
#         i: int
#             The index of the first line of the first structure in the out_slice
#         """
#         for i, line in enumerate(out_slice):
#             if self.out_slice_start_flag in line:
#                 return i
#         return i


#     def get_step_bounds(self, out_slice: list[str]) -> list[list[int, int]]:
#         """
#         Returns a list of lists of integers where each sublist contains the start and end
#         of an individual optimization step (or SCF cycle if no optimization)

#         Parameters
#         ----------
#         out_slice: list[str]
#             A slice of a JDFTx out file (individual call of JDFTx)

#         Returns
#         -------
#         bounds_list: list[list[int, int]]
#             A list of lists of integers where each sublist contains the start and end
#             of an individual optimization step (or SCF cycle if no optimization)
#         """
#         bounds_list = []
#         bounds = None
#         end_started = False
#         for i, line in enumerate(out_slice):
#             if not end_started:
#                 if self.out_slice_start_flag in line:
#                     bounds = [i]
#                 elif bounds is not None:
#                     if self.is_lowdin_start_line(line):
#                         end_started = True
#             elif not len(line.strip()):
#                 bounds.append(i)
#                 bounds_list.append(bounds)
#                 bounds = None
#                 end_started = False
#         return bounds_list

#     def parse_out_slice(self, out_slice: list[str]) -> None:
#         """
#         Set relevant variables for the JStructures object by parsing the out_slice

#         Parameters
#         ----------
#         out_slice: list[str]
#             A slice of a JDFTx out file (individual call of JDFTx)
#         """
#         out_bounds = self.get_step_bounds(out_slice)
#         for bounds in out_bounds:
#             self.append(
#                 JOutStructure.from_text_slice(
#                     out_slice[bounds[0] : bounds[1]], iter_type=self.iter_type
#                 )
#             )

#     def check_convergence(self) -> None:
#         """
#         Check if the geometry and electronic density of last structure in the list has converged
#         """
#         jst = self[-1]
#         if jst.elecMinData.converged:
#             self.elec_converged = True
#             self.elec_converged_reason = jst.elecMinData.converged_reason
#         if jst.geom_converged:
#             self.geom_converged = True
#             self.geom_converged_reason = jst.geom_converged_reason
