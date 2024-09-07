from jdftx.io.JStructure import JStructure


from dataclasses import dataclass


@dataclass
class JStructures(list[JStructure]):

    '''
    A class for storing a series of JStructure objects
    '''

    out_slice_start_flag = "-------- Electronic minimization -----------"
    iter_type: str = None
    geom_converged: bool = False
    geom_converged_reason: str = None
    elec_converged: bool = False
    elec_converged_reason: str = None


    @classmethod
    def from_out_slice(cls, out_slice: list[str], iter_type: str = "IonicMinimize"):
        '''
        Create a JStructures object from a slice of an out file's text corresponding
        to a single JDFTx call

        Args:
            out_slice (list[str]): A slice of a JDFTx out file (individual call of JDFTx)
        '''
        super().__init__([])
        instance = cls()
        if not iter_type in ["IonicMinimize", "LatticeMinimize"]:
            iter_type = instance.correct_iter_type(iter_type)
        instance.iter_type = iter_type
        start_idx = instance.get_start_idx(out_slice)
        instance.parse_out_slice(out_slice[start_idx:])
        if instance.iter_type is None and len(instance) > 1:
            raise Warning("iter type interpreted as single-point calculation, but \
                           multiple structures found")
        return instance


    def correct_iter_type(self, iter_type: str) -> str:
        '''
        Corrects the iter_type to a recognizable string if it is not recognized
        (None may correspond to a single-point calculation)

        Args:
            iter_type (str): The iter_type to be corrected
        '''
        if "lattice" in iter_type.lower():
            iter_type = "LatticeMinimize"
        elif "ionic" in iter_type.lower():
            iter_type = "IonicMinimize"
        else:
            iter_type = None
        return iter_type


    def get_start_idx(self, out_slice: list[str]) -> int:
        '''
        Returns the index of the first line of the first structure in the out_slice

        Args:
            out_slice (list[str]): A slice of a JDFTx out file (individual call of JDFTx)
        '''
        for i, line in enumerate(out_slice):
            if self.out_slice_start_flag in line:
                return i
        return


    def is_lowdin_start_line(self, line_text: str) -> bool:
        '''
        Check if a line in the out file is the start of a Lowdin population analysis

        Args:
            line_text (str): A line of text from a JDFTx out file
        '''
        is_line = "#--- Lowdin population analysis ---" in line_text
        return is_line


    def get_step_bounds(self, out_slice: list[str]) -> list[list[int, int]]:
        '''
        Returns a list of lists of integers where each sublist contains the start and end
        of an individual optimization step (or SCF cycle if no optimization)
        '''
        bounds_list = []
        bounds = None
        end_started = False
        for i, line in enumerate(out_slice):
            if not end_started:
                if self.out_slice_start_flag in line:
                    bounds = [i]
                elif not bounds is None:
                    if self.is_lowdin_start_line(line):
                        end_started = True
            elif not len(line.strip()):
                bounds.append(i)
                bounds_list.append(bounds)
                bounds = None
                end_started = False
        return bounds_list

    def parse_out_slice(self, out_slice: list[str]) -> None:
        '''
        Set relevant variables for the JStructures object by parsing the out_slice

        Args:
            out_slice (list[str]): A slice of a JDFTx out file (individual call of JDFTx)
        '''
        out_bounds = self.get_step_bounds(out_slice)
        for bounds in out_bounds:
            self.append(JStructure.from_text_slice(out_slice[bounds[0]:bounds[1]],
                                                   iter_type=self.iter_type))

    def check_convergence(self) -> None:
        '''
        Check if the geometry and electronic density of last structure in the list has converged
        '''
        jst = self[-1]
        if jst.elecMinData.converged:
            self.elec_converged = True
            self.elec_converged_reason = jst.elecMinData.converged_reason
        if jst.geom_converged:
            self.geom_converged = True
            self.geom_converged_reason = jst.geom_converged_reason