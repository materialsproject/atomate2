from atomate2.jdftx.io.JEiter import JEiter


class JEiters(list):
    """
    Class object for collecting and storing a series of SCF steps done between
    geometric optimization steps
    """

    iter_type: str = None
    etype: str = None
    _iter_flag: str = None
    converged: bool = False
    converged_Reason: str = None

    @classmethod
    def from_text_slice(
        cls, text_slice: list[str], iter_type: str = "ElecMinimize", etype: str = "F"
    ):
        """
        Create a JEiters object from a slice of an out file's text corresponding to a series of SCF steps

        Parameters
        ----------
        text_slice : list[str]
            A slice of text from a JDFTx out file corresponding to a series of SCF steps
        iter_type: str
            The type of electronic minimization step
        etype: str
            The type of energy component
        """
        super().__init__([])
        instance = cls()
        instance._iter_flag = f"{iter_type}: Iter:"
        instance.iter_type = iter_type
        instance.etype = etype
        instance.parse_text_slice(text_slice)
        return instance

    def parse_text_slice(self, text_slice: list[str]) -> None:
        """
        Parses a slice of text from a JDFTx out file corresponding to a series of SCF steps

        Parameters
        ----------
        text_slice: list[str]
            A slice of text from a JDFTx out file corresponding to a series of SCF steps
        """
        lines_collect = []
        _iter_flag = f"{self.iter_type}: Iter:"
        for line_text in text_slice:
            if len(line_text.strip()):
                lines_collect.append(line_text)
                if _iter_flag in line_text:
                    self.append(
                        JEiter._from_lines_collect(
                            lines_collect, self.iter_type, self.etype
                        )
                    )
                    lines_collect = []
            else:
                break
        if len(lines_collect):
            self.parse_ending_lines(lines_collect)
            lines_collect = []

    def parse_ending_lines(self, ending_lines: list[str]) -> None:
        """
        Parses the ending lines of text from a JDFTx out file corresponding to a series of SCF steps

        Parameters
        ----------
        ending_lines: list[str]
            The ending lines of text from a JDFTx out file corresponding to a series of SCF steps
        """
        for i, line in enumerate(ending_lines):
            if self.is_converged_line(i, line):
                self.read_converged_line(line)

    def is_converged_line(self, i: int, line_text: str) -> bool:
        """
        Returns True if the line_text is the start of a log message about convergence for a JDFTx optimization step

        Parameters
        ----------
        i: int
            The index of the line in the text slice
        line_text: str
            A line of text from a JDFTx out file

        Returns
        -------
        is_line: bool
            True if the line_text is the start of a log message about convergence for a JDFTx optimization step
        """
        is_line = f"{self.iter_type}: Converged" in line_text
        return is_line

    def read_converged_line(self, line_text: str) -> None:
        """
        Reads the convergence message from a JDFTx optimization step

        Parameters
        ----------
        line_text: str
            A line of text from a JDFTx out file containing a message about convergence for a JDFTx optimization step
        """
        self.converged = True
        self.converged_reason = line_text.split("(")[1].split(")")[0].strip()
