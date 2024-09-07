from pymatgen.core.units import Ha_to_eV


class JEiter():
    '''
    Class object for storing logged electronic minimization data for a single SCF step
    '''
    iter_type: str = None
    etype: str = None
    #
    iter: int = None
    E: float = None
    grad_K: float = None
    alpha: float = None
    linmin: float = None
    t_s: float = None
    #
    mu: float = None
    nElectrons: float = None
    abs_magneticMoment: float = None
    tot_magneticMoment: float = None
    subspaceRotationAdjust: float = None
    #
    converged: bool = False
    converged_reason: str = None


    @classmethod
    def _from_lines_collect(cls, lines_collect: list[str], iter_type: str, etype: str):
        '''
        Create a JEiter object from a list of lines of text from a JDFTx out file corresponding to a single SCF step

        Parameters:
        ----------
        lines_collect: list[str]
            A list of lines of text from a JDFTx out file corresponding to a single SCF step
        iter_type: str
            The type of electronic minimization step
        etype: str
            The type of energy component
        '''
        instance = cls()
        instance.iter_type = iter_type
        instance.etype = etype
        _iter_flag = f"{iter_type}: Iter: "
        for i, line_text in enumerate(lines_collect):
            if instance.is_iter_line(i, line_text, _iter_flag):
                instance.read_iter_line(line_text)
            elif instance.is_fillings_line(i, line_text):
                instance.read_fillings_line(line_text)
            elif instance.is_subspaceadjust_line(i, line_text):
                instance.read_subspaceadjust_line(line_text)
        return instance

    def is_iter_line(self, i: int, line_text: str, _iter_flag: str) -> bool:
        '''
        Returns True if the line_text is the start of a log message for a JDFTx optimization step

        Parameters:
        ----------
        i: int
            The index of the line in the text slice
        line_text: str 
            A line of text from a JDFTx out file
        _iter_flag:  str
            The flag that indicates the start of a log message for a JDFTx optimization step

        Returns:
        -------
        is_line: bool
            True if the line_text is the start of a log message for a JDFTx optimization step
        '''
        is_line = _iter_flag in line_text
        return is_line

    def read_iter_line(self, line_text: str) -> None:
        '''
        Parses the lines of text corresponding to the electronic minimization data of a JDFTx out file

        Parameters:
        ----------
        line_text: str
            A line of text from a JDFTx out file containing the electronic minimization data
        '''

        self.iter = self._get_colon_var_t1(line_text, "Iter: ")
        self.E = self._get_colon_var_t1(line_text, f"{self.etype}: ") * Ha_to_eV
        self.grad_K = self._get_colon_var_t1(line_text, "|grad|_K: ")
        self.alpha = self._get_colon_var_t1(line_text, "alpha: ")
        self.linmin = self._get_colon_var_t1(line_text, "linmin: ")
        self.t_s = self._get_colon_var_t1(line_text, "t[s]: ")


    def is_fillings_line(self, i: int, line_text: str) -> bool:
        '''
        Returns True if the line_text is the start of a log message for a JDFTx optimization step

        Parameters:
        ----------
        i (int): int
            The index of the line in the text slice
        line_text: str
            A line of text from a JDFTx out file

        Returns:
        -------
        is_line: bool
            True if the line_text is the start of a log message for a JDFTx optimization step
        '''
        is_line = "FillingsUpdate" in line_text
        return is_line


    def read_fillings_line(self, fillings_line: str) -> None:
        '''
        Parses the lines of text corresponding to the electronic minimization data of a JDFTx out file

        Parameters:
        ----------
        fillings_line: str
            A line of text from a JDFTx out file containing the electronic minimization data
        '''
        assert "FillingsUpdate:" in fillings_line
        self.set_mu(fillings_line)
        self.set_nElectrons(fillings_line)
        if "magneticMoment" in fillings_line:
            self.set_magdata(fillings_line)


    def is_subspaceadjust_line(self, i: int, line_text: str) -> bool:
        '''
        Returns True if the line_text is the start of a log message for a JDFTx optimization step

        Parameters:
        ----------
        i: int
            The index of the line in the text slice
        line_text: str
            A line of text from a JDFTx out file

        Returns:
        -------
        is_line: bool
            True if the line_text is the start of a log message for a JDFTx optimization step
        '''
        is_line = "SubspaceRotationAdjust" in line_text
        return is_line


    def read_subspaceadjust_line(self, line_text: str) -> None:
        '''
        Parses the lines of text corresponding to the electronic minimization data of a JDFTx out file

        Parameters:
        ----------
        line_text: str
            A line of text from a JDFTx out file containing the electronic minimization data
        '''
        self.subspaceRotationAdjust = self._get_colon_var_t1(line_text, "SubspaceRotationAdjust: set factor to")



    def set_magdata(self, fillings_line: str) -> None:
        '''
        Parses the lines of text corresponding to the electronic minimization data of a JDFTx out file

        Parameters:
        ----------
        fillings_line: str
            A line of text from a JDFTx out file containing the electronic minimization data
        '''
        _fillings_line = fillings_line.split("magneticMoment: [ ")[1].split(" ]")[0].strip()
        self.abs_magneticMoment = self._get_colon_var_t1(_fillings_line, "Abs: ")
        self.tot_magneticMoment = self._get_colon_var_t1(_fillings_line, "Tot: ")


    def _get_colon_var_t1(self, linetext: str, lkey: str) -> float | None:
        '''
        Reads a float from an elec minimization line assuming value appears as
        "... lkey value ..."

        Parameters:
        ----------
        linetext: str
            A line of text from a JDFTx out file
        lkey: str
            The key to search for in the line of text

        Returns:
        -------
        colon_var: float | None 
            The float value found in the line of text
        '''
        colon_var = None
        if lkey in linetext:
            colon_var = float(linetext.split(lkey)[1].strip().split(" ")[0])
        return colon_var


    def set_mu(self, fillings_line: str) -> None:
        '''
        Parses the lines of text corresponding to the electronic minimization data of a JDFTx out file

        Parameters:
        ----------
        fillings_line: str
            A line of text from a JDFTx out file containing the electronic minimization data
        '''
        self.mu = self._get_colon_var_t1(fillings_line, "mu: ") * Ha_to_eV


    def set_nElectrons(self, fillings_line: str) -> None:
        '''
        Parses the lines of text corresponding to the electronic minimization data of a JDFTx out file

        Parameters:
        ----------
        fillings_line: str
            A line of text from a JDFTx out file containing the electronic minimization data
        '''
        self.nElectrons = self._get_colon_var_t1(fillings_line, "nElectrons: ")