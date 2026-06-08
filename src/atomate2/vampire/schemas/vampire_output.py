from monty.json import MSONable

class VampireOutput(MSONable):
    """This class processes results from a Vampire Monte Carlo simulation
    and parses the critical temperature.
    """

    def __init__(self, parsed_out=None, nmats=None, critical_temp=None):
        """
        Args:
            parsed_out (str): JSON rep of parsed stdout DataFrame.
            nmats (int): Number of distinct materials (1 for each specie and up/down spin).
            critical_temp (float): Monte Carlo Tc result.
        """
        self.parsed_out = parsed_out
        self.nmats = nmats
        self.critical_temp = critical_temp
