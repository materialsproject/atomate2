"""This module implements basic kinds of jobs for JDFTx runs."""

import logging
import os
import subprocess

from custodian.custodian import Job

logger = logging.getLogger(__name__)


class JDFTxJob(Job):
    """
    A basic JDFTx job. Runs whatever is in the working directory.
    """

    # If testing, use something like:
    # job = JDFTxJob()
    # job.run()  # assumes input files already written to directory

    # Used Cp2kJob developed by Nick Winner as a template.

    def __init__(
        self,
        jdftx_cmd,
        input_file="jdftx.in",
        output_file="jdftx.out",
        stderr_file="std_err.txt",
    ) -> None:
        """
        This constructor is necessarily complex due to the need for
        flexibility. For standard kinds of runs, it's often better to use one
        of the static constructors. The defaults are usually fine too.

        Args:
            jdftx_cmd (str): Command to run JDFTx as a string.
            input_file (str): Name of the file to use as input to JDFTx
                executable. Defaults to "input.in"
            output_file (str): Name of file to direct standard out to.
                Defaults to "jdftx.out".
            stderr_file (str): Name of file to direct standard error to.
                Defaults to "std_err.txt".
        """
        self.jdftx_cmd = jdftx_cmd
        self.input_file = input_file
        self.output_file = output_file
        self.stderr_file = stderr_file

    def setup(self, directory="./") -> None:
        """
        No setup required.
        """
        pass

    def run(self, directory="./"):
        """
        Perform the actual JDFTx run.

        Returns:
            (subprocess.Popen) Used for monitoring.
        """
        cmd = self.jdftx_cmd + " -i " + self.input_file
        logger.info(f"Running {cmd}")
        with (
            open(os.path.join(directory, self.output_file), "w") as f_std,
            open(os.path.join(directory, self.stderr_file), "w", buffering=1) as f_err,
        ):
            result = subprocess.run([cmd], cwd=directory, stdout=f_std, stderr=f_err, shell=True)

        # Review the return code
        if result.returncode == 0:
            logger.info(f"Command executed successfully with return code {result.returncode}.")
        else:
            logger.error(f"Command failed with return code {result.returncode}.")
        # Optionally, you can log or print additional information here
            with open(os.path.join(directory, self.stderr_file), 'r') as f_err:
                error_output = f_err.read()
            logger.error(f"Standard Error Output:\n{error_output}")
    
        return result

            # use line buffering for stderr
        #    return subprocess.run([cmd], cwd=directory, stdout=f_std, stderr=f_err, shell=True)


    def postprocess(self, directory="./") -> None:
        """No post-processing required."""
        pass

    def terminate(self, directory="./") -> None:
        """Terminate JDFTx."""
        # This will kill any running process with "jdftx" in the name,
        # this might have unintended consequences if running multiple jdftx processes
        # on the same node.
        for cmd in self.jdftx_cmd:
            if "jdftx" in cmd:
                try:
                    os.system(f"killall {cmd}")
                except Exception:
                    pass