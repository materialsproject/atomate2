#!/usr/bin/env python

# Optional for debugging, use via "pip install stackprinter" first:
import stackprinter
import os

stackprinter.set_excepthook(style="darkbg2")

# if running from directory containing jobs.py for testing, can also use "from jobs import JDFTxJob"
from atomate2.jdftx.jobs.jobs import JDFTxJob

# assumes running this script from directory containing already-generated input files
# if input files are in a different directory, change "$PWD" below
pwd = os.getcwd()

job = JDFTxJob(
    jdftx_cmd=f"docker run -t --rm -v {pwd}:/root/research jdftx jdftx",
    input_file="input-tutorial.in",
)
job.run()
