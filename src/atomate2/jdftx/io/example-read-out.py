#!/usr/bin/env python3

from JDFTXOutfile import JDFTXOutfile
from pathlib import Path

filename = "jdftx.out"
jout = JDFTXOutfile.from_file(filename)
print(jout)
