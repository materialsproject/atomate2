#!/usr/bin/env python3

from JDFTXOutfile import JDFTXOutfile


filename = 'jdftx.out'
jout = JDFTXOutfile.from_file(filename)
print(jout)

