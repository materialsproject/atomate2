import yaml
from jobflow import run_locally
from yaml import CLoader as Loader

from atomate2.common.jobs.phonons import PhononBSDOSDoc
from atomate2.common.jobs.qha import analyze_free_energy


def test_analyze_free_energy(clean_dir, test_dir):
    # The following code and the test files have been adapted from Phonopy
    # Copyright (C) 2015 Atsushi Togo
    # All rights reserved.
    #
    # This file is part of phonopy.
    #
    # Redistribution and use in source and binary forms, with or without
    # modification, are permitted provided that the following conditions
    # are met:
    #
    # * Redistributions of source code must retain the above copyright
    #   notice, this list of conditions and the following disclaimer.
    #
    # * Redistributions in binary form must reproduce the above copyright
    #   notice, this list of conditions and the following disclaimer in
    #   the documentation and/or other materials provided with the
    #   distribution.
    #
    # * Neither the name of the phonopy project nor the names of its
    #   contributors may be used to endorse or promote products derived
    #   from this software without specific prior written permission.
    #
    # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    # "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    # LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
    # FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
    # COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    # INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
    # BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    # LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    # LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
    # ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    # POSSIBILITY OF SUCH DAMAGE.

    volumes = []
    energies = []
    for line in open(f"{test_dir}/forcefields/qha/e-v.dat"):
        v, e = line.split()
        volumes.append(float(v))
        energies.append(float(e))

    phonon_docs = []
    for index, energy, volume in zip(range(-5, 6), energies, volumes):
        filename = f"{test_dir}/forcefields/qha/thermal_properties.yaml-{str(index)}"
        thermal_properties = yaml.load(open(filename), Loader=Loader)["thermal_properties"]
        temperatures = [v["temperature"] for v in thermal_properties]
        cv = [v["heat_capacity"] for v in thermal_properties]
        entropy = [v["entropy"] for v in thermal_properties]
        fe = [v["free_energy"] for v in thermal_properties]

        phonon_docs.append(
            PhononBSDOSDoc(free_energies=fe, heat_capacities=cv, entropies=entropy, temperatures=temperatures,
                           total_dft_energy=energy, volume_per_formula_unit=volume, formula_units=1))

    job = analyze_free_energy(phonon_docs)

    responses=run_locally(job)
    print(responses)
    result = responses[job.uuid][1].output
    assert result == False
    #assert isinstance(result, QHADoc)
