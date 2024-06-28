from ruamel.yaml import YAML
from jobflow import run_locally
from pymatgen.core.structure import Structure

from atomate2.common.jobs.phonons import PhononBSDOSDoc
from atomate2.common.jobs.qha import PhononQHADoc, analyze_free_energy


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

    structure = Structure.from_file(f"{test_dir}/forcefields/qha/POSCAR")

    volumes = []
    energies = []
    with open(f"{test_dir}/forcefields/qha/e-v.dat", "r") as f:
        for line in f:
            v, e = line.split()
            volumes.append(float(v))
            energies.append(float(e))

    phonon_docs = []
    for index, energy, volume in zip(range(-5, 6), energies, volumes):
        filename = f"{test_dir}/forcefields/qha/thermal_properties.yaml-{index!s}"
        yaml = YAML()
        with open(filename,"r") as f:
            thermal_properties = yaml.load(f)["thermal_properties"]

        temperatures = [v["temperature"] for v in thermal_properties]
        cv = [v["heat_capacity"] for v in thermal_properties]
        entropy = [v["entropy"] for v in thermal_properties]
        fe = [v["free_energy"] * 1000.0 for v in thermal_properties]

        phonon_docs.append(
            PhononBSDOSDoc(
                free_energies=fe,
                heat_capacities=cv,
                entropies=entropy,
                temperatures=temperatures,
                total_dft_energy=energy,
                volume_per_formula_unit=volume,
                formula_units=1,
            )
        )

    job = analyze_free_energy(phonon_docs, structure)

    responses = run_locally(job, create_folders=True)

    assert isinstance(responses[job.uuid][1].output, PhononQHADoc)
