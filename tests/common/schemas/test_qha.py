from jobflow import run_locally
from numpy.testing import assert_allclose
from pymatgen.core.structure import Structure
from ruamel.yaml import YAML

from atomate2.common.jobs.phonons import PhononBSDOSDoc
from atomate2.common.jobs.qha import PhononQHADoc, analyze_free_energy


def test_analyze_free_energy(tmp_dir, test_dir):
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

    structure = Structure.from_file(f"{test_dir}/qha/POSCAR")

    volumes = []
    energies = []
    with open(f"{test_dir}/qha/e-v.dat") as f:
        for line in f:
            v, e = line.split()
            volumes.append(float(v))
            energies.append(float(e))

    phonon_docs = []
    for index, energy, volume in zip(range(-5, 6), energies, volumes, strict=True):
        filename = f"{test_dir}/qha/thermal_properties.yaml-{index!s}"
        yaml = YAML()
        with open(filename) as f:
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

    final_job = analyze_free_energy(phonon_docs, structure, t_max=300)

    responses = run_locally(final_job, create_folders=True)
    qha_doc = responses[final_job.uuid][1].output
    assert isinstance(qha_doc, PhononQHADoc)


def test_analyze_free_energy_small(tmp_dir, test_dir):
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

    structure = Structure.from_file(f"{test_dir}/qha/POSCAR")

    volumes = []
    energies = []
    with open(f"{test_dir}/qha/e-v.dat") as f:
        for line in f:
            v, e = line.split()
            volumes.append(float(v))
            energies.append(float(e))

    phonon_docs = []
    for index, energy, volume in zip(range(-5, 6), energies, volumes, strict=True):
        filename = f"{test_dir}/qha/thermal_properties.yaml-{index!s}"
        yaml = YAML()
        with open(filename) as f:
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

    final_job = analyze_free_energy(phonon_docs, structure, t_max=100)

    responses = run_locally(final_job, create_folders=True)
    qha_doc = responses[final_job.uuid][1].output
    assert isinstance(qha_doc, PhononQHADoc)

    assert_allclose(qha_doc.free_energies[2][3], 16731.6961, atol=1e-3)
    assert_allclose(
        qha_doc.thermal_expansion,
        [
            0.0,
            1.831119123607067e-09,
            9.148973151361889e-09,
            2.7017789271277713e-08,
            6.088014972218931e-08,
            1.1515165778200777e-07,
            1.9496802680232199e-07,
            3.0641868909807643e-07,
            4.552990401329302e-07,
            6.494501812420191e-07,
            8.974155176557891e-07,
            1.207332028224478e-06,
            1.5862208767416424e-06,
            2.0394834137465825e-06,
            2.570360113076277e-06,
            3.179774554650815e-06,
            3.866321209647544e-06,
            4.627286241852315e-06,
            5.4581864561591606e-06,
            6.353148438780477e-06,
            7.306354204991414e-06,
            8.311190958000843e-06,
            9.360841120203561e-06,
            1.0448694933358668e-05,
            1.1568045652932015e-05,
            1.271299332738757e-05,
            1.3878059279242092e-05,
            1.5057513995703604e-05,
            1.6246352430517004e-05,
            1.744029676489808e-05,
            1.863529302001722e-05,
            1.9827789982046526e-05,
            2.10143402444651e-05,
            2.219226637041667e-05,
            2.335913709158151e-05,
            2.4512770543911004e-05,
            2.56514642633451e-05,
            2.6773402770983187e-05,
            2.7877418959917155e-05,
            2.8962491851875697e-05,
            3.002751181698872e-05,
            3.107196821745603e-05,
            3.209565524167667e-05,
            3.309766802775616e-05,
            3.4077776539955804e-05,
            3.5036207313626614e-05,
            3.597289058097189e-05,
            3.688760488626581e-05,
            3.778051225090235e-05,
            3.8652042179548925e-05,
            3.9502279659813565e-05,
        ],
        atol=1e-8,
    )
    assert qha_doc.pressure is None
    assert qha_doc.volumes == [
        56.51,
        58.31,
        60.15,
        62.03,
        63.95,
        65.91,
        67.9,
        69.94,
        72.02,
        74.14,
        76.29,
    ]
    assert_allclose(qha_doc.temperatures[5], 10.0)
    assert_allclose(qha_doc.heat_capacities[10][9], 2.5144627, atol=1e-3)
    assert_allclose(qha_doc.entropies[10][2], 0.1449301, atol=1e-3)
    assert_allclose(qha_doc.bulk_modulus_temperature[30], 75.0604563982221, atol=1e-3)
    assert_allclose(qha_doc.bulk_modulus, 0.48559238394681514, atol=1e-3)
    assert_allclose(qha_doc.gibbs_temperature[10], -14.814345401430508, atol=1e-3)
    assert_allclose(qha_doc.gruneisen_temperature[5], 3.001253318775402, atol=1e-2)
    assert_allclose(
        qha_doc.heat_capacity_p_numerical[3], 0.024082049058881838, atol=1e-3
    )
    assert_allclose(qha_doc.helmholtz_volume[2][2], -14.631379092600538, atol=1e-3)
    assert_allclose(qha_doc.t_max, 100)
