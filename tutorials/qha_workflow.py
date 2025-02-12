#!/usr/bin/env python
# coding: utf-8

# This first part is only needed as we have to mock VASP here as we cannot run it directly in a jupyter notebook:

# In[1]:


from mock_vasp import TEST_DIR, mock_vasp

ref_paths = {
    "phonon static 1/1": "Si_qha_2/phonon_static_1_1",
    "static": "Si_qha_2/static",
    "tight relax 1 EOS equilibrium relaxation": "Si_qha_2/tight_relax_1", # in fact, we replace all relaxation steps with the same output, also the ISIF=4 ones to save storage
    "tight relax 2 EOS equilibrium relaxation": "Si_qha_2/tight_relax_2",
    "tight relax 1 deformation 0": "Si_qha_2/tight_relax_1_d0",
    "tight relax 1 deformation 1": "Si_qha_2/tight_relax_1_d1",
    "tight relax 1 deformation 2": "Si_qha_2/tight_relax_1_d2",
    "tight relax 1 deformation 3": "Si_qha_2/tight_relax_1_d3",
    "tight relax 1 deformation 4": "Si_qha_2/tight_relax_1_d4",
    "tight relax 1 deformation 5": "Si_qha_2/tight_relax_1_d5",
    "tight relax 2 deformation 0": "Si_qha_2/tight_relax_2_d0",
    "tight relax 2 deformation 1": "Si_qha_2/tight_relax_2_d1",
    "tight relax 2 deformation 2": "Si_qha_2/tight_relax_2_d2",
    "tight relax 2 deformation 3": "Si_qha_2/tight_relax_2_d3",
    "tight relax 2 deformation 4": "Si_qha_2/tight_relax_2_d4",
    "tight relax 2 deformation 5": "Si_qha_2/tight_relax_2_d5",
    "dft phonon static eos deformation 1":"Si_qha_2/dft_phonon_static_eos_deformation_1",
    "dft phonon static eos deformation 2":"Si_qha_2/dft_phonon_static_eos_deformation_2",
    "dft phonon static eos deformation 3":"Si_qha_2/dft_phonon_static_eos_deformation_3",
    "dft phonon static eos deformation 4":"Si_qha_2/dft_phonon_static_eos_deformation_4",
    "dft phonon static eos deformation 5":"Si_qha_2/dft_phonon_static_eos_deformation_5",
    "dft phonon static eos deformation 6":"Si_qha_2/dft_phonon_static_eos_deformation_6",
    "dft phonon static eos deformation 7":"Si_qha_2/dft_phonon_static_eos_deformation_7",
    "dft phonon static 1/1 eos deformation 1": "Si_qha_2/dft_phonon_static_1_1_eos_deformation_1",
    "dft phonon static 1/1 eos deformation 2": "Si_qha_2/dft_phonon_static_1_1_eos_deformation_2",
    "dft phonon static 1/1 eos deformation 3": "Si_qha_2/dft_phonon_static_1_1_eos_deformation_3",
    "dft phonon static 1/1 eos deformation 4": "Si_qha_2/dft_phonon_static_1_1_eos_deformation_4",
    "dft phonon static 1/1 eos deformation 5": "Si_qha_2/dft_phonon_static_1_1_eos_deformation_5",
    "dft phonon static 1/1 eos deformation 6": "Si_qha_2/dft_phonon_static_1_1_eos_deformation_6",
    "dft phonon static 1/1 eos deformation 7": "Si_qha_2/dft_phonon_static_1_1_eos_deformation_7",
}


# QHA workflow

# This tutorial will make use of a quasi-harmonic workflow that allows to include volume-dependent anharmonicity into the calculation of phonon free energies. Please check out the paper by Togo to learn about the exact implementation as we will rely on Phonopy to perform the quasi-harmonic approximation. https://doi.org/10.7566/JPSJ.92.012001. At the moment, we perform harmonic free energy calculation along a volume curve to arrive at free energy-volume curves that are the starting point for the quasi-harmonic approximation.

# ## Let's run the workflow
# Now, we load a structure and other important functions and classes for running the qha workflow.

# In[2]:


from jobflow import JobStore, run_locally
from maggma.stores import MemoryStore
from pymatgen.core import Structure

from atomate2.vasp.flows.qha import QhaMaker

job_store = JobStore(MemoryStore(), additional_stores={"data": MemoryStore()})
si_structure = Structure.from_file(TEST_DIR / "structures" / "Si_diamond.cif")
si_structure=si_structure.to_conventional()
from mp_api.client import MPRester
mpr = MPRester(api_key='Z4aKTAgeEudmS0bMPkKVS3EtOnej1zah')


si_structure = mpr.get_structure_by_material_id("mp-149", conventional_unit_cell=True)


# Then one can use the `QhaMaker` to generate a `Flow`. First, the structure will be optimized than the structures will be optimized at constant volume along an energy volume curve. Please make sure the structural optimizations are tight enough. At each of these volumes, a phonon run will then be performed. The quasi-harmonic approximation is only valid if the harmonic phonon curves don't show any imaginary modes. However, for testing, you can also switch off this option.

# Before we start the quasi-harmonic workflow, we adapt the first relaxation, the relaxation with different volumes and the static runs for the phonon calculation. As we deal with Si, we will not add the non-analytical term correction.

# In[3]:


from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.core import TightRelaxMaker
from atomate2.vasp.sets.core import StaticSetGenerator, TightRelaxSetGenerator
from atomate2.vasp.flows.phonons import PhononMaker
from atomate2.vasp.jobs.phonons import PhononDisplacementMaker
phonon_bulk_relax_maker_isif3 = DoubleRelaxMaker.from_relax_maker(
    TightRelaxMaker(
        run_vasp_kwargs={"handlers": ()},
        input_set_generator=TightRelaxSetGenerator(
            user_incar_settings={
                "GGA": "PE",
                "ISPIN": 1,
                "KSPACING": 0.1,
                # "EDIFFG": 1e-5,
                "ALGO": "Normal",
                "LAECHG": False,
                "ISMEAR": 0,
                "ENCUT": 700,
                "IBRION": 1,
                "ISYM": 0,
                "SIGMA": 0.05,
                "LCHARG": False,  # Do not write the CHGCAR file
                "LWAVE": False,  # Do not write the WAVECAR file
                "LVTOT": False,  # Do not write LOCPOT file
                "LORBIT": None,  # No output of projected or partial DOS in EIGENVAL, PROCAR and DOSCAR
                "LOPTICS": False,  # No PCDAT file
                "LREAL": False,
                "ISIF": 3,
                # to be removed
                "NPAR": 4,
            }
        ),
    )
)

phonon_displacement_maker = PhononDisplacementMaker(
    run_vasp_kwargs={"handlers": ()}, input_set_generator=StaticSetGenerator(
        user_incar_settings={
            "GGA": "PE",
            "IBRION": -1,
            "ISPIN": 1,
            "ISMEAR": 0,
            "ISIF": 3,
            "ENCUT": 700,
            "EDIFF": 1e-7,
            "LAECHG": False,
            "LREAL": False,
            "ALGO": "Normal",
            "NSW": 0,
            "LCHARG": False,  # Do not write the CHGCAR file
            "LWAVE": False,  # Do not write the WAVECAR file
            "LVTOT": False,  # Do not write LOCPOT file
            "LORBIT": None,  # No output of projected or partial DOS in EIGENVAL, PROCAR and DOSCAR
            "LOPTICS": False,  # No PCDAT file
            "SIGMA": 0.05,
            "ISYM": 0,
            "KSPACING": 0.1,
            "NPAR": 4,
        },
        auto_ispin=False,
    )
)



phonon_bulk_relax_maker_isif4 = DoubleRelaxMaker.from_relax_maker(
    TightRelaxMaker(
        run_vasp_kwargs={"handlers": ()},
        input_set_generator=TightRelaxSetGenerator(
            user_incar_settings={
                "GGA": "PE",
                "ISPIN": 1,
                "KSPACING": 0.1,
                "ALGO": "Normal",
                "LAECHG": False,
                "ISMEAR": 0,
                "ENCUT": 700,
                "IBRION": 1,
                "ISYM": 0,
                "SIGMA": 0.05,
                "LCHARG": False,  # Do not write the CHGCAR file
                "LWAVE": False,  # Do not write the WAVECAR file
                "LVTOT": False,  # Do not write LOCPOT file
                "LORBIT": None,  # No output of projected or partial DOS in EIGENVAL, PROCAR and DOSCAR
                "LOPTICS": False,  # No PCDAT file
                "LREAL": False,
                "ISIF": 4,
                # to be removed
                "NPAR": 4,
            }
        ),
    )
)

phonon_displacement_maker.name = "dft phonon static"



# In[4]:


flow = QhaMaker(
    initial_relax_maker=phonon_bulk_relax_maker_isif3,
    eos_relax_maker=phonon_bulk_relax_maker_isif4,
    min_length=10,
    phonon_maker=PhononMaker(generate_frequencies_eigenvectors_kwargs={"tmin": 0, "tmax": 1000, "tstep": 10},
                             bulk_relax_maker=None,
                             born_maker=None,
                             static_energy_maker=phonon_displacement_maker,
                             phonon_displacement_maker=phonon_displacement_maker),
    linear_strain=(-0.15, 0.15),
    number_of_frames=6,
    pressure=None,
    t_max=None,
    ignore_imaginary_modes=False,
    skip_analysis=False,
    eos_type="vinet"
).make(structure=si_structure)


# In[5]:


with mock_vasp(ref_paths=ref_paths) as mf:
    run_locally(
        flow,
        create_folders=True,
        ensure_success=True,
        raise_immediately=True,
        store=job_store,
    )


# In[ ]:




