import json
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read

from atomate2.forcefields.flows.phonons import PhononMaker
from atomate2.forcefields.jobs import GAPRelaxMaker
from atomate2.forcefields.jobs import GAPStaticMaker
from atomate2.forcefields.jobs import CHGNetRelaxMaker, CHGNetStaticMaker
from pymatgen.core import Structure
from jobflow.managers.fireworks import flow_to_workflow
from fireworks import LaunchPad
from jobflow import run_locally
import os
from jobflow import SETTINGS

from mp_api.client import MPRester

mpr = MPRester(api_key='ajGziP3VMy57gFn9yzlJSuWQMNdjDa8q')

st = mpr.get_structure_by_material_id('mp-10635')

phonons = PhononMaker(generate_frequencies_eigenvectors_kwargs={"npoints_band": 50}, bulk_relax_maker=None).make(structure=st)


resp = run_locally(phonons, create_folders=False, allow_external_references=True, store=SETTINGS.JOB_STORE)