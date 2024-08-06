from atomate2.jdftx.sets.base import JdftxInputSet, JdftxInputGenerator
from atomate2.jdftx.io.JDFTXInfile import JDFTXInfile
import os
import pathlib

p = pathlib.Path(__file__)
filepath = p.parents[1] / pathlib.Path("io/CO.in")
# jin = JDFTXInfile.from_file(filepath)
# jset = JdftxInputSet(jin)
# jset.write_input(p.parents[0], make_dir=True)

in_dict = {"coords-type": "Cartesian"}
jin = JDFTXInfile(in_dict)
# jin.write_file(p.parents[0] / pathlib.Path("inputs.in"))

in_dict = {
    'elec-ex-corr': 'gga', 
    'van-der-waals': 'D3', 
    'elec-cutoff': {'Ecut': 20, 'EcutRho': 100}, 
    'electronic-minimize': {'nIterations': 100, 'energyDiffThreshold': 1e-07}, 
    'elec-smearing': {'smearingType': 'Fermi', 'smearingWidth': 0.001}, 
    # 'elec-initial-magnetization': {'M': 0, 'constraint': False}, 
    'spintype': 'z-spin', 
    'core-overlap-check': 'none', 
    'converge-empty-states': True, # changed from 'yes'
    'band-projection-params': {'ortho': True, 'norm': False}, 
    'latt-move-scale': {'s0': 0, 's1': 0, 's2': 0}, 
    'lattice-minimize': {'nIterations': 0}, 
    'symmetries': 'none', 
    'ion-species': 'GBRV_v1.5/$ID_pbe_v1.uspp', 
    'dump': [{"freq": "End", "var": "Dtot"}, {"freq": "End", "var": "State"}],
}

# jin = JDFTXInfile.from_dict(in_dict)
# print(jin)
generator = JdftxInputGenerator()
jset = generator.get_input_set()
print(jset.jdftxinput)