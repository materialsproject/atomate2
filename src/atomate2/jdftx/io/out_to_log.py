"""
Use me on a JDFTx out file running an ionic/lattice minimization
(pairs well with monitoring a lattice optimization ran through https://github.com/benrich37/perlStuff/blob/master/opt.py)
"""


import sys
#file = sys.argv[1]
from jdftx.io.JDFTXOutfile import get_atoms_list_from_out, is_done
import numpy as np

logx_init_str = "\n Entering Link 1 \n \n"


def opt_spacer(i, nSteps):
    dump_str = "\n GradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGrad\n"
    dump_str += f"\n Step number   {i+1}\n"
    if i == nSteps:
        dump_str += " Optimization completed.\n"
        dump_str += "    -- Stationary point found.\n"
    dump_str += "\n GradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGrad\n"
    return dump_str

def scf_str(atoms, e_conv=(1/27.211397)):
    got_it = False
    try:
        E = atoms.get_potential_energy()
        got_it = True
    except:
        pass
    try:
        E = atoms.E
        got_it = True
    except:
        pass
    if not got_it:
        E = 0
    return f"\n SCF Done:  E =  {0*e_conv}\n\n"

def log_input_orientation(atoms, do_cell=False):
    dump_str = "                          Input orientation:                          \n"
    dump_str += " ---------------------------------------------------------------------\n"
    dump_str += " Center     Atomic      Atomic             Coordinates (Angstroms)\n"
    dump_str += " Number     Number       Type             X           Y           Z\n"
    dump_str += " ---------------------------------------------------------------------\n"
    at_ns = atoms.get_atomic_numbers()
    at_posns = atoms.positions
    nAtoms = len(at_ns)
    for i in range(nAtoms):
        dump_str += f" {i+1} {at_ns[i]} 0 "
        for j in range(3):
            dump_str += f"{at_posns[i][j]} "
        dump_str += "\n"
    if do_cell:
        cell = atoms.cell
        for i in range(3):
            dump_str += f"{i + nAtoms + 1} -2 0 "
            for j in range(3):
                dump_str += f"{cell[i][j]} "
            dump_str += "\n"
    dump_str += " ---------------------------------------------------------------------\n"
    return dump_str

def get_charges(atoms):
    es = []
    charges = None
    try:
        charges = atoms.get_charges()
    except Exception as e:
        es.append(e)
        pass
    if charges is None:
        try:
            charges = atoms.charges
        except Exception as e:
            es.append(e)
            pass
    if charges is None:
        try:
            charges = atoms.arrays["initial_charges"]
        except Exception as e:
            es.append(e)
            print(es)
            assert False
    return charges

def log_charges(atoms):
    try:
        charges = get_charges(atoms)
        nAtoms = len(atoms.positions)
        symbols = atoms.get_chemical_symbols()
    except:
        return " "
    dump_str = " **********************************************************************\n\n"
    dump_str += "            Population analysis using the SCF Density.\n\n"
    dump_str = " **********************************************************************\n\n Mulliken charges:\n    1\n"
    for i in range(nAtoms):
        dump_str += f"{int(i+1)} {symbols[i]} {charges[i]} \n"
    dump_str += f" Sum of Mulliken charges = {np.sum(charges)}\n"
    return dump_str

def get_do_cell(pbc):
    return np.sum(pbc) > 0


def get_start_line(outfname):
    start = 0
    for i, line in enumerate(open(outfname)):
        if "JDFTx 1." in line:
            start = i
    return start


logx_finish_str = " Normal termination of Gaussian 16"

def log_forces(atoms):
    dump_str = ""
    # dump_str += " Calling FoFJK, ICntrl=      2527 FMM=F ISym2X=1 I1Cent= 0 IOpClX= 0 NMat=1 NMatS=1 NMatT=0.\n"
    # dump_str += " ***** Axes restored to original set *****\n"
    dump_str += "-------------------------------------------------------------------\n"
    dump_str += " Center     Atomic                   Forces (Hartrees/Bohr)\n"
    dump_str += " Number     Number              X              Y              Z\n"
    dump_str += " -------------------------------------------------------------------\n"
    forces = []
    try:
        momenta = atoms.get_momenta()
    except Exception as e:
        print(e)
        momenta = np.zeros([len(atoms.get_atomic_numbers()), 3])
    for i, number in enumerate(atoms.get_atomic_numbers()):
        add_str = f" {i+1} {number}"
        force = momenta[i]
        forces.append(np.linalg.norm(force))
        for j in range(3):
            add_str += f"\t{force[j]:.9f}"
        add_str += "\n"
        dump_str += add_str
    dump_str += " -------------------------------------------------------------------\n"
    forces = np.array(forces)
    dump_str += f" Cartesian Forces:  Max {max(forces):.9f} RMS {np.std(forces):.9f}\n"
    return dump_str

def out_to_logx_str(outfile, use_force=False, e_conv=(1/27.211397)):
    atoms_list = get_atoms_list_from_out(outfile)
    dump_str = logx_init_str
    do_cell = get_do_cell(atoms_list[0].cell)
    if use_force:
        do_cell = False
    if use_force:
        for i in range(len(atoms_list)):
            dump_str += log_input_orientation(atoms_list[i], do_cell=do_cell)
            dump_str += f"\n SCF Done:  E =  {atoms_list[i].E*e_conv}\n\n"
            dump_str += log_charges(atoms_list[i])
            dump_str += log_forces(atoms_list[i])
            dump_str += opt_spacer(i, len(atoms_list))
    else:
        for i in range(len(atoms_list)):
            dump_str += log_input_orientation(atoms_list[i], do_cell=do_cell)
            dump_str += f"\n SCF Done:  E =  {atoms_list[i].E*e_conv}\n\n"
            dump_str += log_charges(atoms_list[i])
            dump_str += opt_spacer(i, len(atoms_list))
    if is_done(outfile):
        dump_str += log_input_orientation(atoms_list[-1])
        dump_str += logx_finish_str
    return dump_str

# assert "out" in file
# with open(file + ".logx", "w") as f:
#     f.write(out_to_logx_str(file))
#     f.close()
# # with open(file + "_wforce.logx", "w") as f:
# #     f.write(out_to_logx_str(file, use_force=True))
# #     f.close()