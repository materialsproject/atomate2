"""
Use me on a JDFTx out file running an ionic/lattice minimization
(pairs well with monitoring a lattice optimization ran through https://github.com/benrich37/perlStuff/blob/master/opt.py)
"""


import sys
#file = sys.argv[1]
import numpy as np
from ase.units import Bohr
from ase import Atoms, Atom

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


def get_atoms_from_outfile_data(names, posns, R, charges=None, E=0, momenta=None):
    atoms = Atoms()
    posns *= Bohr
    R = R.T*Bohr
    atoms.cell = R
    if charges is None:
        charges = np.zeros(len(names))
    if momenta is None:
        momenta = np.zeros([len(names), 3])
    for i in range(len(names)):
        atoms.append(Atom(names[i], posns[i], charge=charges[i], momentum=momenta[i]))
    atoms.E = E
    return atoms


def get_atoms_list_from_out_reset_vars(nAtoms=100, _def=100):
    R = np.zeros([3, 3])
    posns = []
    names = []
    chargeDir = {}
    active_lattice = False
    lat_row = 0
    active_posns = False
    log_vars = False
    coords = None
    new_posn = False
    active_lowdin = False
    idxMap = {}
    j = 0
    E = 0
    if nAtoms is None:
        nAtoms = _def
    charges = np.zeros(nAtoms, dtype=float)
    forces = []
    active_forces = False
    coords_forces = None
    return R, posns, names, chargeDir, active_posns, active_lowdin, active_lattice, posns, coords, idxMap, j, lat_row, \
        new_posn, log_vars, E, charges, forces, active_forces, coords_forces


def get_initial_lattice(outfile, start):
    start_key = "lattice  \\"
    active = False
    R = np.zeros([3, 3])
    lat_row = 0
    for i, line in enumerate(open(outfile)):
        if i > start:
            if active:
                if lat_row < 3:
                    R[lat_row, :] = [float(x) for x in line.split()[0:3]]
                    lat_row += 1
                else:
                    active = False
                    lat_row = 0
            elif start_key in line:
                active = True
    return R

def get_input_coord_vars_from_outfile(outfname):
    start_line = get_start_line(outfname)
    names = []
    posns = []
    R = np.zeros([3,3])
    lat_row = 0
    active_lattice = False
    with open(outfname) as f:
        for i, line in enumerate(f):
            if i > start_line:
                tokens = line.split()
                if len(tokens) > 0:
                    if tokens[0] == "ion":
                        names.append(tokens[1])
                        posns.append(np.array([float(tokens[2]), float(tokens[3]), float(tokens[4])]))
                    elif tokens[0] == "lattice":
                        active_lattice = True
                    elif active_lattice:
                        if lat_row < 3:
                            R[lat_row, :] = [float(x) for x in tokens[:3]]
                            lat_row += 1
                        else:
                            active_lattice = False
                    elif "Initializing the Grid" in line:
                        break
    if not len(names) > 0:
        raise ValueError("No ion names found")
    if len(names) != len(posns):
        raise ValueError("Unequal ion positions/names found")
    if np.sum(R) == 0:
        raise ValueError("No lattice matrix found")
    return names, posns, R

def get_start_lines(outfname, add_end=False):
    start_lines = []
    end_line = 0
    for i, line in enumerate(open(outfname)):
        if "JDFTx 1." in line:
            start_lines.append(i)
        end_line = i
    if add_end:
        start_lines.append(i)
    return start_lines


def get_atoms_list_from_out_slice(outfile, i_start, i_end):
    charge_key = "oxidation-state"
    opts = []
    nAtoms = None
    R, posns, names, chargeDir, active_posns, active_lowdin, active_lattice, posns, coords, idxMap, j, lat_row, \
        new_posn, log_vars, E, charges, forces, active_forces, coords_forces = get_atoms_list_from_out_reset_vars()
    for i, line in enumerate(open(outfile)):
        if i > i_start and i < i_end:
            if new_posn:
                if "Lowdin population analysis " in line:
                    active_lowdin = True
                elif "R =" in line:
                    active_lattice = True
                elif "# Forces in" in line:
                    active_forces = True
                    coords_forces = line.split()[3]
                elif line.find('# Ionic positions in') >= 0:
                    coords = line.split()[4]
                    active_posns = True
                elif active_lattice:
                    if lat_row < 3:
                        R[lat_row, :] = [float(x) for x in line.split()[1:-1]]
                        lat_row += 1
                    else:
                        active_lattice = False
                        lat_row = 0
                elif active_posns:
                    tokens = line.split()
                    if len(tokens) and tokens[0] == 'ion':
                        names.append(tokens[1])
                        posns.append(np.array([float(tokens[2]), float(tokens[3]), float(tokens[4])]))
                        if tokens[1] not in idxMap:
                            idxMap[tokens[1]] = []
                        idxMap[tokens[1]].append(j)
                        j += 1
                    else:
                        posns = np.array(posns)
                        active_posns = False
                        nAtoms = len(names)
                        if len(charges) < nAtoms:
                            charges = np.zeros(nAtoms)
                ##########
                elif active_forces:
                    tokens = line.split()
                    if len(tokens) and tokens[0] == 'force':
                        forces.append(np.array([float(tokens[2]), float(tokens[3]), float(tokens[4])]))
                    else:
                        forces = np.array(forces)
                        active_forces = False
                ##########
                elif "Minimize: Iter:" in line:
                    if "F: " in line:
                        E = float(line[line.index("F: "):].split(' ')[1])
                    elif "G: " in line:
                        E = float(line[line.index("G: "):].split(' ')[1])
                elif active_lowdin:
                    if charge_key in line:
                        look = line.rstrip('\n')[line.index(charge_key):].split(' ')
                        symbol = str(look[1])
                        line_charges = [float(val) for val in look[2:]]
                        chargeDir[symbol] = line_charges
                        for atom in list(chargeDir.keys()):
                            for k, idx in enumerate(idxMap[atom]):
                                charges[idx] += chargeDir[atom][k]
                    elif "#" not in line:
                        active_lowdin = False
                        log_vars = True
                elif log_vars:
                    if np.sum(R) == 0.0:
                        R = get_input_coord_vars_from_outfile(outfile)[2]
                    if coords != 'cartesian':
                        posns = np.dot(posns, R)
                    if len(forces) == 0:
                        forces = np.zeros([nAtoms, 3])
                    if coords_forces.lower() != 'cartesian':
                        forces = np.dot(forces, R)
                    opts.append(get_atoms_from_outfile_data(names, posns, R, charges=charges, E=E, momenta=forces))
                    R, posns, names, chargeDir, active_posns, active_lowdin, active_lattice, posns, coords, idxMap, j, lat_row, \
                        new_posn, log_vars, E, charges, forces, active_forces, coords_forces = get_atoms_list_from_out_reset_vars(
                        nAtoms=nAtoms)
            elif "Computing DFT-D3 correction:" in line:
                new_posn = True
    return opts

def get_atoms_list_from_out(outfile):
    start_lines = get_start_lines(outfile, add_end=True)
    atoms_list = []
    for i in range(len(start_lines) - 1):
        atoms_list += get_atoms_list_from_out_slice(outfile, start_lines[i], start_lines[i+1])
    return atoms_list


# def get_atoms_list_from_out(outfile):
#     start = get_start_line(outfile)
#     charge_key = "oxidation-state"
#     opts = []
#     nAtoms = None
#     R, posns, names, chargeDir, active_posns, active_lowdin, active_lattice, posns, coords, idxMap, j, lat_row, \
#         new_posn, log_vars, E, charges, forces, active_forces, coords_forces = get_atoms_list_from_out_reset_vars()
#     for i, line in enumerate(open(outfile)):
#         if i > start:
#             if new_posn:
#                 if "Lowdin population analysis " in line:
#                     active_lowdin = True
#                 elif "R =" in line:
#                     active_lattice = True
#                 elif "# Forces in" in line:
#                     active_forces = True
#                     coords_forces = line.split()[3]
#                 elif line.find('# Ionic positions in') >= 0:
#                     coords = line.split()[4]
#                     active_posns = True
#                 elif active_lattice:
#                     if lat_row < 3:
#                         R[lat_row, :] = [float(x) for x in line.split()[1:-1]]
#                         lat_row += 1
#                     else:
#                         active_lattice = False
#                         lat_row = 0
#                 elif active_posns:
#                     tokens = line.split()
#                     if len(tokens) and tokens[0] == 'ion':
#                         names.append(tokens[1])
#                         posns.append(np.array([float(tokens[2]), float(tokens[3]), float(tokens[4])]))
#                         if tokens[1] not in idxMap:
#                                 idxMap[tokens[1]] = []
#                         idxMap[tokens[1]].append(j)
#                         j += 1
#                     else:
#                         posns=np.array(posns)
#                         active_posns = False
#                         nAtoms = len(names)
#                         if len(charges) < nAtoms:
#                             charges=np.zeros(nAtoms)
#                 ##########
#                 elif active_forces:
#                     tokens = line.split()
#                     if len(tokens) and tokens[0] == 'force':
#                         forces.append(np.array([float(tokens[2]), float(tokens[3]), float(tokens[4])]))
#                     else:
#                         forces=np.array(forces)
#                         active_forces = False
#                 ##########
#                 elif "Minimize: Iter:" in line:
#                     if "F: " in line:
#                         E = float(line[line.index("F: "):].split(' ')[1])
#                     elif "G: " in line:
#                         E = float(line[line.index("G: "):].split(' ')[1])
#                 elif active_lowdin:
#                     if charge_key in line:
#                         look = line.rstrip('\n')[line.index(charge_key):].split(' ')
#                         symbol = str(look[1])
#                         line_charges = [float(val) for val in look[2:]]
#                         chargeDir[symbol] = line_charges
#                         for atom in list(chargeDir.keys()):
#                             for k, idx in enumerate(idxMap[atom]):
#                                 charges[idx] += chargeDir[atom][k]
#                     elif "#" not in line:
#                         active_lowdin = False
#                         log_vars = True
#                 elif log_vars:
#                     if np.sum(R) == 0.0:
#                         R = get_input_coord_vars_from_outfile(outfile)[2]
#                     if coords != 'cartesian':
#                         posns = np.dot(posns, R)
#                     if len(forces) == 0:
#                         forces = np.zeros([nAtoms, 3])
#                     if coords_forces.lower() != 'cartesian':
#                         forces = np.dot(forces, R)
#                     opts.append(get_atoms_from_outfile_data(names, posns, R, charges=charges, E=E, momenta=forces))
#                     R, posns, names, chargeDir, active_posns, active_lowdin, active_lattice, posns, coords, idxMap, j, lat_row, \
#                         new_posn, log_vars, E, charges, forces, active_forces, coords_forces = get_atoms_list_from_out_reset_vars(nAtoms=nAtoms)
#             elif "Computing DFT-D3 correction:" in line:
#                 new_posn = True
#     return opts

def is_done(outfile):
    start_line = get_start_line(outfile)
    done = False
    with open(outfile, "r") as f:
        for i, line in enumerate(f):
            if i > start_line:
                if "Minimize: Iter:" in line:
                    done = False
                elif "Minimize: Converged" in line:
                    done = True
    return done

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