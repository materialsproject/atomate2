from copy import deepcopy
from os import write

from .generic_tags import BoolTag, StrTag, IntTag, FloatTag, TagContainer, MultiformatTag, BoolTagContainer, DumpTagContainer, InitMagMomTag

JDFTXDumpFreqOptions = [
    "Electronic", "End", "Fluid", "Gummel", "Init", "Ionic"
]
JDFTXDumpVarOptions = [
    "BandEigs",  # Band Eigenvalues
    "BandProjections",  # Projections of each band state against each atomic orbital
    "BandUnfold",  # Unfold band structure from supercell to unit cell (see command band-unfold)
    "Berry",  # Berry curvature i <dC/dk| X |dC/dk>, only allowed at End (see command Cprime-params)
    "BGW",  # G-space wavefunctions, density and potential for Berkeley GW (requires HDF5 support)
    "BoundCharge",  # Bound charge in the fluid
    "BulkEpsilon",  # Dielectric constant of a periodic solid (see command bulk-epsilon)
    "ChargedDefect",  # Calculate energy correction for charged defect (see command charged-defect)
    "CoreDensity",  # Total core electron density (from partial core corrections)
    "Dfluid",  # Electrostatic potential due to fluid alone
    "Dipole",  # Dipole moment of explicit charges (ionic and electronic)
    "Dn",  # First order change in electronic density
    "DOS",  # Density of States (see command density-of-states)
    "Dtot",  # Total electrostatic potential
    "Dvac",  # Electrostatic potential due to explicit system alone
    "DVext",  # External perturbation
    "DVscloc",  # First order change in local self-consistent potential
    "DWfns",  # Perturbation Wavefunctions
    "Ecomponents",  # Components of the energy
    "EigStats",  # Band eigenvalue statistics: HOMO, LUMO, min, max and Fermi level
    "ElecDensity",  # Electronic densities (n or nup,ndn)
    "ElecDensityAccum",  # Electronic densities (n or nup,ndn) accumulated over MD trajectory
    "EresolvedDensity",  # Electron density from bands within specified energy ranges
    "ExcCompare",  # Energies for other exchange-correlation functionals (see command elec-ex-corr-compare)
    "Excitations",  # Dumps dipole moments and transition strength (electric-dipole) of excitations
    "FCI",  # Output Coulomb matrix elements in FCIDUMP format
    "FermiDensity",  # Electron density from fermi-derivative at specified energy
    "FermiVelocity",  # Fermi velocity, density of states at Fermi level and related quantities
    "Fillings",  # Fillings
    "FluidDebug",  # Fluid specific debug output if any
    "FluidDensity",  # Fluid densities (NO,NH,nWater for explicit fluids, cavity function for PCMs)
    "Forces",  # Forces on the ions in the coordinate system selected by command forces-output-coords
    "Gvectors",  # List of G vectors in reciprocal lattice basis, for each k-point
    "IonicDensity",  # Nuclear charge density (with gaussians)
    "IonicPositions",  # Ionic positions in the same format (and coordinate system) as the input file
    "KEdensity",  # Kinetic energy density of the valence electrons
    "Kpoints",  # List of reduced k-points in calculation, and mapping to the unreduced k-point mesh
    "L",  # Angular momentum matrix elements, only allowed at End (see command Cprime-params)
    "Lattice",  # Lattice vectors in the same format as the input file
    "Momenta",  # Momentum matrix elements in a binary file (indices outer to inner: state, cartesian direction, band1, band2)
    "None",  # Dump nothing
    "Ocean",  # Wave functions for Ocean code
    "OrbitalDep",  # Custom output from orbital-dependent functionals (eg. quasi-particle energies, discontinuity potential)
    "Q",  # Quadrupole r*p matrix elements, only allowed at End (see command Cprime-params)
    "QMC",  # Blip'd orbitals and potential for CASINO [27]
    "R",  # Position operator matrix elements, only allowed at End (see command Cprime-params)
    "RealSpaceWfns",  # Real-space wavefunctions (one column per file)
    "RhoAtom",  # Atomic-orbital projected density matrices (only for species with +U enabled)
    "SelfInteractionCorrection",  # Calculates Perdew-Zunger self-interaction corrected Kohn-Sham eigenvalues
    "SlabEpsilon",  # Local dielectric function of a slab (see command slab-epsilon)
    "SolvationRadii",  # Effective solvation radii based on fluid bound charge distribution
    "Spin",  # Spin matrix elements from non-collinear calculations in a binary file (indices outer to inner: state, cartesian direction, band1, band2)
    "State",  # All variables needed to restart calculation: wavefunction and fluid state/fillings if any
    "Stress",  # Dumps dE/dR_ij where R_ij is the i'th component of the j'th lattice vector
    "Symmetries",  # List of symmetry matrices (in covariant lattice coordinates)
    "Vcavity",  # Fluid cavitation potential on the electron density that determines the cavity
    "Velocities",  # Diagonal momentum/velocity matrix elements in a binary file (indices outer to inner: state, band, cartesian direction)
    "VfluidTot",  # Total contribution of fluid to the electron potential
    "Vlocps",  # Local part of pseudopotentials
    "Vscloc",  # Self-consistent potential
    "XCanalysis"  # Debug VW KE density, single-particle-ness and spin-polarzied Hartree potential
    ]


#simple dictionaries deepcopied multiple times into MASTER_TAG_LIST later for different tags
JDFTXMinimize_subtagdict = {
    'alphaTincreaseFactor': FloatTag(),
    'alphaTmin': FloatTag(),
    'alphaTreduceFactor': FloatTag(),
    'alphaTstart': FloatTag(),
    'dirUpdateScheme': StrTag(options = ['FletcherReeves', 'HestenesStiefel', 'L-BFGS', 'PolakRibiere', 'SteepestDescent']),
    'energyDiffThreshold': FloatTag(),
    'fdTest': BoolTag(),
    'history': IntTag(),
    'knormThreshold': FloatTag(),
    'linminMethod': StrTag(options = ['CubicWolfe', 'DirUpdateRecommended', 'Quad', 'Relax']),
    'nAlphaAdjustMax': FloatTag(),
    'nEnergyDiff': IntTag(),
    'nIterations': IntTag(),
    'updateTestStepSize': BoolTag(),
    'wolfeEnergy': FloatTag(),
    'wolfeGradient': FloatTag(),
    }
JDFTXFluid_subtagdict = {
    'epsBulk': FloatTag(),
    'epsInf': FloatTag(),
    'epsLJ': FloatTag(),
    'Nnorm': FloatTag(),
    'pMol': FloatTag(),
    'poleEl': TagContainer(
        can_repeat = True,
        subtags = {
            "omega0": FloatTag(write_tagname=False, optional=False),
            "gamma0": FloatTag(write_tagname=False, optional=False),
            "A0": FloatTag(write_tagname=False, optional=False),
        },
    ),
    # 'poleEl': FloatTag(can_repeat = True),
    'Pvap': FloatTag(),
    'quad_nAlpha': FloatTag(),
    'quad_nBeta': FloatTag(),
    'quad_nGamma': FloatTag(),
    'representation': TagContainer(subtags = {'MuEps': FloatTag(), 'Pomega': FloatTag(), 'PsiAlpha': FloatTag()}),
    'Res': FloatTag(),
    'Rvdw': FloatTag(),
    's2quadType': StrTag(options = ['10design60', '11design70', '12design84', '13design94',
                                    '14design108', '15design120', '16design144', '17design156',
                                    '18design180', '19design204', '20design216', '21design240',
                                    '7design24', '8design36', '9design48', 'Euler',
                                    'Icosahedron', 'Octahedron', 'Tetrahedron']),
    'sigmaBulk': FloatTag(),
    'tauNuc': FloatTag(),
    'translation': StrTag(options = ['ConstantSpline', 'Fourier', 'LinearSpline']),
    }

MASTER_TAG_LIST = {
    "extrafiles": {
        "include": StrTag(can_repeat=True),
    },
    "structure": {
        "latt-scale": TagContainer(
            allow_list_representation=True,
            subtags={
                "s0": IntTag(write_tagname=False, optional=False),
                "s1": IntTag(write_tagname=False, optional=False),
                "s2": IntTag(write_tagname=False, optional=False),
            },
        ),
        "latt-move-scale": TagContainer(
            allow_list_representation=True,
            subtags={
                "s0": FloatTag(write_tagname=False, optional=False),
                "s1": FloatTag(write_tagname=False, optional=False),
                "s2": FloatTag(write_tagname=False, optional=False),
            },
        ),
        "coords-type": StrTag(options=["Cartesian", "Lattice"]),
        # TODO: change lattice tag into MultiformatTag for different symmetry options
        "lattice": TagContainer(
            linebreak_Nth_entry=3,
            optional=False,
            allow_list_representation=True,
            subtags={
                "R00": FloatTag(write_tagname=False, optional=False, prec=12),
                "R01": FloatTag(write_tagname=False, optional=False, prec=12),
                "R02": FloatTag(write_tagname=False, optional=False, prec=12),
                "R10": FloatTag(write_tagname=False, optional=False, prec=12),
                "R11": FloatTag(write_tagname=False, optional=False, prec=12),
                "R12": FloatTag(write_tagname=False, optional=False, prec=12),
                "R20": FloatTag(write_tagname=False, optional=False, prec=12),
                "R21": FloatTag(write_tagname=False, optional=False, prec=12),
                "R22": FloatTag(write_tagname=False, optional=False, prec=12),
            },
        ),
        "ion": TagContainer(
            can_repeat=True,
            optional=False,
            allow_list_representation=True,
            subtags={
                "species-id": StrTag(write_tagname=False, optional=False),
                "x0": FloatTag(write_tagname=False, optional=False, prec=12),
                "x1": FloatTag(write_tagname=False, optional=False, prec=12),
                "x2": FloatTag(write_tagname=False, optional=False, prec=12),
                "v": TagContainer(
                    allow_list_representation=True,
                    subtags={
                        "vx0": FloatTag(write_tagname=False, optional=False, prec=12),
                        "vx1": FloatTag(write_tagname=False, optional=False, prec=12),
                        "vx2": FloatTag(write_tagname=False, optional=False, prec=12),
                    },
                ),
                "moveScale": IntTag(write_tagname=False, optional=False),
            },
        ),
        "perturb-ion": TagContainer(
            subtags={
                "species": StrTag(write_tagname=False, optional=False),
                "atom": IntTag(write_tagname=False, optional=False),
                "dx0": FloatTag(write_tagname=False, optional=False),
                "dx1": FloatTag(write_tagname=False, optional=False),
                "dx2": FloatTag(write_tagname=False, optional=False),
            }
        ),
        "core-overlap-check": StrTag(options=["additive", "vector", "none"]),
        "ion-species": StrTag(can_repeat=True, optional=False),
        "cache-projectors": BoolTag(),
        "ion-width": MultiformatTag(
            format_options=[
                StrTag(options=["Ecut", "fftbox"]),
                FloatTag(),
            ]
        ),
    },
    "symmetries": {
        "symmetries": StrTag(options=["automatic", "manual", "none"]),
        "symmetry-threshold": FloatTag(),
        "symmetry-matrix": TagContainer(
            linebreak_Nth_entry=3,
            can_repeat=True,
            allow_list_representation=True,
            subtags={
                "s00": IntTag(write_tagname=False, optional=False),
                "s01": IntTag(write_tagname=False, optional=False),
                "s02": IntTag(write_tagname=False, optional=False),
                "s10": IntTag(write_tagname=False, optional=False),
                "s11": IntTag(write_tagname=False, optional=False),
                "s12": IntTag(write_tagname=False, optional=False),
                "s20": IntTag(write_tagname=False, optional=False),
                "s21": IntTag(write_tagname=False, optional=False),
                "s22": IntTag(write_tagname=False, optional=False),
                "a0": FloatTag(write_tagname=False, optional=False, prec=12),
                "a1": FloatTag(write_tagname=False, optional=False, prec=12),
                "a2": FloatTag(write_tagname=False, optional=False, prec=12),
            },
        ),
    },
    "k-mesh": {
        "kpoint": TagContainer(
            can_repeat=True,
            allow_list_representation=True,
            subtags={
                "k0": FloatTag(write_tagname=False, optional=False, prec=12),
                "k1": FloatTag(write_tagname=False, optional=False, prec=12),
                "k2": FloatTag(write_tagname=False, optional=False, prec=12),
                "weight": FloatTag(write_tagname=False, optional=False, prec=12),
            },
        ),
        "kpoint-folding": TagContainer(
            allow_list_representation=True,
            subtags={
                "n0": IntTag(write_tagname=False, optional=False),
                "n1": IntTag(write_tagname=False, optional=False),
                "n2": IntTag(write_tagname=False, optional=False),
            },
        ),
        "kpoint-reduce-inversion": BoolTag(),
    },
    "electronic": {
        "elec-ex-corr": MultiformatTag(
            format_options=[
                # note that hyb-HSE06 has a bug in JDFTx and should not be used and is excluded here
                #    use the LibXC version instead (hyb-gga-HSE06)
                StrTag(
                    write_tagname=True,  # Ben: Changing this to True with permission from Jacob!
                    options=[
                        "gga",
                        "gga-PBE",
                        "gga-PBEsol",
                        "gga-PW91",
                        "Hartree-Fock",
                        "hyb-PBE0",
                        "lda",
                        "lda-PW",
                        "lda-PW-prec",
                        "lda-PZ",
                        "lda-Teter",
                        "lda-VWN",
                        "mgga-revTPSS",
                        "mgga-TPSS",
                        "orb-GLLBsc",
                        "pot-LB94",
                    ],
                ),
                # TODO: add all X and C options from here: https://jdftx.org/CommandElecExCorr.html
                #    note: use a separate variable elsewhere for this to not dominate this dictionary
                TagContainer(
                    subtags={
                        "funcX": StrTag(write_tagname=False, optional=False),
                        "funcC": StrTag(write_tagname=False, optional=False),
                    }
                ),
                # TODO: add all XC options from here: https://jdftx.org/CommandElecExCorr.html
                #    note: use a separate variable elsewhere for this to not dominate this dictionary
                TagContainer(
                    subtags={"funcXC": StrTag(write_tagname=False, optional=False)}
                ),
            ]
        ),
        "elec-ex-corr-compare": MultiformatTag(
            can_repeat=True,
            format_options=[
                # note that hyb-HSE06 has a bug in JDFTx and should not be used and is excluded here
                #    use the LibXC version instead (hyb-gga-HSE06)
                StrTag(
                    write_tagname=True,  # Ben: Changing this to True with permission from Jacob!
                    options=[
                        "gga",
                        "gga-PBE",
                        "gga-PBEsol",
                        "gga-PW91",
                        "Hartree-Fock",
                        "hyb-PBE0",
                        "lda",
                        "lda-PW",
                        "lda-PW-prec",
                        "lda-PZ",
                        "lda-Teter",
                        "lda-VWN",
                        "mgga-revTPSS",
                        "mgga-TPSS",
                        "orb-GLLBsc",
                        "pot-LB94",
                    ],
                ),
                # TODO: add all X and C options from here: https://jdftx.org/CommandElecExCorr.html
                #    note: use a separate variable elsewhere for this to not dominate this dictionary
                TagContainer(
                    subtags={
                        "funcX": StrTag(write_tagname=False, optional=False),
                        "funcC": StrTag(write_tagname=False, optional=False),
                    }
                ),
                # TODO: add all XC options from here: https://jdftx.org/CommandElecExCorr.html
                #    note: use a separate variable elsewhere for this to not dominate this dictionary
                TagContainer(
                    subtags={"funcXC": StrTag(write_tagname=False, optional=False)}
                ),
            ],
        ),
        "exchange-block-size": IntTag(),
        "exchange-outer-loop": IntTag(),
        "exchange-parameters": TagContainer(
            subtags={
                "exxScale": FloatTag(write_tagname=False, optional=False),
                "exxOmega": FloatTag(write_tagname=False),
            }
        ),
        "exchange-params": TagContainer(
            multiline_tag=True,
            subtags={
                "blockSize": IntTag(),
                "nOuterVxx": IntTag(),
            },
        ),
        "exchange-regularization": StrTag(
            options=[
                "AuxiliaryFunction",
                "None",
                "ProbeChargeEwald",
                "SphericalTruncated",
                "WignerSeitzTruncated",
            ]
        ),
        "tau-core": TagContainer(
            subtags={
                "species-id": StrTag(write_tagname=False, optional=False),
                "rCut": FloatTag(write_tagname=False),
                "plot": BoolTag(write_tagname=False),
            }
        ),
        "lj-override": FloatTag(),
        "van-der-waals": MultiformatTag(
            format_options=[
                StrTag(options=["D3"]),
                FloatTag(),
            ]
        ),
        "elec-cutoff": TagContainer(
            allow_list_representation=True,
            subtags={
                "Ecut": FloatTag(write_tagname=False, optional=False),
                "EcutRho": FloatTag(write_tagname=False),
            },
        ),
        "elec-smearing": TagContainer(
            allow_list_representation=True,
            subtags={
                "smearingType": StrTag(
                    options=["Cold", "Fermi", "Gauss", "MP1"],
                    write_tagname=False,
                    optional=False,
                ),
                "smearingWidth": FloatTag(write_tagname=False, optional=False),
            },
        ),
        "elec-n-bands": IntTag(),
        "spintype": StrTag(options=["no-spin", "spin-orbit", "vector-spin", "z-spin"]),
        'initial-magnetic-moments': InitMagMomTag(),
        "elec-initial-magnetization": TagContainer(
            subtags={
                "M": FloatTag(write_tagname=False, optional=False),
                "constrain": BoolTag(write_tagname=False, optional=False),
            }
        ),
        "target-Bz": FloatTag(),
        "elec-initial-charge": FloatTag(),
        "converge-empty-states": BoolTag(),
        "band-unfold": TagContainer(
            linebreak_Nth_entry=3,
            allow_list_representation=True,
            subtags={
                "M00": IntTag(write_tagname=False, optional=False),
                "M01": IntTag(write_tagname=False, optional=False),
                "M02": IntTag(write_tagname=False, optional=False),
                "M10": IntTag(write_tagname=False, optional=False),
                "M11": IntTag(write_tagname=False, optional=False),
                "M12": IntTag(write_tagname=False, optional=False),
                "M20": IntTag(write_tagname=False, optional=False),
                "M21": IntTag(write_tagname=False, optional=False),
                "M22": IntTag(write_tagname=False, optional=False),
            },
        ),
        "basis": StrTag(options=["kpoint-dependent", "single"]),
        "fftbox": TagContainer(
            allow_list_representation=True,
            subtags={
                "S0": IntTag(write_tagname=False, optional=False),
                "S1": IntTag(write_tagname=False, optional=False),
                "S2": IntTag(write_tagname=False, optional=False),
            },
        ),
        "electric-field": TagContainer(
            allow_list_representation=True,
            subtags={
                "Ex": IntTag(write_tagname=False, optional=False),
                "Ey": IntTag(write_tagname=False, optional=False),
                "Ez": IntTag(write_tagname=False, optional=False),
            },
        ),
        "perturb-electric-field": TagContainer(
            allow_list_representation=True,
            subtags={
                "Ex": IntTag(write_tagname=False, optional=False),
                "Ey": IntTag(write_tagname=False, optional=False),
                "Ez": IntTag(write_tagname=False, optional=False),
            },
        ),
        "box-potential": TagContainer(
            can_repeat=True,
            subtags={
                "xmin": FloatTag(write_tagname=False, optional=False),
                "xmax": FloatTag(write_tagname=False, optional=False),
                "ymin": FloatTag(write_tagname=False, optional=False),
                "ymax": FloatTag(write_tagname=False, optional=False),
                "zmin": FloatTag(write_tagname=False, optional=False),
                "zmax": FloatTag(write_tagname=False, optional=False),
                "Vin": FloatTag(write_tagname=False, optional=False),
                "Vout": FloatTag(write_tagname=False, optional=False),
                "convolve_radius": FloatTag(write_tagname=False),
            },
        ),
        "ionic-gaussian-potential": TagContainer(
            can_repeat=True,
            subtags={
                "species": StrTag(write_tagname=False, optional=False),
                "U0": FloatTag(write_tagname=False, optional=False),
                "sigma": FloatTag(write_tagname=False, optional=False),
                "geometry": StrTag(
                    options=["Spherical", "Cylindrical", "Planar"],
                    write_tagname=False,
                    optional=False,
                ),
            },
        ),
        "bulk-epsilon": TagContainer(
            subtags={
                "DtotFile": StrTag(write_tagname=False, optional=False),
                "Ex": FloatTag(write_tagname=False),
                "Ey": FloatTag(write_tagname=False),
                "Ez": FloatTag(write_tagname=False),
            }
        ),
        "charged-defect": TagContainer(
            can_repeat=True,
            subtags={
                "x0": FloatTag(write_tagname=False, optional=False),
                "x1": FloatTag(write_tagname=False, optional=False),
                "x2": FloatTag(write_tagname=False, optional=False),
                "q": FloatTag(write_tagname=False, optional=False),
                "sigma": FloatTag(write_tagname=False, optional=False),
            },
        ),
        "charged-defect-correction": TagContainer(
            subtags={
                "Slab": TagContainer(
                    subtags={
                        "dir": StrTag(
                            options=["100", "010", "001"], write_tagname=False
                        ),
                    }
                ),
                "DtotFile": StrTag(write_tagname=False, optional=False),
                "Eps": MultiformatTag(
                    format_options=[
                        FloatTag(write_tagname=False, optional=False),
                        StrTag(write_tagname=False, optional=False),
                    ]
                ),
                "rMin": FloatTag(write_tagname=False, optional=False),
                "rSigma": FloatTag(write_tagname=False, optional=False),
            }
        ),
        "Cprime-params": TagContainer(
            subtags={
                "dk": FloatTag(write_tagname=False),
                "degeneracyThreshold": FloatTag(write_tagname=False),
                "vThreshold": FloatTag(write_tagname=False),
                "realSpaceTruncated": BoolTag(write_tagname=False),
            }
        ),
        "electron-scattering": TagContainer(
            multiline_tag=True,
            subtags={
                "eta": FloatTag(optional=False),
                "Ecut": FloatTag(),
                "fCut": FloatTag(),
                "omegaMax": FloatTag(),
                "RPA": BoolTag(),
                "dumpEpsilon": BoolTag(),
                "slabResponse": BoolTag(),
                "EcutTransverse": FloatTag(),
                "computeRange": TagContainer(
                    subtags={
                        "iqStart": FloatTag(write_tagname=False, optional=False),
                        "iqStop": FloatTag(write_tagname=False, optional=False),
                    }
                ),
            },
        ),
        "perturb-test": BoolTag(write_value=False),
        "perturb-wavevector": TagContainer(
            subtags={
                "q0": FloatTag(write_tagname=False, optional=False),
                "q1": FloatTag(write_tagname=False, optional=False),
                "q2": FloatTag(write_tagname=False, optional=False),
            }
        ),
    },
    "truncation": {
        "coulomb-interaction": MultiformatTag(
            format_options=[
                # note that the first 2 and last 2 TagContainers could be combined, but keep separate so there is less ambiguity on formatting
                TagContainer(
                    subtags={
                        "truncationType": StrTag(
                            options=["Periodic", "Isolated"],
                            write_tagname=False,
                            optional=False,
                        )
                    }
                ),
                TagContainer(
                    subtags={
                        "truncationType": StrTag(
                            options=["Spherical"], write_tagname=False, optional=False
                        ),
                        "Rc": FloatTag(write_tagname=False),
                    }
                ),
                TagContainer(
                    subtags={
                        "truncationType": StrTag(
                            options=["Slab", "Wire"],
                            write_tagname=False,
                            optional=False,
                        ),
                        "dir": StrTag(
                            options=["001", "010", "100"],
                            write_tagname=False,
                            optional=False,
                        ),
                    }
                ),
                TagContainer(
                    subtags={
                        "truncationType": StrTag(
                            options=["Cylindrical"], write_tagname=False, optional=False
                        ),
                        "dir": StrTag(
                            options=["001", "010", "100"],
                            write_tagname=False,
                            optional=False,
                        ),
                        "Rc": FloatTag(write_tagname=False),
                    }
                ),
            ]
        ),
        "coulomb-truncation-embed": TagContainer(
            subtags={
                "c0": FloatTag(write_tagname=False, optional=False),
                "c1": FloatTag(write_tagname=False, optional=False),
                "c2": FloatTag(write_tagname=False, optional=False),
            }
        ),
        "coulomb-truncation-ion-margin": FloatTag(),
    },
    "restart": {
        "initial-state": StrTag(),
        "elec-initial-eigenvals": StrTag(),
        "elec-initial-fillings": TagContainer(
            subtags={
                "read": BoolTag(write_value=False, optional=False),
                "filename": StrTag(write_tagname=False, optional=False),
                "nBandsOld": IntTag(write_tagname=False),
            }
        ),
        "wavefunction": MultiformatTag(
            format_options=[
                TagContainer(
                    subtags={"lcao": BoolTag(write_value=False, optional=False)}
                ),
                TagContainer(
                    subtags={"random": BoolTag(write_value=False, optional=False)}
                ),
                TagContainer(
                    subtags={
                        "read": StrTag(write_value=False, optional=False),
                        "nBandsOld": IntTag(write_tagname=False),
                        "EcutOld": FloatTag(write_tagname=False),
                    }
                ),
                TagContainer(
                    subtags={
                        "read-rs": StrTag(write_value=False, optional=False),
                        "nBandsOld": IntTag(write_tagname=False),
                        "NxOld": IntTag(write_tagname=False),
                        "NyOld": IntTag(write_tagname=False),
                        "NzOld": IntTag(write_tagname=False),
                    }
                ),
            ]
        ),
        "fluid-initial-state": StrTag(),
        "perturb-incommensurate-wavefunctions": TagContainer(
            subtags={
                "filename": StrTag(write_tagname=False, optional=False),
                "EcutOld": IntTag(write_tagname=False),
            }
        ),
        "perturb-rhoExternal": StrTag(),
        "perturb-Vexternal": StrTag(),
        "fix-electron-density": StrTag(),
        "fix-electron-potential": StrTag(),
        "Vexternal": MultiformatTag(
            format_options=[
                TagContainer(
                    subtags={"filename": StrTag(write_value=False, optional=False)}
                ),
                TagContainer(
                    subtags={
                        "filenameUp": StrTag(write_value=False, optional=False),
                        "filenameDn": StrTag(write_tagname=False, optional=False),
                    }
                ),
            ]
        ),
        "rhoExternal": TagContainer(
            subtags={
                "filename": StrTag(write_tagname=False, optional=False),
                "includeSelfEnergy": FloatTag(write_tagname=False),
            }
        ),
        "slab-epsilon": TagContainer(
            subtags={
                "DtotFile": StrTag(write_tagname=False, optional=False),
                "sigma": FloatTag(write_tagname=False, optional=False),
                "Ex": FloatTag(write_tagname=False),
                "Ey": FloatTag(write_tagname=False),
                "Ez": FloatTag(write_tagname=False),
            }
        ),
    },
    "minimization": {
        "lcao-params": TagContainer(
            subtags={
                "nIter": IntTag(write_tagname=False),
                "Ediff": FloatTag(write_tagname=False),
                "smearingWidth": FloatTag(write_tagname=False),
            }
        ),
        "elec-eigen-algo": StrTag(options=["CG", "Davidson"]),
        "ionic-minimize": TagContainer(
            multiline_tag=True,
            subtags={
                **deepcopy(JDFTXMinimize_subtagdict),
            },
        ),
        "lattice-minimize": TagContainer(
            multiline_tag=True,
            subtags={
                **deepcopy(JDFTXMinimize_subtagdict),
            },
        ),
        "electronic-minimize": TagContainer(
            multiline_tag=True,
            subtags={
                **deepcopy(JDFTXMinimize_subtagdict),
            },
        ),
        "electronic-scf": TagContainer(
            multiline_tag=True,
            subtags={
                "energyDiffThreshold": FloatTag(),
                "history": IntTag(),
                "mixFraction": FloatTag(),
                "nIterations": IntTag(),
                "qMetric": FloatTag(),
                "residualThreshold": FloatTag(),
                "eigDiffThreshold": FloatTag(),
                "mixedVariable": StrTag(),
                "mixFractionMag": FloatTag(),
                "nEigSteps": IntTag(),
                "qKappa": FloatTag(),
                "qKerker": FloatTag(),
                "verbose": BoolTag(),
            },
        ),
        "fluid-minimize": TagContainer(
            multiline_tag=True,
            subtags={
                **deepcopy(JDFTXMinimize_subtagdict),
            },
        ),
        "davidson-band-ratio": FloatTag(),
        "wavefunction-drag": BoolTag(),
        "subspace-rotation-factor": TagContainer(
            subtags={
                "factor": FloatTag(write_tagname=False, optional=False),
                "adjust": BoolTag(write_tagname=False, optional=False),
            }
        ),
        "perturb-minimize": TagContainer(
            multiline_tag=True,
            subtags={
                "algorithm": StrTag(options=["MINRES", "CGIMINRES"]),
                "CGBypass": BoolTag(),
                "nIterations": IntTag(),
                "recomputeResidual": BoolTag(),
                "residualDiffThreshold": FloatTag(),
                "residualTol": FloatTag(),
            },
        ),
    },
    "fluid": {
        "target-mu": TagContainer(
            allow_list_representation=True,
            subtags={
                "mu": FloatTag(write_tagname=False, optional=False),
                "outerLoop": BoolTag(write_tagname=False),
            },
        ),
        "fluid": TagContainer(
            subtags={
                "type": StrTag(
                    options=[
                        "None",
                        "LinearPCM",
                        "NonlinearPCM",
                        "SaLSA",
                        "ClassicalDFT",
                    ],
                    write_tagname=False,
                    optional=False,
                ),
                "Temperature": FloatTag(write_tagname=False),
                "Pressure": FloatTag(write_tagname=False),
            }
        ),
        "fluid-solvent": MultiformatTag(
            can_repeat=True,
            format_options=[
                TagContainer(
                    subtags={
                        "name": StrTag(
                            options=[
                                "CarbonDisulfide",
                                "CCl4",
                                "CH2Cl2",
                                "CH3CN",
                                "Chlorobenzene",
                                "DMC",
                                "DMF",
                                "DMSO",
                                "EC",
                                "Ethanol",
                                "EthyleneGlycol",
                                "EthylEther",
                                "Glyme",
                                "H2O",
                                "Isobutanol",
                                "Methanol",
                                "Octanol",
                                "PC",
                                "THF",
                            ],
                            write_tagname=False,
                        ),
                        "concentration": FloatTag(write_tagname=False),
                        "functional": StrTag(
                            options=[
                                "BondedVoids",
                                "FittedCorrelations",
                                "MeanFieldLJ",
                                "ScalarEOS",
                            ],
                            write_tagname=False,
                        ),
                        **deepcopy(JDFTXFluid_subtagdict),
                    }
                ),
                TagContainer(
                    subtags={
                        "name": StrTag(
                            options=[
                                "CarbonDisulfide",
                                "CCl4",
                                "CH2Cl2",
                                "CH3CN",
                                "Chlorobenzene",
                                "DMC",
                                "DMF",
                                "DMSO",
                                "EC",
                                "Ethanol",
                                "EthyleneGlycol",
                                "EthylEther",
                                "Glyme",
                                "H2O",
                                "Isobutanol",
                                "Methanol",
                                "Octanol",
                                "PC",
                                "THF",
                            ],
                            write_tagname=False,
                        ),
                        "concentration": StrTag(options=["bulk"], write_tagname=False),
                        "functional": StrTag(
                            options=[
                                "BondedVoids",
                                "FittedCorrelations",
                                "MeanFieldLJ",
                                "ScalarEOS",
                            ],
                            write_tagname=False,
                        ),
                        **deepcopy(JDFTXFluid_subtagdict),
                    }
                ),
            ],
        ),
        "fluid-anion": TagContainer(
            subtags={
                "name": StrTag(
                    options=["Cl-", "ClO4-", "F-"], write_tagname=False, optional=False
                ),
                "concentration": FloatTag(write_tagname=False, optional=False),
                "functional": StrTag(
                    options=[
                        "BondedVoids",
                        "FittedCorrelations",
                        "MeanFieldLJ",
                        "ScalarEOS",
                    ],
                    write_tagname=False,
                ),
                **deepcopy(JDFTXFluid_subtagdict),
            }
        ),
        "fluid-cation": TagContainer(
            subtags={
                "name": StrTag(
                    options=["K+", "Na+"], write_tagname=False, optional=False
                ),
                "concentration": FloatTag(write_tagname=False, optional=False),
                "functional": StrTag(
                    options=[
                        "BondedVoids",
                        "FittedCorrelations",
                        "MeanFieldLJ",
                        "ScalarEOS",
                    ],
                    write_tagname=False,
                ),
                **deepcopy(JDFTXFluid_subtagdict),
            }
        ),
        "fluid-dielectric-constant": TagContainer(
            subtags={
                "epsBulkOverride": FloatTag(write_tagname=False),
                "epsInfOverride": FloatTag(write_tagname=False),
            }
        ),
        "fluid-dielectric-tensor": TagContainer(
            subtags={
                "epsBulkXX": FloatTag(write_tagname=False, optional=False),
                "epsBulkYY": FloatTag(write_tagname=False, optional=False),
                "epsBulkZZ": FloatTag(write_tagname=False, optional=False),
            }
        ),
        "fluid-ex-corr": TagContainer(
            subtags={
                "kinetic": StrTag(
                    write_tagname=False, optional=False
                ),  # TODO: add options from: https://jdftx.org/CommandFluidExCorr.html
                "exchange-correlation": StrTag(
                    write_tagname=False
                ),  # TODO: add same options as elec-ex-corr
            }
        ),
        "fluid-mixing-functional": TagContainer(
            can_repeat=True,
            subtags={
                "fluid1": StrTag(
                    options=[
                        "CCl4",
                        "CH3CN",
                        "CHCl3",
                        "Cl-",
                        "ClO4-",
                        "CustomAnion",
                        "CustomCation",
                        "F-",
                        "H2O",
                        "Na(H2O)4+",
                        "Na+",
                    ],
                    write_tagname=False,
                    optional=False,
                ),
                "fluid2": StrTag(
                    options=[
                        "CCl4",
                        "CH3CN",
                        "CHCl3",
                        "Cl-",
                        "ClO4-",
                        "CustomAnion",
                        "CustomCation",
                        "F-",
                        "H2O",
                        "Na(H2O)4+",
                        "Na+",
                    ],
                    write_tagname=False,
                    optional=False,
                ),
                "energyScale": FloatTag(write_tagname=False, optional=False),
                "lengthScale": FloatTag(write_tagname=False),
                "FMixType": StrTag(
                    options=["LJPotential", "GaussianKernel"], write_tagname=False
                ),
            },
        ),
        "fluid-vdwScale": FloatTag(),
        "fluid-gummel-loop": TagContainer(
            subtags={
                "maxIterations": IntTag(write_tagname=False, optional=False),
                "Atol": FloatTag(write_tagname=False, optional=False),
            }
        ),
        "fluid-solve-frequency": StrTag(options=["Default", "Gummel", "Inner"]),
        "fluid-site-params": TagContainer(
            multiline_tag=True,
            can_repeat=True,
            subtags={
                "component": StrTag(
                    options=[
                        "CCl4",
                        "CH3CN",
                        "CHCl3",
                        "Cl-",
                        "ClO4-",
                        "CustomAnion",
                        "CustomCation",
                        "F-",
                        "H2O",
                        "Na(H2O)4+",
                        "Na+",
                    ],
                    optional=False,
                ),
                "siteName": StrTag(optional=False),
                "aElec": FloatTag(),
                "alpha": FloatTag(),
                "aPol": FloatTag(),
                "elecFilename": StrTag(),
                "elecFilenameG": StrTag(),
                "rcElec": FloatTag(),
                "Rhs": FloatTag(),
                "sigmaElec": FloatTag(),
                "sigmaNuc": FloatTag(),
                "Zelec": FloatTag(),
                "Znuc": FloatTag(),
            },
        ),
        "pcm-variant": StrTag(
            options=[
                "CANDLE",
                "CANON",
                "FixedCavity",
                "GLSSA13",
                "LA12",
                "SCCS_anion",
                "SCCS_cation",
                "SCCS_g03",
                "SCCS_g03beta",
                "SCCS_g03p",
                "SCCS_g03pbeta",
                "SCCS_g09",
                "SCCS_g09beta",
                "SGA13",
                "SoftSphere",
            ]
        ),
        "pcm-nonlinear-scf": TagContainer(
            multiline_tag=True,
            subtags={
                "energyDiffThreshold": FloatTag(),
                "history": IntTag(),
                "mixFraction": FloatTag(),
                "nIterations": IntTag(),
                "qMetric": FloatTag(),
                "residualThreshold": FloatTag(),
            },
        ),
        "pcm-params": TagContainer(
            multiline_tag=True,
            subtags={
                "cavityFile": StrTag(),
                "cavityPressure": FloatTag(),
                "cavityScale": FloatTag(),
                "cavityTension": FloatTag(),
                "eta_wDiel": FloatTag(),
                "ionSpacing": FloatTag(),
                "lMax": FloatTag(),
                "nc": FloatTag(),
                "pCavity": FloatTag(),
                "rhoDelta": FloatTag(),
                "rhoMax": FloatTag(),
                "rhoMin": FloatTag(),
                "screenOverride": FloatTag(),
                "sigma": FloatTag(),
                "sqrtC6eff": FloatTag(),
                "Zcenter": FloatTag(),
                "zMask0": FloatTag(),
                "zMaskH": FloatTag(),
                "zMaskIonH": FloatTag(),
                "zMaskSigma": FloatTag(),
                "Ztot": FloatTag(),
            },
        ),
    },
    "dynamics": {
        "vibrations": TagContainer(
            subtags={
                "dr": FloatTag(),
                "centralDiff": BoolTag(),
                "useConstraints": BoolTag(),
                "translationSym": BoolTag(),
                "rotationSym": BoolTag(),
                "omegaMin": FloatTag(),
                "T": FloatTag(),
                "omegaResolution": FloatTag(),
            }
        ),
        "barostat-velocity": TagContainer(
            subtags={
                "v1": FloatTag(write_tagname=False, optional=False),
                "v2": FloatTag(write_tagname=False, optional=False),
                "v3": FloatTag(write_tagname=False, optional=False),
                "v4": FloatTag(write_tagname=False, optional=False),
                "v5": FloatTag(write_tagname=False, optional=False),
                "v6": FloatTag(write_tagname=False, optional=False),
                "v7": FloatTag(write_tagname=False, optional=False),
                "v8": FloatTag(write_tagname=False, optional=False),
                "v9": FloatTag(write_tagname=False, optional=False),
            }
        ),
        "thermostat-velocity": TagContainer(
            subtags={
                "v1": FloatTag(write_tagname=False, optional=False),
                "v2": FloatTag(write_tagname=False, optional=False),
                "v3": FloatTag(write_tagname=False, optional=False),
            }
        ),
        "ionic-dynamics": TagContainer(
            multiline_tag=True,
            subtags={
                "B0": FloatTag(),
                "chainLengthP": FloatTag(),
                "chainLengthT": FloatTag(),
                "dt": FloatTag(),
                "nSteps": IntTag(),
                "P0": FloatTag(),  # can accept numpy.nan
                "statMethod": StrTag(options=["Berendsen", "None", "NoseHoover"]),
                "stress0": TagContainer(  # can accept numpy.nan
                    subtags={
                        "xx": FloatTag(write_tagname=False, optional=False),
                        "yy": FloatTag(write_tagname=False, optional=False),
                        "zz": FloatTag(write_tagname=False, optional=False),
                        "yz": FloatTag(write_tagname=False, optional=False),
                        "zx": FloatTag(write_tagname=False, optional=False),
                        "xy": FloatTag(write_tagname=False, optional=False),
                    }
                ),
                "T0": FloatTag(),
                "tDampP": FloatTag(),
                "tDampT": FloatTag(),
            },
        ),
    },
    "export": {
        # note that the below representation should be possible, but dealing with
        # arbitrary length nested TagContainers within the same line requires a lot of code changes
        # 'dump-name': TagContainer(allow_list_representation = True,
        #     subtags = {
        #     'format': StrTag(write_tagname = False, optional = False),
        #     # 'freq1': StrTag(),
        #     # 'format1': StrTag(),
        #     # 'freq2': StrTag(),
        #     # 'format2': StrTag(),
        #     'extra': TagContainer(can_repeat = True, write_tagname = False,
        #         subtags = {
        #         'freq': StrTag(write_tagname = False, optional = False),
        #         'format': StrTag(write_tagname = False, optional = False),
        #         }),
        'dump-name': StrTag(),
        # 'dump': TagContainer(can_repeat = True,
        #     subtags = {
        #     'freq': StrTag(write_tagname = False, optional = False),
        #     'var': StrTag(write_tagname = False, optional = False)
        #     }),
        'dump-interval': TagContainer(can_repeat = True,
            subtags = {
            'freq': StrTag(options = ['Ionic', 'Electronic', 'Fluid', 'Gummel'], write_tagname = False, optional = False),
            'var': IntTag(write_tagname = False, optional = False)
            }),
        'dump-only': BoolTag(write_value = False),
        'band-projection-params': TagContainer(
            subtags = {
            'ortho': BoolTag(write_tagname = False, optional = False),
            'norm': BoolTag(write_tagname = False, optional = False),
            }),
        'density-of-states': TagContainer(multiline_tag = True,
            subtags = {
            'Total': BoolTag(write_value = False),
            'Slice': TagContainer(can_repeat = True,
                subtags = {
                'c0': FloatTag(write_tagname = False, optional = False),
                'c1': FloatTag(write_tagname = False, optional = False),
                'c2': FloatTag(write_tagname = False, optional = False),
                'r': FloatTag(write_tagname = False, optional = False),
                'i0': FloatTag(write_tagname = False, optional = False),
                'i1': FloatTag(write_tagname = False, optional = False),
                'i2': FloatTag(write_tagname = False, optional = False),
                }),
            'Sphere': TagContainer(can_repeat = True,
                subtags = {
                'c0': FloatTag(write_tagname = False, optional = False),
                'c1': FloatTag(write_tagname = False, optional = False),
                'c2': FloatTag(write_tagname = False, optional = False),
                'r': FloatTag(write_tagname = False, optional = False),
                }),
            'AtomSlice': TagContainer(can_repeat = True,
                subtags = {
                'species': StrTag(write_tagname = False, optional = False),
                'atomIndex': IntTag(write_tagname = False, optional = False),
                'r': FloatTag(write_tagname = False, optional = False),
                'i0': FloatTag(write_tagname = False, optional = False),
                'i1': FloatTag(write_tagname = False, optional = False),
                'i2': FloatTag(write_tagname = False, optional = False),
                }),
            'AtomSphere': TagContainer(can_repeat = True,
                subtags = {
                'species': StrTag(write_tagname = False, optional = False),
                'atomIndex': IntTag(write_tagname = False, optional = False),
                'r': FloatTag(write_tagname = False, optional = False),
                }),
            'File': StrTag(),
            'Orbital': TagContainer(can_repeat = True,
                subtags = {
                'species': StrTag(write_tagname = False, optional = False),
                'atomIndex': IntTag(write_tagname = False, optional = False),
                'orbDesc': StrTag(write_tagname = False, optional = False),
                }),
            'OrthoOrbital': TagContainer(can_repeat = True,
                subtags = {
                'species': StrTag(write_tagname = False, optional = False),
                'atomIndex': IntTag(write_tagname = False, optional = False),
                'orbDesc': StrTag(write_tagname = False, optional = False),
                }),
            'Etol': FloatTag(),
            'Esigma': FloatTag(),
            'EigsOverride': StrTag(),
            'Occupied': BoolTag(write_value = False),
            'Complete': BoolTag(write_value = False),
            'SpinProjected': TagContainer(can_repeat = True,
                subtags = {
                'theta': FloatTag(write_tagname = False, optional = False),
                'phi': FloatTag(write_tagname = False, optional = False),
                }),
            'SpinTotal': BoolTag(write_value = False),
            }),
        'dump-Eresolved-density': TagContainer(
            subtags = {
            'Emin': FloatTag(write_tagname = False, optional = False),
            'Emax': FloatTag(write_tagname = False, optional = False),
            }),
        'dump-fermi-density': MultiformatTag(can_repeat = True,
            format_options = [
            BoolTag(write_value = False),
            FloatTag(),
            ]),
        'bgw-params': TagContainer(multiline_tag = True,
            subtags = {
            'nBandsDense': IntTag(),
            'nBandsV': IntTag(),
            'blockSize': IntTag(),
            'clusterSize': IntTag(),
            'Ecut_rALDA': FloatTag(),
            'EcutChiFluid': FloatTag(),
            'rpaExx': BoolTag(),
            'saveVxc': BoolTag(),
            'saveVxx': BoolTag(),
            'offDiagV': BoolTag(),
            'elecOnly': BoolTag(),
            'freqBroaden_eV': FloatTag(),
            'freqNimag': IntTag(),
            'freqPlasma': FloatTag(),
            'freqReMax_eV': FloatTag(),
            'freqReStep_eV': FloatTag(),
            'kernelSym_rALDA': BoolTag(),
            'kFcut_rALDA': FloatTag(),
            'q0': TagContainer(
                subtags = {
                'q0x': FloatTag(write_tagname = False, optional = False),
                'q0y': FloatTag(write_tagname = False, optional = False),
                'q0z': FloatTag(write_tagname = False, optional = False),
                })
            }),
        'forces-output-coords': StrTag(options = ['Cartesian', 'Contravariant', 'Lattice', 'Positions']),
        'polarizability': TagContainer(
            subtags = {
            'eigenBasis': StrTag(options = ['External', 'NonInteracting', 'Total'], write_tagname = False, optional = False),
            'Ecut': FloatTag(write_tagname = False),
            'nEigs': IntTag(write_tagname = False),
            }),
        'polarizability-kdiff': TagContainer(
            subtags = {
            'dk0': FloatTag(write_tagname = False, optional = False),
            'dk1': FloatTag(write_tagname = False, optional = False),
            'dk2': FloatTag(write_tagname = False, optional = False),
            'dkFilenamePattern': StrTag(write_tagname = False),
            }),
        'potential-subtraction': BoolTag(),
    },
    "misc": {
        "debug": StrTag(
            options=[
                "Ecomponents",
                "EigsFillings",
                "Fluid",
                "Forces",
                "KpointsBasis",
                "MuSearch",
                "Symmetries",
            ],
            can_repeat=True,
        ),
        "pcm-nonlinear-debug": TagContainer(
            subtags={
                "linearDielectric": BoolTag(write_tagname=False, optional=False),
                "linearScreening": BoolTag(write_tagname=False, optional=False),
            }
        ),
    },
}


def get_dump_tag_container():
    subtags = {}
    for freq in JDFTXDumpFreqOptions:
        subsubtags = {}
        for var in JDFTXDumpVarOptions:
            subsubtags[var] = BoolTag(write_value = False)
        subtags[freq] = BoolTagContainer(subtags = subsubtags, write_tagname = True, can_repeat=True)
    dump_tag_container = DumpTagContainer(subtags = subtags, write_tagname = True, can_repeat=True)
    return dump_tag_container
MASTER_TAG_LIST["export"]["dump"] = get_dump_tag_container()


__PHONON_TAGS__ = ["phonon"]
__WANNIER_TAGS__ = [
    "wannier",
    "wannier-center-pinned",
    "wannier-dump-name",
    "wannier-initial-state",
    "wannier-minimize",
    "defect-supercell",
]
__TAG_LIST__ = [tag for group in MASTER_TAG_LIST for tag in MASTER_TAG_LIST[group]]
__TAG_GROUPS__ = {
    tag: group for group in MASTER_TAG_LIST for tag in MASTER_TAG_LIST[group]
}


def get_tag_object(tag):
    return MASTER_TAG_LIST[__TAG_GROUPS__[tag]][tag]
