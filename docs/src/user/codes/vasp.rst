====
VASP
====

Configuration
-------------

List of settings.

setting vasp_cmd (also this can be set as an environment variable.

VASP defaults, PBEsol etc.

List of VASP workflows
----------------------

.. csv-table::
   :file: vasp-workflows.csv
   :widths: 40, 20, 40
   :header-rows: 1

Static
^^^^^^

A static VASP calculation (i.e., no relaxation).

Relax
^^^^^

A VASP relaxation calculation. Full structural relaxation is performed.

Tight Relax
^^^^^^^^^^^

A VASP relaxation calculation using tight convergence parameters. Full structural
relaxation is performed. This workflow is useful when small forces are required, such
as before calculating phonon properties.

Dielectric
^^^^^^^^^^

A VASP calculation to obtain dielectric properties. The static and high-frequency
dielectric constants are obtained using density functional perturbation theory.

HSE06 Static
^^^^^^^^^^^^

A static VASP calculation (i.e., no relaxation) using the HSE06 exchange correlation
functional.

HSE06 Relax
^^^^^^^^^^^

A VASP relaxation calculation using the HSE06 functional. Full structural relaxation
is performed.

HSE06 Tight Relax
^^^^^^^^^^^^^^^^^

A VASP relaxation calculation using tight convergence parameters with the HSE06
functional. Full structural relaxation is performed.

Double Relax
^^^^^^^^^^^^

Perform two back-to-back relaxations. This can often help avoid errors arising from
Pulay stress.

Band Structure
^^^^^^^^^^^^^^

Calculate the electronic band structure. This flow consists of three calculations:

1. A static calculation to generate the charge density.
2. A non-self-consistent field calculation on a dense uniform mesh.
3. A non-self-consistent field calculation on the high-symmetry k-point path to generate
   the line mode band structure.

.. Note::

   Band structure objects are automatically stored in the ``data`` store due to
   limitations on mongoDB collection sizes.

HSE06 Band Structure
^^^^^^^^^^^^^^^^^^^^

Calculate the electronic band structure using HSE06. This flow consists of three
calculations:

1. A HSE06 static calculation to generate the charge density.
2. A HSE06 calculation on a dense uniform mesh.
3. A HSE06 calculation on the high-symmetry k-point path using zero weighted k-points.

.. Note::

   Band structure objects are automatically stored in the ``data`` store due to
   limitations on mongoDB collection sizes.

Elastic Constant
^^^^^^^^^^^^^^^^

Calculate the elastic constant of a material. Initially, a tight structural relaxation
is performed to obtain the structure in a state of approximately zero stress.
Subsequently, perturbations are applied to the lattice vectors and the resulting
stress tensor is calculated from DFT, while allowing for relaxation of the ionic degrees
of freedom. Finally, constitutive relations from linear elasticity, relating stress and
strain, are employed to fit the full 6x6 elastic tensor. From this, aggregate properties
such as Voigt and Reuss bounds on the bulk and shear moduli are derived.

See the Materials Project `documentation on elastic constants
<https://docs.materialsproject.org/methodology/elasticity/>`_ for more details.

.. Note::
    It is strongly recommended to symmetrize the structure before passing running this
    this workflow. Otherwise, the symmetry reduction routines will not be as
    effective at reducing the number of deformations needed.

Modifying input sets
--------------------

The inputs for a calculation can be modified in several ways. Every VASP job
takes a :obj:`.VaspInputSetGenerator` as an argument (``input_set_generator``). One
option is to specify an alternative input set generator:

.. code-block:: python

    from atomate2.vasp.sets.core import StaticSetGenerator
    from atomate2.vasp.jobs.core import StaticMaker

    # create a custom input generator set with a larger ENCUT
    my_custom_set = StaticSetGenerator(user_incar_settings={"ENCUT": 800})

    # initialise the static maker to use the custom input set generator
    static_maker = StaticMaker(input_set_generator=my_custom_set)

    # create a job using the customised maker
    static_job = static_maker.make(structure)

The second approach is to edit the job after it has been made. All VASP jobs have a
``maker`` attribute containing a *copy* of the ``Maker`` that made them. Updating
the ``input_set_generator`` attribute maker will update the input set that gets
written:

.. code-block:: python

    static_job.maker.input_set_generator.user_incar_settings["LOPTICS"] = True

Finally, sometimes you have workflow containing many VASP jobs. In this case it can be
tedious to update the input sets for each job individually. Atomate2 provides helper
functions called "powerups" that can apply settings updates to all VASP jobs in a flow.
These powerups also contain filters for the name of the job and the maker used to
generate them.

.. code-block:: python

    from atomate2.vasp.powerups import update_user_incar_settings
    from atomate2.vasp.flows.elastic import ElasticMaker
    from atomate2.vasp.core.elastic import ElasticRelaxMaker

    # make a flow to calculate the elastic constants
    elastic_flow = ElasticMaker().make(structure)

    # update the ENCUT of all VASP jobs in the flow
    update_user_incar_settings(elastic_flow, {"ENCUT": 200})

    # only update VASP jobs which have "deformation" in the job name.
    update_user_incar_settings(elastic_flow, {"ENCUT": 200}, name_filter="deformation")

    # only update VASP jobs which were generated by an ElasticRelaxMaker
    update_user_incar_settings(elastic_flow, {"ENCUT": 200}, class_filter=ElasticRelaxMaker)

Chaining workflows
------------------

All VASP workflows are constructed using the ``Maker.make()`` function. The arguments
for this function always include:

- ``structure``: A pymatgen structure.
- ``prev_vasp_dir``: A previous VASP directory to copy output files from.

There are two options when chaining workflows:

1. Use only the structure from the previous calculation. This can be achieved by only
   setting the ``structure`` argument.
2. Use the structure and additional outputs from a previous calculation. By default,
   these outputs include INCAR settings, the band gap (used to automatically
   set KSPACING), and the magnetic moments. Some workflows will also use other outputs.
   For example, the `Band Structure`_ workflow will copy the CHGCAR file (charge
   density) from the previous calculation. This can be achieve by setting both the
   ``structure`` and ``prev_vasp_dir`` arguments.

These two examples are illustrated in the code below, where we chain a relaxation
calculation and a static calculation.

.. code-block:: python

    from jobflow import Flow
    from atomate2.vasp.jobs.core import RelaxMaker, StaticMaker
    from pymatgen.core.structure import Structure

    si_structure = Structure.from_file("Si.cif")

    # create a relax job
    relax_job = RelaxMaker().make(structure=si_structure)

    # create a static job that will use only the structure from the relaxation
    static_job = StaticMaker().make(structure=relax_job.output.structure)

    # create a static job that will use additional outputs from the relaxation
    static_job = StaticMaker().make(
        structure=relax_job.output.structure, prev_vasp_dir=relax_job.output.dir_name
    )

    # create a flow including the two jobs and set the output to be that of the static
    my_flow = Flow([relax_job, static_job], output=static_job.output)
