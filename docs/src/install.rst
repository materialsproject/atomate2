Install jobflow
===============

You can install jobflow with ``pip`` or from source.

Pip
---

You can install the basic functionality of jobflow using pip::

   pip install jobflow

If you are planing to use jobflow with fireworks, you can install the optional fireworks
components::

   pip install jobflow[fireworks]

We also maintain other dependency sets for different subsets of functionality::

   pip install jobflow[vis]  # Install requirements for visualizing jobs and flows


Install from source
-------------------

To install jobflow from source, clone the repository from `github
<https://github.com/materialsproject/jobflow>`_::

    git clone https://github.com/materialsproject/jobflow.git
    cd jobflow
    pip install .

You can also install fireworks dependencies::

    pip install .[fireworks]

Or do a developer install by using the ``-e`` flag::

    pip install -e .


Test
----

Unit tests can be run from the source folder using ``pytest``. First, the requirements
to run tests must be installed::

    pip install .[tests]

And the tests run using::

    pytest

Building the documentation
--------------------------

The jobflow documentation can be built using the sphinx package. First, install the
necessary requirement::

    pip install .[docs]

Next, the docs can be built to the ``docs_build`` directory::

    sphinx-build docs/src docs_build
