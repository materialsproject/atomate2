============
Contributing
============

Thank you for your interest in contributing to atomate2siesta! This guide will help you get started.

----

Getting Started
===============

Development Installation
------------------------

Clone the repository and install in development mode:

.. code-block:: bash

   git clone https://github.com/materialsproject/atomate2.git
   cd atomate2siesta
   pip install -e ".[dev,tests,docs]"

This installs atomate2siesta with all development dependencies:

* **dev**: Pre-commit hooks, code formatters
* **tests**: pytest, coverage tools
* **docs**: Sphinx documentation tools

Pre-commit Hooks
----------------

Set up pre-commit hooks for code quality:

.. code-block:: bash

   pre-commit install

The hooks will automatically run:

* ``ruff`` - Python linter and formatter
* ``mypy`` - Type checking
* ``trailing-whitespace`` - Remove trailing spaces
* ``end-of-file-fixer`` - Ensure files end with newline

----

Development Workflow
====================

1. Create a Branch
------------------

Create a feature branch for your changes:

.. code-block:: bash

   git checkout -b feature/your-feature-name

Use descriptive branch names:

* ``feature/phonon-animations`` - New feature
* ``fix/scf-convergence-bug`` - Bug fix
* ``docs/tutorial-update`` - Documentation
* ``test/parser-coverage`` - Test improvements

2. Make Changes
---------------

Follow the project conventions:

**Code Style**:

* Follow PEP 8 conventions
* Use type hints for all functions
* Maximum line length: 88 characters (Black default)
* Use docstrings for all public functions

**Naming Conventions**:

* Classes: ``PascalCase`` (e.g., ``RelaxMaker``)
* Functions/variables: ``snake_case`` (e.g., ``make_structure``)
* Constants: ``UPPER_CASE`` (e.g., ``DEFAULT_CUTOFF``)
* Private methods: ``_leading_underscore``

**Documentation**:

* Add docstrings to all public APIs
* Follow NumPy docstring format
* Include examples in docstrings
* Update relevant tutorials

3. Write Tests
--------------

All new features must include tests:

.. code-block:: python

   # tests/jobs/test_my_feature.py
   import pytest
   from atomate2.siesta.jobs.my_feature import MyFeatureMaker

   class TestMyFeatureMaker:
       """Tests for MyFeatureMaker"""

       def test_basic_functionality(self, si_structure):
           """Test basic maker creation"""
           maker = MyFeatureMaker()
           job = maker.make(si_structure)

           assert job.name == "my_feature"
           assert job.input_set_generator is not None

       def test_with_user_params(self, si_structure):
           """Test with custom parameters"""
           maker = MyFeatureMaker(
               user_params={"PAO.BasisSize": "DZP"}
           )
           job = maker.make(si_structure)

           assert "PAO.BasisSize" in job.input_set_generator.user_params

**Testing Patterns**:

* Use fixtures from ``tests/conftest.py``
* Mock SIESTA execution (don't run real calculations)
* Test edge cases (None values, empty structures)
* Use parametrize for multiple test cases
* Aim for >80% coverage for new code

**Run Tests**:

.. code-block:: bash

   # Run all tests
   pytest

   # Run specific test file
   pytest tests/jobs/test_my_feature.py

   # Run with coverage
   pytest --cov=atomate2.siesta --cov-report=term-missing

   # Run fast (skip slow tests)
   pytest -m "not slow"

4. Update Documentation
-----------------------

Update relevant documentation files:

**Code Documentation**:

* Add/update docstrings
* Include examples in docstrings
* Document parameters and return types

**User Documentation**:

* Update ``docs/source/features.rst`` for new features
* Create tutorial if needed (``tutorials/XX-feature-name/``)
* Update ``ATOMATE2SIESTA.md`` for architecture changes

**Developer Documentation**:

* Update changelog for feature additions
* Document test strategies and coverage improvements

5. Run Quality Checks
----------------------

Before committing, run quality checks:

.. code-block:: bash

   # Format code
   ruff format .

   # Check linting
   ruff check .

   # Type checking
   mypy src

   # Run tests
   pytest

   # Build docs (optional)
   cd docs && make html

6. Commit Changes
-----------------

Write clear commit messages:

.. code-block:: bash

   git add .
   git commit -m "feat: add phonon animation workflow"

**Commit Message Format**:

* ``feat:`` - New feature
* ``fix:`` - Bug fix
* ``docs:`` - Documentation changes
* ``test:`` - Test additions/changes
* ``refactor:`` - Code refactoring
* ``perf:`` - Performance improvements
* ``chore:`` - Maintenance tasks

**Good Commit Messages**:

* ``feat: add thermal expansion plotting function``
* ``fix: correct PAO.BasisSize parameter matching``
* ``test: add parser edge case coverage``
* ``docs: update test coverage achievements``

7. Push and Create PR
----------------------

Push your branch and create a pull request:

.. code-block:: bash

   git push origin feature/your-feature-name

On GitHub:

* Create pull request with clear description
* Reference related issues
* Add examples/screenshots if relevant
* Wait for CI checks to pass
* Address review comments

----

Contribution Areas
==================

Where to Contribute
-------------------

**High Priority**:

* **Core Job Tests** - RelaxMaker, StaticMaker, BandStructureMaker (CRITICAL)
* **Flow Tests** - Workflow composition and integration
* **Input Set Tests** - FDF file generation and parameter merging
* **Documentation** - Tutorial updates and examples

**Medium Priority**:

* **Feature Testing** - Adsorption, surface energy, phonons
* **Dataclass Validation** - Edge cases and parameter validation
* **Performance Optimization** - Profiling and improvements
* **Tutorial Creation** - Missing tutorials (Tutorial 19, etc.)

**Feature Requests**:

* **Phonon Animations** - Mode visualization
* **Optical Properties** - Enhanced analysis
* **Defect Calculations** - Point defects workflow
* **Machine Learning Potentials** - Integration with ML tools

Reporting Bugs
--------------

Found a bug? Please report it:

1. Check existing issues: https://github.com/materialsproject/atomate2/issues
2. If new, create issue with:
   * Clear description of problem
   * Steps to reproduce
   * Expected vs actual behavior
   * Environment (OS, Python version, SIESTA version)
   * Minimal example code

Requesting Features
-------------------

Have a feature idea?

1. Open discussion: https://github.com/materialsproject/atomate2/discussions
2. Provide:
   * Use case description
   * Proposed API/interface
   * Example usage
   * Why it's valuable

----

Code Review Process
===================

What Reviewers Look For
-----------------------

* **Correctness**: Does the code work as intended?
* **Tests**: Are there sufficient tests? Do they cover edge cases?
* **Documentation**: Are docstrings clear? Is user documentation updated?
* **Style**: Does code follow project conventions?
* **Performance**: Are there obvious performance issues?
* **Compatibility**: Does it break existing functionality?

Review Timeline
---------------

* Initial review: Usually within 1-3 days
* Follow-up: Within 1-2 days of addressing comments
* Merge: After approval and passing CI checks

----

Testing Guidelines
==================

Testing Philosophy
------------------

* **Mock-based testing**: Don't run real SIESTA calculations in tests
* **Fast execution**: Test suite should complete in < 30 seconds
* **Clear naming**: Test names describe what they test
* **Edge cases**: Test None, empty, and error conditions
* **Fixtures**: Use shared fixtures from ``conftest.py``

Coverage Goals
--------------

**Current Status**:

* Overall: 44% (target: 80%)
* Tier system: 100% ✅
* Schemas: 87-97% ✅
* File client: 86% ✅
* Parser: 39% 🔄
* Jobs/Flows: ~15% ⏭️

**Target Coverage by Component**:

* Critical infrastructure (jobs, flows, sets): 70-80%
* Workflows (phonons, surfaces, etc.): 75%
* Utilities and helpers: many
* Dataclass validation: 60%

Test Organization
-----------------

.. code-block:: text

   tests/
   ├── conftest.py              # Shared fixtures
   ├── jobs/
   │   ├── test_core.py         # RelaxMaker, StaticMaker, etc.
   │   ├── test_phonopy.py      # Phonon calculations
   │   └── test_adsorption.py   # Adsorption workflows
   ├── flows/
   │   ├── test_core.py         # Core flow infrastructure
   │   └── test_phonons.py      # Phonon flows
   ├── sets/
   │   ├── test_core.py         # Input set generators
   │   ├── test_parser.py       # SIESTA output parsing
   │   └── test_tiers.py        # Tier system
   ├── schemas/
   │   ├── test_task.py         # Task documents
   │   └── test_calculation.py  # Calculation schemas
   ├── utils/
   │   └── test_file_client.py  # SSH/SFTP operations
   └── integration/
       └── test_workflows.py    # End-to-end tests

----

Documentation Guidelines
========================

Docstring Format
----------------

Use NumPy-style docstrings:

.. code-block:: python

   def plot_thermal_expansion(
       gruneisen_doc: dict | GruneisenParameterDocument,
       bulk_modulus: float | None = None,
       temperature_range: tuple[float, float] = (0, 1000),
       output_file: str = "thermal_expansion.png",
   ) -> None:
       """
       Plot temperature-dependent thermal expansion coefficient.

       Uses the Debye model to calculate volumetric thermal expansion:
       α_V = γ · C_V / (B · V)

       Parameters
       ----------
       gruneisen_doc : dict or GruneisenParameterDocument
           Grüneisen parameter document from workflow
       bulk_modulus : float or None, optional
           Bulk modulus in GPa. If None, will estimate from structure
       temperature_range : tuple of float, optional
           (T_min, T_max) in Kelvin, default (0, 1000)
       output_file : str, optional
           Output filename, default "thermal_expansion.png"

       Returns
       -------
       None
           Saves plot to file

       Examples
       --------
       >>> from atomate2.siesta.jobs.gruneisen_plotting import plot_thermal_expansion
       >>> plot_thermal_expansion(
       ...     gruneisen_doc,
       ...     bulk_modulus=100.0,
       ...     temperature_range=(0, 500)
       ... )

       Notes
       -----
       If bulk modulus is not provided, it will be estimated using:
       B ≈ 100 GPa for most materials. This is approximate and should
       be replaced with calculated or experimental values for accuracy.

       See Also
       --------
       calculate_thermal_expansion : Calculate α(T) without plotting
       plot_gruneisen_vs_frequency : Plot Grüneisen vs frequency
       """
       # Implementation

Tutorial Structure
------------------

Each tutorial should include:

1. **Overview**: What the tutorial covers
2. **Prerequisites**: Required knowledge/previous tutorials
3. **Theory** (optional): Background physics/chemistry
4. **Setup**: Environment and structure preparation
5. **Step-by-step**: Code examples with explanations
6. **Results**: Expected output and interpretation
7. **Exercises** (optional): Try-it-yourself tasks
8. **Next Steps**: What to learn next

----

Getting Help
============

Resources
---------

* **Documentation**: https://atomate2siesta.readthedocs.io
* **GitHub Issues**: https://github.com/materialsproject/atomate2/issues
* **Discussions**: https://github.com/materialsproject/atomate2/discussions

Contact
-------

* **Maintainer**: Arsalan Akhtar (sr.arsalan.akhtar@gmail.com)
* **Issues**: Use GitHub issue tracker
* **Feature Discussions**: Use GitHub discussions

----

Code of Conduct
===============

Be Respectful
-------------

* Treat all contributors with respect
* Welcome newcomers and help them learn
* Be patient with questions
* Provide constructive feedback

Be Professional
---------------

* Focus on the code, not the person
* Accept criticism gracefully
* Acknowledge contributions
* Give credit where due

Be Collaborative
----------------

* Share knowledge and best practices
* Help others learn and improve
* Review PRs thoughtfully
* Contribute to discussions

----

Recognition
===========

Contributors are recognized in:

* ``CONTRIBUTORS.md`` file (if exists)
* Release notes
* Commit history
* Co-authorship on papers (for significant contributions)

Thank you for contributing to atomate2siesta! 🎉

----

.. seealso::

   * :doc:`changelog` - Release history
   * :doc:`features` - Feature documentation
