# Developer Installation

You can install atomate2 with `pip` or from source.

## Install using pip

You can install the basic functionality of atomate2 using pip:

```bash
pip install atomate2
```

If you are planning to use atomate2 with fireworks, you can install the optional
fireworks components:

```bash
pip install atomate2[fireworks]
```

We also maintain other dependency sets for different subsets of functionality:

```bash
pip install atomate2[amset]  # Install requirements for running AMSET calculations
```

## Install from source

To install atomate2 from source, clone the repository from [github](https://github.com/materialsproject/atomate2)

```bash
git clone https://github.com/materialsproject/atomate2
cd atomate2
pip install .
```

You can also install fireworks dependencies:

```bash
pip install .[fireworks]
```

Or do a developer install by using the `-e` flag:

```bash
pip install -e .
```

### Installing pre-commit

If you're planning on contributing to the atomate2 source, you should also install
the developer requirements with:

```bash
pip install -e . --group dev
pre-commit install
```

The `pre-commit` command will ensure that changes to the source code match the
atomate2 style guidelines by running code linters such as `black`, `ruff`,
and `mypy` automatically with each commit.

## Running unit tests

Unit tests can be run from the source folder using `pytest`. First, the requirements
to run tests must be installed:

```bash
pip install . --group tests
```

And the tests run using:

```bash
pytest
```

## Building the documentation

The atomate2 documentation can be built using the sphinx package. First, install the requirements:

```bash
pip install . --group docs
```

Next, the docs can be built to the `docs_build` directory:

```bash
sphinx-build docs docs_build
```
