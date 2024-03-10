(key_concepts_overview)=

# Key concepts in atomate2: `Job` & `Flow` `Makers`, `InputSet`, `TaskDocument`, and `Builder`

# Introduction
This tutorial is supposed to give the user a high-level overview of the key concepts in atomate2, that will eloborate on job & flow makers, input sets, task documents, and builders in greater detail.

## `Job` and `Flow` makers

`Job` and `Flow` makers are dataclasses that inherit from the `Maker` [jobflow](https://github.com/materialsproject/jobflow/blob/main/src/jobflow/core/maker.py) dataclass.
The `Maker` is a base class for constructing the aforementioned `Job` and `Flow` objects. It is inheriting from the `MSONable` (Monty JSON) class, that allows job or flow output to be in a JSON serializable dict format.
This makes it easier to handle and unify output from the various computational chemistry tasks (e.g. chemical bonding analysis, elastic constant calculations, force field applications and many more) and of the many supported software packages (like VASP, phonopy and more).  <!--is there a list of supported software on the github pages?-->
The output data is handled with so-called `TaskDocuments` in a very convenient way.

The `Maker` class from jobflow has two main functionalities, that are vital for any inheriting job or flow maker: the `make` function and the functionality to update keyword arguments (kwargs).

```
@dataclass
class Maker(MSONable):
    """
    Base maker (factory) class for constructing :obj:`Job` and :obj:`Flow` objects.
    [...]
    """
    def make(...) -> jobflow.Flow | jobflow.Job:
        """Make a job or a flow - must be overridden with a concrete implementation."""
        raise NotImplementedError
    [...]
    def update_kwargs(...):
```

The functions that raise an `NotImplementedError` like the `make` function have be overridden for each specific job or flow maker with its own specific make functionalities.

An example for a `Job` `Maker` is the `LobsterMaker`:

```
@dataclass
class LobsterMaker(Maker):
    """
    LOBSTER job maker.
    [...]
    """
    name: str = "lobster"
    [...]

    @job(output_schema=LobsterTaskDocument, data=[CompleteCohp, LobsterCompleteDos])
    def make(
        self,
        wavefunction_dir: str | Path = None,
        basis_dict: dict | None = None,
    ) -> LobsterTaskDocument:
        """
        Run a LOBSTER calculation.
        [...]
        """
```
This class incorporates [LOBSTER](http://cohp.de/) specific input and output data, i.e. the `wavefunction_dir` and `basis_dict` as input in `make` that returns the `LobsterMaker`-class specific output as a `TaskDocument`.
As a job maker, this maker will then create jobs to execute the LOBSTER runs and store the output in the `LobsterTaskDocument` format.

In contrast to a job maker, the `make` function of a `Flow Maker` will return a `Flow` object (sequential collection of `Job` objects or other `Flow` objects) instead of a task document.
As an example we take the `BasePhononMaker` (that can be used in combination with VASP or the various (machine learned) force fields):
```
@dataclass
class BasePhononMaker(Maker, ABC):
    """
    Maker to calculate harmonic phonons with a DFT/force field code and Phonopy.

    [...]
    """

    name: str = "phonon"
    [...]

    def make(...) -> Flow:
```
This maker will return a flow that provides all the necessary steps and subroutines that are needed to complete the phonon calculations.
Oftentimes, such flows then involve dynamic jobs, as e.g. for the phonon calculations, the number of supercells with individual atomic displacements will be decided upon runtime.
In this particular case, the flow maker `BasePhononMaker` is also inheriting from `ABC` (Abstract Base Classes). <!--more comprehensive explanation on ABCs?-->


# InputSet
tbc

# TaskDocument
a TaskDocument (TaskDoc for short) is a dictionary object that contains all the information of the respective computational chemistry calculation run.

tbc

# Builder
tbc
