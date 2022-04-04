# atomate2 documentation

**Date**: {{ date }} | **Version**: {{ env.config.version }}

**Useful links**:
[Source Repository](https://github.com/materialsproject/atomate2) |
[Issues & Ideas](https://github.com/materialsproject/atomate2/issues) |
[Q&A Support](https://matsci.org/c/atomate)

Atomate2 is an open source library providing computational workflows for
automating first principles calculations.

````{panels}
:card: + intro-card text-center
:column: col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex
---
:img-top: _static/index_user_guide.svg

User guide
^^^^^^^^^^
The user guide provides in-depth information and tutorials for using *atomate2*.
+++
```{link-button} user
:type: ref
:text: To the user guide
:classes: btn-block btn-secondary stretched-link
```
---
:img-top: _static/index_support.svg

Support forum
^^^^^^^^^^^^^
You've read the user guide but still need help? Ask questions on the *atomate2*
support forum.
+++
```{link-button} https://matsci.org/c/atomate
:text: To the help forum
:classes: btn-block btn-secondary stretched-link
```
---
:img-top: _static/index_api.svg

API reference
^^^^^^^^^^^^^
The reference guide contains a detailed description of the *atomate2* API. It assumes
that you have an understanding of the key concepts.
+++
```{link-button} api
:type: ref
:text: To the reference guide
:classes: btn-block btn-secondary stretched-link
```
---
:img-top: _static/index_contribute.svg

Developer guide
^^^^^^^^^^^^^^^
Do you want to develop your own *atomate2* workflows? Want to improve
existing functionalities? The contributing guidelines will guide
you through the process of improving and developing on *atomate2*.
+++
```{link-button} dev
:type: ref
:text: To the development guide
:classes: btn-block btn-secondary stretched-link
```
````

```{toctree}
---
maxdepth: 2
hidden:
----
User Guide <user/index>
API Reference <reference/index>
Developer Guide <dev/index>
Support <https://matsci.org/c/atomate>
```
