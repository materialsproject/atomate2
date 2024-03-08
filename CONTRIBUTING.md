# Contributing to atomate2

We love your input! We want to make contributing as easy and
transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing or implementing new features
- Becoming a maintainer

## Reporting bugs, getting help, and discussion

atomate2 is still in development, so at the moment we
do not have a dedicated help forum. For the time being, please
submit questions and bugs to the
[GitHub issues page](https://github.com/materialsproject/atomate2/issues).

If you are making a bug report, incorporate as many elements of the
following as possible to ensure a timely response and avoid the
need for followups:

- A quick summary and/or background.
- Steps to reproduce - be specific! **Provide sample code.**
- What you expected would happen, compared to what actually happens.
- The full stack trace of any errors you encounter.
- Notes (possibly including why you think this might be happening,
  or steps you tried that didn't work).

We love thorough bug reports as this means the development team can
make quick and meaningful fixes. When we confirm your bug report,
we'll move it to the GitHub issues where its progress can be
further tracked.

## Contributing code modifications or additions through GitHub

We use GitHub to host code, to track issues and feature requests,
as well as accept pull requests. We maintain a list of all
contributors [here](https://materialsproject.github.io/atomate2/contributors.html).

Pull requests are the best way to propose changes to the codebase.
Follow the [GitHub flow](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow)
for more information on this procedure.

The basic procedure for making a PR is:

- Fork the repo and create your branch from master.
- Commit your improvements to your branch and push to your GitHub fork (repo).
- When you're finished, go to your fork and make a Pull Request. It will
  automatically update if you need to make further changes.

## How to Make a Great Pull Request

We have a few tips for writing good PRs that are accepted into the main repo:

- Use the Numpy Code style for all of your code. Find an example [here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy).
- Your code should have (4) spaces instead of tabs.
- If needed, update the documentation.
- **Write tests** for new features! Good tests are 100%, absolutely necessary
  for good code. We use the python `pytest` framework -- see some of the
  other tests in this repo for examples, or review the [Hitchhiker's guide
  to python](https://docs.python-guide.org/writing/tests) for some good
  resources on writing good tests.
- Understand your contributions will fall under the same license as this repo.

When you submit your PR, our CI service will automatically run your tests.
We welcome good discussion on the best ways to write your code, and the comments
on your PR are an excellent area for discussion.
