Change log
==========

v0.1.2
------

New features:

- ``ensure_success`` option added to ``run_locally``.
- Better graph visualisation.
- Updating the name of a job from a maker now propogates the name change to the maker.
- ``Job.update_maker_kwargs`` with ``nested=True`` now applies the updates to makers
  in the kwargs or args of the job.

v0.1.1
------

Docs updates.

v0.1.0
------

Major changes:

- ``Schema`` class removed. Any pydantic model can now be an output schema.

Enhancements:

- ``JobStore.get_output`` now resolves references in the output of other jobs.
- ``JobStore.get_output``: ``which`` now supports specifying a specific job index.
- Better support for circular and missing references in ``JobStore.get_output`` and
  ``OutputReference.resolve``.
- Update dependencies to use latest jsanitize features.

Bug fixes:

- Fixed issue with references in flow of flows (@davidwaroquiers, #18).
- Makes now allows non-default parameters (fixes: #13).
- Fix reference cache with multiple indexes.

v0.0.2
------

Testing automated releases.

v0.0.1
------

Initial release containing:

- ``Job``, ``Flow``, ``Maker``, and ``JobStore`` API.
- Tools for running Flows locally.
- Fireworks integration.
