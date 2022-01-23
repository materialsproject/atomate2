Change log
==========

v0.0.4
------

Lots of improvements and bug fixes this release.

New features:

- AMSET workflow.
- Electron phonon band gap renormalisation workflow.
- Specific uniform and line mode band structure makers.
- Optics maker.
- Transmuter maker.

Enhancements:

- Support for automatic handling of ISPIN.
- Add MP base sets (@arosen93) [#27]
- Docs updates (@arosen93) [#13, #17]
- Options to strip band structure and DOS projects to reduce object sizes.
- Input sets now use generators to avoid serialization issues.
- Use smart efermi finding to remove errors with tetrahedron integration in VASP 6.
- Powerups can now work on ``Maker`` objects directly.

Bug fixes:

- Use PBEsol by default.
- Increase number of signficant figures when writing POSCAR files.
- Remove unused INCAR settings (@arosen93)
- Add missing LASPH flags on vdW functionals (@arosen93) [#31]
- Use ``NSW=0`` in static calculations (@arosen93) [#10]
- ``LREAL = False`` in static jobs by default (@arosen93) [#23]
- Add missing functionals in output schema (@arosen93) [#12]
- Many output schema fixes.
- Better support for FireWorks.
- Support writing additional files in VASP jobs.

v0.0.3
------

Many updates to use the latest jobflow store features.

v0.0.2
------

Automated releases.

v0.0.1
------

Initial release.
