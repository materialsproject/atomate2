Change log
==========

v0.0.7
------

New features:

- Include band-related features (e.g. band center, bandwidth, skewness, kurtosis) in
  VASP schema ([@arosen93](https://github.com/arosen93), [#92](https://github.com/materialsproject/atomate2/pull/92))
- Add `use_auto_ispin` and `update_user_potcar_functional` powerups

Enhancements:

- Add `is_hubbard` and `hubbards` to VASP task doc.
- Migrate build system to pyproject.toml.
- Migrate docs to jupyter-book.
- Docs improvements ([@janosh](https://github.com/janosh), [@mjwen](https://github.com/mjwen))

Bug fixes:

- Fix HSE tags.
- Fix running bader.
- Make potcar_spec argument usable ([@jmmshn](https://github.com/jmmshn), [#83](https://github.com/materialsproject/atomate2/pull/83))
- Replace monty which with shutil which ([@arosen93](https://github.com/arosen93), [#92](https://github.com/materialsproject/atomate2/pull/92))
- Fix `calculate_deformation_potentials()` ([@janosh](https://github.com/janosh), [#94](https://github.com/materialsproject/atomate2/pull/94))
- Fix gzipping of files with numerical suffixes ([@jmmshn](https://github.com/jmmshn), [#116](https://github.com/materialsproject/atomate2/pull/116))

v0.0.6
------

New features:

- cclib task document supporting virtually all popular molecular DFT codes out-of-the-box
  ([@arosen93](https://github.com/arosen93), [#64](https://github.com/materialsproject/atomate2/pull/64))

Enhancements:

- Add mag_density to VASP output doc ([@arosen93](https://github.com/arosen93), [#65](https://github.com/materialsproject/atomate2/pull/66))
- Double relax maker now supports two different Makers ([@arosen93](https://github.com/arosen93), [#32](https://github.com/materialsproject/atomate2/pull/32))

Bug fixes:

- Store band structure efermi in CalculationOutput ([@arosen93](https://github.com/arosen93), [#66](https://github.com/materialsproject/atomate2/pull/66))
- Support for VASP6 and latest pymatgen ([@arosen93](https://github.com/arosen93), [#75](https://github.com/materialsproject/atomate2/pull/75))
- Fixed atomate2 version string.
- Disabled orbital projections in the electron-phonon workflow.

v0.0.5
------

This version removed Python 3.7 support following numpy and pymatgen.

New features:

- Base schema for molecule task documents ([@arosen93](https://github.com/arosen93), [#54](https://github.com/materialsproject/atomate2/pull/54))

Bug fixes:

- Fix VASP relaxation using custodian "FULL_OPT" ([@arosen93](https://github.com/arosen93), [#42](https://github.com/materialsproject/atomate2/pull/42))
- Fix supercell generation and input sets in electron-phonon workflow.
- Fix `HSEBSSetGenerator` INCAR settings.
- Fix issue with magnetism in SOC structures.
- Fix bug with Fermi level and IBRION=1
- Better handling of URI generation.
- Tweak k-spacing formula to stop large band gaps giving negative values

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
- Add MP base sets ([@arosen93](https://github.com/arosen93), [#27](https://github.com/materialsproject/atomate2/pull/27))
- Docs updates ([@arosen93](https://github.com/arosen93), [#13](https://github.com/materialsproject/atomate2/pull/13) [#17](https://github.com/materialsproject/atomate2/pull/17))
- Options to strip band structure and DOS projects to reduce object sizes.
- Input sets now use generators to avoid serialization issues.
- Use smart efermi finding to remove errors with tetrahedron integration in VASP 6.
- Powerups can now work on `Maker` objects directly.

Bug fixes:

- Use PBEsol by default.
- Increase number of significant figures when writing POSCAR files.
- Remove unused INCAR settings ([@arosen93](https://github.com/arosen93))
- Add missing LASPH flags on vdW functionals ([@arosen93](https://github.com/arosen93), [#31](https://github.com/materialsproject/atomate2/pull/31))
- Use `NSW=0` in static calculations ([@arosen93](https://github.com/arosen93), [#10](https://github.com/materialsproject/atomate2/pull/10))
- `LREAL = False` in static jobs by default ([@arosen93](https://github.com/arosen93), [#23](https://github.com/materialsproject/atomate2/pull/23))
- Add missing functionals in output schema ([@arosen93](https://github.com/arosen93), [#12](https://github.com/materialsproject/atomate2/pull/12))
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
