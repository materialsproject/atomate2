Change log
==========

v0.0.11
-------

Emmet new labels:
- Potcar
  - pot_type
- InputDoc
  - parameters
  - psuedo_potentials
  - potcar_spec
  - xc_override
  - is_lasph
  - is_hubbard
  - hubbards
- OutputDoc
  - energy_per_atom
  - bandgap
- TaskDoc
  - Now subclasses StructureMetadata
  - structure
  - included_objects
  - vasp_objects
  - entry
  - author
  - icsd_id
  - transformations
  - additional_json
  - dir_name
- TaskType now imported from emmet.core.vasp.enums (existed already)
- CalcsReversedDoc -> Calculation from emmet.core.vasp.calculation (added in this PR)
 - Existing fields remain the same, lots of new fields added.

Schema changes:
- PsuedoPotentialSummary -> Potcar
  - labels -> symbols
- AnalysisSymmary -> AnalysisDoc
  - delta_volume_as_percent -> delta_volume_percent
- InputSummary -> InputDoc
- OutputSummary -> OutputDoc
  - density added
- Status -> TaskState
- TaskDocument -> TaskDoc
  - task_type added
- Status -> TaskState


v0.0.10
-------

Lobster workflow with VASP implementation ([@JaGeo][jageo], [@naik-aakash][naik-aakash] [#200](https://github.com/materialsproject/atomate2/pull/200))

v0.0.9
------

New features:

- Defect formation energy workflow with VASP implementation ([@jmmshn][jmmshn], [#215](https://github.com/materialsproject/atomate2/pull/215))
- Job to retrieve a structure from the MP API at run-time ([@mkhorton][mkhorton], [#176](https://github.com/materialsproject/atomate2/pull/176]))

Enhancements:

- Documentation of phonon workflow ([@QuantumChemist][quantumchemist], [#232](https://github.com/materialsproject/atomate2/pull/232))
- Refactor defect code ([@jmmshn][jmmshn], [#214](https://github.com/materialsproject/atomate2/pull/214))
- Use `ruff` for linting ([@janosh][janosh], [#250](https://github.com/materialsproject/atomate2/pull/250))


Bug fixes:

- Use correct k-point density in phonon workflow ([@JaGeo][jageo], [#177](https://github.com/materialsproject/atomate2/pull/177))
- Fix use of `expanduser` path ([@nwinner][nwinner], [#180](https://github.com/materialsproject/atomate2/pull/180))
- Correct `calcs_reversed` to be in the proper order ([@Zhuoying][zhuoying], [#182](https://github.com/materialsproject/atomate2/pull/182))
- Bugfix for `store_volumetric_data` ([@jmmshn][jmmshn], [#212](https://github.com/materialsproject/atomate2/pull/212))


v0.0.8
------

New features:

- VASP Phonopy workflow ([@JaGeo][jageo], [#137](https://github.com/materialsproject/atomate2/pull/137))
- Molecular dynamics VASP job ([@mjwen][mjwen], [#134](https://github.com/materialsproject/atomate2/pull/134))

Enhancements:

- Update IO classes to use pymatgen base classes ([@rkingsbury][rkingsbury], [#141](https://github.com/materialsproject/atomate2/pull/141))
- Read and write VASP structures with higher precision ([@JaGeo][jageo], [#167](https://github.com/materialsproject/atomate2/pull/167))

Bug fixes:

- Fix code examples in docs ([@JaGeo][jageo], [#169](https://github.com/materialsproject/atomate2/pull/169))
- Fix f-orbital DOS properties ([@arosen93][arosen], [#138](https://github.com/materialsproject/atomate2/pull/138))
- Fix `mock_run_vasp` testing to accept args ([@mjwen][mjwen], [#151](https://github.com/materialsproject/atomate2/pull/151))
- Regenerate calc_types enum ([@mjwen][mjwen], [#153](https://github.com/materialsproject/atomate2/pull/153))

v0.0.7
------

New features:

- Include band-related features (e.g. band center, bandwidth, skewness, kurtosis) in
  VASP schema ([@arosen93][arosen], [#92](https://github.com/materialsproject/atomate2/pull/92))
- Add `use_auto_ispin` and `update_user_potcar_functional` powerups

Enhancements:

- Add `is_hubbard` and `hubbards` to VASP task doc.
- Migrate build system to pyproject.toml.
- Migrate docs to jupyter-book.
- Docs improvements ([@janosh][janosh], [@mjwen][mjwen])

Bug fixes:

- Fix HSE tags.
- Fix running bader.
- Make potcar_spec argument usable ([@jmmshn][jmmshn], [#83](https://github.com/materialsproject/atomate2/pull/83))
- Replace monty which with shutil which ([@arosen93][arosen], [#92](https://github.com/materialsproject/atomate2/pull/92))
- Fix `calculate_deformation_potentials()` ([@janosh][janosh], [#94](https://github.com/materialsproject/atomate2/pull/94))
- Fix gzipping of files with numerical suffixes ([@jmmshn][jmmshn], [#116](https://github.com/materialsproject/atomate2/pull/116))

v0.0.6
------

New features:

- cclib task document supporting virtually all popular molecular DFT codes out-of-the-box
  ([@arosen93][arosen], [#64](https://github.com/materialsproject/atomate2/pull/64))

Enhancements:

- Add mag_density to VASP output doc ([@arosen93][arosen], [#65](https://github.com/materialsproject/atomate2/pull/66))
- Double relax maker now supports two different Makers ([@arosen93][arosen], [#32](https://github.com/materialsproject/atomate2/pull/32))

Bug fixes:

- Store band structure efermi in CalculationOutput ([@arosen93][arosen], [#66](https://github.com/materialsproject/atomate2/pull/66))
- Support for VASP6 and latest pymatgen ([@arosen93][arosen], [#75](https://github.com/materialsproject/atomate2/pull/75))
- Fixed atomate2 version string.
- Disabled orbital projections in the electron-phonon workflow.

v0.0.5
------

This version removed Python 3.7 support following numpy and pymatgen.

New features:

- Base schema for molecule task documents ([@arosen93][arosen], [#54](https://github.com/materialsproject/atomate2/pull/54))

Bug fixes:

- Fix VASP relaxation using custodian "FULL_OPT" ([@arosen93][arosen], [#42](https://github.com/materialsproject/atomate2/pull/42))
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
- Add MP base sets ([@arosen93][arosen], [#27](https://github.com/materialsproject/atomate2/pull/27))
- Docs updates ([@arosen93][arosen], [#13](https://github.com/materialsproject/atomate2/pull/13) [#17](https://github.com/materialsproject/atomate2/pull/17))
- Options to strip band structure and DOS projects to reduce object sizes.
- Input sets now use generators to avoid serialization issues.
- Use smart efermi finding to remove errors with tetrahedron integration in VASP 6.
- Powerups can now work on `Maker` objects directly.

Bug fixes:

- Use PBEsol by default.
- Increase number of significant figures when writing POSCAR files.
- Remove unused INCAR settings ([@arosen93][arosen])
- Add missing LASPH flags on vdW functionals ([@arosen93][arosen], [#31](https://github.com/materialsproject/atomate2/pull/31))
- Use `NSW=0` in static calculations ([@arosen93][arosen], [#10](https://github.com/materialsproject/atomate2/pull/10))
- `LREAL = False` in static jobs by default ([@arosen93][arosen], [#23](https://github.com/materialsproject/atomate2/pull/23))
- Add missing functionals in output schema ([@arosen93][arosen], [#12](https://github.com/materialsproject/atomate2/pull/12))
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

[contributors]: <> (CONTRIBUTOR SECTION)
[nwinner]: https://github.com/nwinner
[jageo]: https://github.com/JaGeo
[zhuoying]: https://github.com/Zhuoying
[jmmshn]: https://github.com/jmmshn
[mkhorton]: https://github.com/mkhorton
[QuantumChemist]: https://github.com/QuantumChemist
[janosh]: https://github.com/janosh
[mjwen]: https://github.com/mjwen
[arosen]: https://github.com/arosen93
[rkingsbury]: https://github.com/rkingsbury
[naik-aakash]: https://github.com/naik-aakash
