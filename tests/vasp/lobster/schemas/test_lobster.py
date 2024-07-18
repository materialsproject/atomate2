import os

import pytest
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.cohp import Cohp, CompleteCohp
from pymatgen.electronic_structure.dos import LobsterCompleteDos
from pymatgen.io.lobster import (
    Bandoverlaps,
    Charge,
    Grosspop,
    Icohplist,
    MadelungEnergies,
    SitePotential,
)

from atomate2.common.files import copy_files, gunzip_files
from atomate2.lobster.schemas import (
    CohpPlotData,
    CondensedBondingAnalysis,
    LobsterinModel,
    LobsteroutModel,
    LobsterTaskDocument,
    StrongestBonds,
    read_saved_json,
)


def test_lobster_task_document(lobster_test_dir):
    """
    Test the CCDDocument schema, this test needs to be placed here
    since we are using the VASP TaskDocuments for testing.
    """
    doc = LobsterTaskDocument.from_directory(
        dir_name=lobster_test_dir / "lobsteroutputs/mp-2534",
        save_cohp_plots=False,
        calc_quality_kwargs={"n_bins": 100, "potcar_symbols": ["Ga_d", "As"]},
        lobsterpy_kwargs={"cutoff_icohp": 0.10, "noise_cutoff": 0.1},
        save_cba_jsons=False,
        save_computational_data_jsons=False,
        add_coxxcar_to_task_document=True,
    )
    assert isinstance(doc.structure, Structure)
    assert isinstance(doc.lobsterout, LobsteroutModel)
    assert doc.lobsterout.charge_spilling[0] == pytest.approx(0.00989999, abs=1e-7)

    assert isinstance(doc.lobsterin, LobsterinModel)
    assert doc.lobsterin.cohpstartenergy == -5
    assert isinstance(doc.strongest_bonds, StrongestBonds)
    assert doc.strongest_bonds.strongest_bonds_icohp["As-Ga"] == pytest.approx(
        {"bond_strength": -4.32971, "length": 2.4899}
    )
    assert doc.strongest_bonds.strongest_bonds_icobi["As-Ga"] == pytest.approx(
        {"bond_strength": 0.82707, "length": 2.4899}
    )
    assert doc.strongest_bonds.strongest_bonds_icoop["As-Ga"] == pytest.approx(
        {"bond_strength": 0.31405, "length": 2.4899}
    )
    assert doc.strongest_bonds.which_bonds == "all"
    assert doc.strongest_bonds_cation_anion.strongest_bonds_icohp[
        "As-Ga"
    ] == pytest.approx({"bond_strength": -4.32971, "length": 2.4899})
    assert doc.strongest_bonds_cation_anion.strongest_bonds_icobi[
        "As-Ga"
    ] == pytest.approx({"bond_strength": 0.82707, "length": 2.4899})
    assert doc.strongest_bonds_cation_anion.strongest_bonds_icoop[
        "As-Ga"
    ] == pytest.approx({"bond_strength": 0.31405, "length": 2.4899})
    assert doc.strongest_bonds_cation_anion.which_bonds == "cation-anion"
    assert isinstance(doc.lobsterpy_data.cohp_plot_data.data["Ga1: 4 x As-Ga"], Cohp)
    assert doc.lobsterpy_data.which_bonds == "all"
    assert doc.lobsterpy_data_cation_anion.which_bonds == "cation-anion"
    assert doc.lobsterpy_data.number_of_considered_ions == 2
    assert isinstance(
        doc.lobsterpy_data_cation_anion.cohp_plot_data.data["Ga1: 4 x As-Ga"], Cohp
    )
    assert isinstance(doc.lobsterpy_text, str)
    assert isinstance(doc.lobsterpy_text_cation_anion, str)

    assert {*map(type, (doc.cohp_data, doc.cobi_data, doc.coop_data))} == {CompleteCohp}
    assert isinstance(doc.dos, LobsterCompleteDos)
    assert isinstance(doc.charges, Charge)
    assert isinstance(doc.madelung_energies, MadelungEnergies)
    assert isinstance(doc.site_potentials, SitePotential)
    assert isinstance(doc.band_overlaps, Bandoverlaps)
    assert {*map(type, (doc.icohp_list, doc.icobi_list, doc.icoop_list))} == {Icohplist}
    assert isinstance(doc.gross_populations, Grosspop)
    assert doc.chemsys == "As-Ga"

    doc2 = LobsterTaskDocument.from_directory(
        dir_name=lobster_test_dir / "lobsteroutputs/mp-754354",
        save_cohp_plots=False,
        calc_quality_kwargs={"n_bins": 100, "potcar_symbols": ["Ba_sv", "O", "F"]},
        save_cba_jsons=False,
        save_computational_data_jsons=False,
        add_coxxcar_to_task_document=True,
    )
    assert doc2.strongest_bonds.strongest_bonds_icohp["Ba-O"] == pytest.approx(
        {"bond_strength": -0.55689, "length": 2.57441}
    ), doc2.strongest_bonds.strongest_bonds_icohp["Ba-O"]
    assert doc2.strongest_bonds.strongest_bonds_icohp["Ba-F"] == pytest.approx(
        {"bond_strength": -0.44806, "length": 2.62797}
    ), doc2.strongest_bonds.strongest_bonds_icohp["Ba-F"]
    assert isinstance(doc2.charges, Charge)
    assert isinstance(doc2.madelung_energies, MadelungEnergies)
    assert isinstance(doc2.site_potentials, SitePotential)
    assert isinstance(doc2.band_overlaps, Bandoverlaps)
    assert {*map(type, (doc2.icohp_list, doc2.icobi_list, doc2.icoop_list))} == {
        Icohplist
    }
    assert isinstance(doc2.gross_populations, Grosspop)


def test_lobster_task_document_non_gzip(lobster_test_dir, tmp_path):
    """
    Test the CCDDocument schema, this test needs to be placed here
    since we are using the VASP TaskDocuments for testing.
    """

    # copy test files to temp path
    copy_files(src_dir=lobster_test_dir / "lobsteroutputs/mp-2534", dest_dir=tmp_path)
    # Unzip test files to check if schema still works
    gunzip_files(tmp_path)

    doc = LobsterTaskDocument.from_directory(
        dir_name=tmp_path,  # lobster_test_dir / "lobsteroutputs/mp-2534",
        save_cohp_plots=False,
        calc_quality_kwargs={"n_bins": 100, "potcar_symbols": ["Ga_d", "As"]},
        lobsterpy_kwargs={"cutoff_icohp": 0.10, "noise_cutoff": 0.01},
        save_cba_jsons=False,
        save_computational_data_jsons=False,
        add_coxxcar_to_task_document=True,
    )
    assert isinstance(doc.structure, Structure)
    assert isinstance(doc.lobsterout, LobsteroutModel)
    assert doc.lobsterout.charge_spilling[0] == pytest.approx(0.00989999, abs=1e-7)

    assert isinstance(doc.lobsterin, LobsterinModel)
    assert doc.lobsterin.cohpstartenergy == -5
    assert isinstance(doc.strongest_bonds, StrongestBonds)
    assert doc.strongest_bonds.strongest_bonds_icohp["As-Ga"] == pytest.approx(
        {"bond_strength": -4.32971, "length": 2.4899}
    )
    assert doc.strongest_bonds.strongest_bonds_icobi["As-Ga"] == pytest.approx(
        {"bond_strength": 0.82707, "length": 2.4899}
    )
    assert doc.strongest_bonds.strongest_bonds_icoop["As-Ga"] == pytest.approx(
        {"bond_strength": 0.31405, "length": 2.4899}
    )
    assert doc.strongest_bonds.which_bonds == "all"
    assert doc.strongest_bonds_cation_anion.strongest_bonds_icohp[
        "As-Ga"
    ] == pytest.approx({"bond_strength": -4.32971, "length": 2.4899})
    assert doc.strongest_bonds_cation_anion.strongest_bonds_icobi[
        "As-Ga"
    ] == pytest.approx({"bond_strength": 0.82707, "length": 2.4899})
    assert doc.strongest_bonds_cation_anion.strongest_bonds_icoop[
        "As-Ga"
    ] == pytest.approx({"bond_strength": 0.31405, "length": 2.4899})
    assert doc.strongest_bonds_cation_anion.which_bonds == "cation-anion"
    assert isinstance(doc.lobsterpy_data.cohp_plot_data.data["Ga1: 4 x As-Ga"], Cohp)
    assert doc.lobsterpy_data.which_bonds == "all"
    assert doc.lobsterpy_data_cation_anion.which_bonds == "cation-anion"
    assert doc.lobsterpy_data.number_of_considered_ions == 2
    assert isinstance(doc.lobsterpy_data_cation_anion.cohp_plot_data, CohpPlotData)
    assert isinstance(doc.lobsterpy_text, str)
    assert isinstance(doc.lobsterpy_text_cation_anion, str)

    assert {*map(type, (doc.cohp_data, doc.cobi_data, doc.coop_data))} == {CompleteCohp}
    assert isinstance(doc.dos, LobsterCompleteDos)
    assert isinstance(doc.charges, Charge)
    assert isinstance(doc.madelung_energies, MadelungEnergies)
    assert isinstance(doc.site_potentials, SitePotential)
    assert isinstance(doc.band_overlaps, Bandoverlaps)
    assert {*map(type, (doc.icohp_list, doc.icobi_list, doc.icoop_list))} == {Icohplist}
    assert isinstance(doc.gross_populations, Grosspop)

    assert doc.chemsys == "As-Ga"


def test_lobster_task_doc_saved_jsons(lobster_test_dir):
    """
    Test if jsons saved are valid
    """

    # Generate condensed bonding analysis (cba) json using lobstertaskdoc

    _ = LobsterTaskDocument.from_directory(
        dir_name=lobster_test_dir / "lobsteroutputs/mp-2534",
        save_cohp_plots=False,
        calc_quality_kwargs={"n_bins": 100, "potcar_symbols": ["Ga_d", "As"]},
        save_cba_jsons=True,
        add_coxxcar_to_task_document=False,
        save_computational_data_jsons=False,
    )

    expected_cba_keys_json = [
        "cation_anion_bonds",
        "all_bonds",
        "madelung_energies",
        "charges",
        "calc_quality_summary",
        "calc_quality_text",
        "dos",
        "lso_dos",
        "builder_meta",
    ]

    for cba_key in expected_cba_keys_json:
        # read data from saved json as pymatgen objects
        json_data = read_saved_json(
            filename=lobster_test_dir / "lobsteroutputs/mp-2534/cba.json.gz",
            pymatgen_objs=True,
            query=cba_key,
        )

        if "dos" in cba_key and json_data[cba_key]:
            assert isinstance(json_data[cba_key], LobsterCompleteDos)

        if cba_key in ("all_bonds", "cation_anion_bonds") and json_data[cba_key]:
            assert isinstance(
                json_data[cba_key]["lobsterpy_data"], CondensedBondingAnalysis
            )
            assert isinstance(
                json_data[cba_key]["lobsterpy_data"].cohp_plot_data, CohpPlotData
            )
            # assert isinstance(cohp_data, Cohp)

    # read cba saved jsons without converting it to non pymatgen objects (read as dict)

    for cba_key in expected_cba_keys_json:
        json_data = read_saved_json(
            filename=lobster_test_dir / "lobsteroutputs/mp-2534/cba.json.gz",
            pymatgen_objs=False,
            query=cba_key,
        )
        if "dos" in cba_key and json_data[cba_key]:
            assert isinstance(json_data[cba_key], dict)

        if cba_key in ("all_bonds", "cation_anion_bonds") and json_data[cba_key]:
            for cohp_data in json_data[cba_key]["lobsterpy_data"]["cohp_plot_data"][
                "data"
            ].values():
                assert isinstance(cohp_data, dict)

    # delete the cba json after the test
    os.remove(lobster_test_dir / "lobsteroutputs/mp-2534/cba.json.gz")

    # Generate computational JSON data from LobsterTaskDocument
    _ = LobsterTaskDocument.from_directory(
        dir_name=lobster_test_dir / "lobsteroutputs/mp-754354",
        save_cohp_plots=False,
        calc_quality_kwargs={"n_bins": 100, "potcar_symbols": ["Ba_sv", "O", "F"]},
        save_cba_jsons=False,
        add_coxxcar_to_task_document=False,
        save_computational_data_jsons=True,
    )

    expected_computational_data_keys_json = [
        "builder_meta",
        "structure",
        "charges",
        "lobsterout",
        "lobsterin",
        "lobsterpy_data",
        "lobsterpy_text",
        "calc_quality_summary",
        "calc_quality_text",
        "strongest_bonds",
        "lobsterpy_data_cation_anion",
        "lobsterpy_text_cation_anion",
        "strongest_bonds_cation_anion",
        "dos",
        "lso_dos",
        "madelung_energies",
        "site_potentials",
        "gross_populations",
        "band_overlaps",
        "cohp_data",
        "coop_data",
        "cobi_data",
        "icobi_list",
        "icoop_list",
        "icohp_list",
    ]

    # Read the data from saved computational data json as pymatgen objects
    for task_doc_key in expected_computational_data_keys_json:
        json_data = read_saved_json(
            filename=lobster_test_dir
            / "lobsteroutputs/mp-754354/computational_data.json.gz",
            pymatgen_objs=True,
            query=task_doc_key,
        )
        if "dos" in task_doc_key and json_data[task_doc_key]:
            assert isinstance(json_data[task_doc_key], LobsterCompleteDos)

        if "lobsterpy_data" in task_doc_key and json_data[task_doc_key]:
            assert isinstance(json_data[task_doc_key], CondensedBondingAnalysis)
            for cohp_data in json_data[task_doc_key].cohp_plot_data.data.values():
                assert isinstance(cohp_data, Cohp)

        if (
            task_doc_key in {"cohp_data", "cobi_data", "coop_data"}
            and json_data[task_doc_key]
        ):
            assert isinstance(json_data[task_doc_key], CompleteCohp)

    # delete the computational data json after the test
    os.remove(lobster_test_dir / "lobsteroutputs/mp-754354/computational_data.json.gz")
