def test_LobsterTaskDocument(lobster_test_dir):
    """
    Test the CCDDocument schema, this test needs to be placed here
    since we are using the VASP TaskDocuments for testing.
    """

    import numpy as np
    from pymatgen.core.structure import Structure
    from pymatgen.electronic_structure.cohp import Cohp, CompleteCohp
    from pymatgen.electronic_structure.dos import LobsterCompleteDos

    from atomate2.lobster.schemas import (
        LobsterinModel,
        LobsteroutModel,
        LobsterTaskDocument,
        StrongestBonds,
    )

    doc = LobsterTaskDocument.from_directory(
        dir_name=lobster_test_dir / "lobsteroutputs/mp-2534", save_cohp_plots=False
    )
    assert isinstance(doc.structure, Structure)
    assert isinstance(doc.lobsterout, LobsteroutModel)
    assert np.isclose(doc.lobsterout.charge_spilling[0], 0.009899999999999999)

    assert isinstance(doc.lobsterin, LobsterinModel)
    assert np.isclose(doc.lobsterin.cohpstartenergy, -5)
    assert isinstance(doc.strongest_bonds_icohp, StrongestBonds)
    assert np.isclose(
        doc.strongest_bonds_icohp.strongest_bonds["As-Ga"]["ICOHP"], -4.32971
    )
    assert np.isclose(
        doc.strongest_bonds_icobi.strongest_bonds["As-Ga"]["ICOBI"], 0.8269299999999999
    )
    assert np.isclose(
        doc.strongest_bonds_icoop.strongest_bonds["As-Ga"]["ICOOP"], 0.31381
    )
    assert np.isclose(
        doc.strongest_bonds_icohp.strongest_bonds["As-Ga"]["length"], 2.4899
    )
    assert np.isclose(
        doc.strongest_bonds_icobi.strongest_bonds["As-Ga"]["length"], 2.4899
    )
    assert np.isclose(
        doc.strongest_bonds_icoop.strongest_bonds["As-Ga"]["length"], 2.4899
    )
    assert doc.strongest_bonds_icoop.which_bonds == "all"
    assert doc.strongest_bonds_icohp.which_bonds == "all"
    assert doc.strongest_bonds_icobi.which_bonds == "all"
    assert np.isclose(
        doc.strongest_bonds_icohp_cation_anion.strongest_bonds["As-Ga"]["ICOHP"],
        -4.32971,
    )
    assert np.isclose(
        doc.strongest_bonds_icobi_cation_anion.strongest_bonds["As-Ga"]["ICOBI"],
        0.8269299999999999,
    )
    assert np.isclose(
        doc.strongest_bonds_icoop_cation_anion.strongest_bonds["As-Ga"]["ICOOP"],
        0.31381,
    )
    assert np.isclose(
        doc.strongest_bonds_icohp_cation_anion.strongest_bonds["As-Ga"]["length"],
        2.4899,
    )
    assert np.isclose(
        doc.strongest_bonds_icobi_cation_anion.strongest_bonds["As-Ga"]["length"],
        2.4899,
    )
    assert np.isclose(
        doc.strongest_bonds_icoop_cation_anion.strongest_bonds["As-Ga"]["length"],
        2.4899,
    )
    assert doc.strongest_bonds_icoop_cation_anion.which_bonds == "cation-anion"
    assert doc.strongest_bonds_icohp_cation_anion.which_bonds == "cation-anion"
    assert doc.strongest_bonds_icobi_cation_anion.which_bonds == "cation-anion"
    assert isinstance(doc.lobsterpy_data.cohp_plot_data["Ga1: 4 x As-Ga"], Cohp)
    assert doc.lobsterpy_data.which_bonds == "all"
    assert doc.lobsterpy_data_cation_anion.which_bonds == "cation-anion"
    assert doc.lobsterpy_data.number_of_considered_ions == 2
    assert isinstance(
        doc.lobsterpy_data_cation_anion.cohp_plot_data["Ga1: 4 x As-Ga"], Cohp
    )
    assert isinstance(doc.lobsterpy_text, str)
    assert isinstance(doc.lobsterpy_text_cation_anion, str)

    assert isinstance(doc.cohp_data, CompleteCohp)
    assert isinstance(doc.cobi_data, CompleteCohp)
    assert isinstance(doc.coop_data, CompleteCohp)
    assert isinstance(doc.dos, LobsterCompleteDos)
    assert np.isclose(doc.madelung_energies["Mulliken"], -0.68)
    assert np.allclose(
        doc.charges["Mulliken"],
        [0.13, 0.13, 0.13, 0.13, -0.13, -0.13, -0.13, -0.13],
        rtol=1e-2,
    )

    doc2 = LobsterTaskDocument.from_directory(
        dir_name=lobster_test_dir / "lobsteroutputs/mp-754354", save_cohp_plots=False
    )
    assert np.isclose(
        doc2.strongest_bonds_icohp.strongest_bonds["Ba-O"]["ICOHP"], -0.55689
    )
    assert np.isclose(
        doc2.strongest_bonds_icohp.strongest_bonds["Ba-F"]["ICOHP"], -0.44806
    )
