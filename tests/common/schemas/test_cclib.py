import gzip
import os
import shutil

import pytest

try:
    import cclib
except ImportError:
    cclib = None


@pytest.mark.skipif(cclib is None, reason="requires cclib to be installed")
def test_cclib_taskdoc(test_dir):
    from monty.json import MontyDecoder, jsanitize

    from atomate2.common.schemas.cclib import TaskDocument

    p = test_dir / "schemas"

    # Plain parsing of task doc. We do not check all cclib entries
    # because they will evolve over time. We only check the ones we have
    # added and some important ones.
    doc = TaskDocument.from_logfile(p, ".log.gz").dict()
    assert doc["energy"] == pytest.approx(-4091.763)
    assert doc["natoms"] == 2
    assert doc["charge"] == 0
    assert doc["spin_multiplicity"] == 3
    assert doc["nelectrons"] == 16
    assert "schemas" in doc["dir_name"]
    assert "gau_testopt.log.gz" in doc["logfile"]
    assert doc.get("attributes", None) is not None
    assert doc.get("metadata", None) is not None
    assert doc["metadata"]["success"] is True
    assert doc["attributes"]["molecule_initial"][0].coords == pytest.approx([0, 0, 0])
    assert doc["molecule"][0].coords == pytest.approx([0.397382, 0.0, 0.0])
    assert doc["last_updated"] is not None
    assert doc["attributes"]["homo_energies"] == pytest.approx(
        [-7.054007346511501, -11.618445074798501]
    )
    assert doc["attributes"]["lumo_energies"] == pytest.approx(
        [4.2384453353880005, -3.9423854660440005]
    )
    assert doc["attributes"]["homo_lumo_gaps"] == pytest.approx(
        [11.292452681899501, 7.6760596087545006]
    )
    assert doc["attributes"]["min_homo_lumo_gap"] == pytest.approx(7.6760596087545006)

    # Now we will try two possible extensions, but we will make sure that
    # it fails because the newest log file (.txt) is not valid
    with open(p / "test.txt", "w") as f:
        f.write("I am a dummy log file")
    with pytest.raises(Exception) as e:
        doc = TaskDocument.from_logfile(p, [".log", ".txt"]).dict()
    os.remove(p / "test.txt")
    assert "Could not parse" in str(e.value)

    # Test a population analysis
    doc = TaskDocument.from_logfile(p, ".out", analysis="MBO").dict()
    assert doc["attributes"]["mbo"] is not None

    # Let's try with two analysis (also check case-insensitivity)
    doc = TaskDocument.from_logfile(p, ".out", analysis=["mbo", "density"]).dict()
    assert doc["attributes"]["mbo"] is not None
    assert doc["attributes"]["density"] is not None

    # Test a population analysis that will fail
    doc = TaskDocument.from_logfile(p, ".log", analysis="MBO").dict()
    assert doc["attributes"]["mbo"] is None

    # Let's try a volumetric analysis
    # We'll gunzip the .cube.gz file because cclib can't read cube.gz files yet.
    # Can remove the gzip part when https://github.com/cclib/cclib/issues/108 is closed.
    with gzip.open(p / "psi_test.cube.gz", "r") as f_in, open(
        p / "psi_test.cube", "wb"
    ) as f_out:
        shutil.copyfileobj(f_in, f_out)
    doc = TaskDocument.from_logfile(p, ".out", analysis=["Bader"]).dict()
    os.remove(p / "psi_test.cube")
    assert doc["attributes"]["bader"] is not None

    # Make sure storing the trajectory works
    doc = TaskDocument.from_logfile(p, ".log", store_trajectory=True).dict()
    assert len(doc["attributes"]["trajectory"]) == 7
    assert doc["attributes"]["trajectory"][0] == doc["attributes"]["molecule_initial"]
    assert doc["attributes"]["trajectory"][-1] == doc["molecule"]

    # Make sure additional fields can be stored
    doc = TaskDocument.from_logfile(p, ".log", additional_fields={"test": "hi"})
    assert doc.dict()["test"] == "hi"

    # test document can be jsanitized
    d = jsanitize(doc, enum_values=True)

    # and decoded
    MontyDecoder().process_decoded(d)
