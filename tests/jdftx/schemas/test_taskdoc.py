# test that TaskDoc is loaded with the right attributes

from atomate2.jdftx.schemas.task import TaskDoc

def test_taskdoc(jdftx_test_dir):
    """
    Test the JDFTx TaskDoc to verify that attributes are created properly.
    """
    inputs = TaskDoc.from_directory()