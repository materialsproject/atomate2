from __future__ import annotations
import unittest

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable
from unittest.mock import patch, MagicMock
from pathlib import Path
from pymatgen.core import Structure
from atomate2.jdftx.sets.base import JdftxInputGenerator
from atomate2.jdftx.jobs.base import BaseJdftxMaker 
import stackprinter
stackprinter.set_excepthook(style='darkbg2')

class TestBaseJdftxMaker(unittest.TestCase):
    def setUp(self):
        # Create a simple cubic structure for testing
        lattice = [[3.8401979337, 0.00, 0.00],
                   [0.00, 3.8401979337, 0.00],
                   [0.00, 0.00, 3.8401979337]]
        species = ["Sr", "Ti", "O", "O", "O"]
        coords = [[0.00, 0.00, 0.00],
                  [0.50, 0.50, 0.50],
                  [0.50, 0.50, 0.00],
                  [0.50, 0.00, 0.50],
                  [0.00, 0.50, 0.50]]
        self.structure = Structure(lattice, species, coords)
        
        # Initialize the BaseJdftxMaker
        self.maker = BaseJdftxMaker()

    @patch('atomate2.jdftx.jobs.base.write_jdftx_input_set')
    @patch('atomate2.jdftx.jobs.base.run_jdftx')
    @patch('atomate2.jdftx.jobs.base.Path.cwd')
    @patch('atomate2.jdftx.jobs.base.Path.glob')
    @patch('atomate2.jdftx.jobs.base.Path.is_file')

    def test_make(self, mock_is_file, mock_glob, mock_cwd, mock_run_jdftx, mock_write_input):
        print("\nStarting test_make")
        
        # Set up mocks
        mock_cwd.return_value = Path('/fake/path')
        mock_files = [Path('/fake/path/file1.txt'), Path('/fake/path/file2.txt')]
        mock_glob.return_value = mock_files
        mock_is_file.return_value = True  # Assume all paths are files

        
        print(f"Mock setup complete. mock_files: {mock_files}")


        # Run the make method
        print("Before make() call")
        response = self.maker.make(self.structure)
        print("After make() call")

        print(f"\nAssertions:")
        print("mock_write_input called:", mock_write_input.call_count, "times")
        print("mock_run_jdftx called:", mock_run_jdftx.call_count, "times")
        print("mock_cwd called:", mock_cwd.call_count, "times")
        print("mock_glob called:", mock_glob.call_count, "times")
        print(f"mock_is_file called: {mock_is_file.call_count} times")


        # Check that write_jdftx_input_set was called
        mock_write_input.assert_called_once()

        # Check that run_jdftx was called
        mock_run_jdftx.assert_called_once()
        mock_cwd.assert_called_once()
        mock_glob.assert_called_once_with('*')

        # Check the Response object
        self.assertEqual(response.output['directory'], '/fake/path')
      #  self.assertEqual(response.output['files'], ['/fake/path/file1.txt', '/fake/path/file2.txt'])
        self.assertEqual(response.output['files'], [str(f) for f in mock_files])
        self.assertEqual(response.stored_data['job_type'], 'JDFTx')
        self.assertEqual(response.stored_data['status'], 'completed')

        print("Repsonse:", response)

if __name__ == '__main__':
    unittest.main()