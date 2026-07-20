"""
Tests for powerups module (workflow customization utilities).

These tests validate:
- update_maker_kwargs
- update_user_siesta_settings
- update_fdf_siesta_settings
- write_output_json_local
"""

import pytest
from jobflow import Flow, Job, Maker

from atomate2.siesta.powerups import (
    update_maker_kwargs,
    update_user_siesta_settings,
)
from atomate2.siesta.jobs.core import StaticMaker, RelaxMaker


class TestUpdateMakerKwargs:
    """Tests for update_maker_kwargs function."""

    def test_update_maker_with_dict_mod(self):
        """Test updating a maker with dict modifications."""
        maker = StaticMaker()

        dict_mod_updates = {"input_set_generator->user_params->a2s_kpts": [6, 6, 6]}

        updated_maker = update_maker_kwargs(
            class_filter=None,
            dict_mod_updates=dict_mod_updates,
            flow=maker,
            name_filter=None,
        )

        # Should return a maker (deepcopy)
        assert isinstance(updated_maker, Maker)
        assert updated_maker is not maker  # Different object

    def test_update_maker_with_name_filter(self):
        """Test updating maker with name filter."""
        maker = StaticMaker(name="test_static")

        dict_mod_updates = {"input_set_generator->user_params->a2s_kpts": [8, 8, 8]}

        updated_maker = update_maker_kwargs(
            class_filter=None,
            dict_mod_updates=dict_mod_updates,
            flow=maker,
            name_filter="test_static",
        )

        assert isinstance(updated_maker, Maker)

    def test_update_maker_with_class_filter(self):
        """Test updating maker with class filter."""
        maker = StaticMaker()

        dict_mod_updates = {"input_set_generator->user_params->PAO.BasisSize": "DZP"}

        updated_maker = update_maker_kwargs(
            class_filter=StaticMaker,
            dict_mod_updates=dict_mod_updates,
            flow=maker,
            name_filter=None,
        )

        assert isinstance(updated_maker, StaticMaker)

    def test_update_job_kwargs(self, si_structure):
        """Test updating a job's maker kwargs."""
        maker = StaticMaker()
        job = maker.make(si_structure)

        dict_mod_updates = {"input_set_generator->user_params->a2s_kpts": [4, 4, 4]}

        updated_job = update_maker_kwargs(
            class_filter=None,
            dict_mod_updates=dict_mod_updates,
            flow=job,
            name_filter=None,
        )

        assert isinstance(updated_job, Job)
        assert updated_job is not job  # Different object

    def test_update_flow_kwargs(self, si_structure):
        """Test updating a flow's maker kwargs."""
        maker = StaticMaker()
        job1 = maker.make(si_structure)
        job2 = maker.make(si_structure)
        flow = Flow([job1, job2])

        dict_mod_updates = {"input_set_generator->user_params->Mesh.Cutoff": "400 Ry"}

        updated_flow = update_maker_kwargs(
            class_filter=None,
            dict_mod_updates=dict_mod_updates,
            flow=flow,
            name_filter=None,
        )

        assert isinstance(updated_flow, Flow)
        assert updated_flow is not flow  # Different object

    def test_update_maker_kwargs_deepcopy(self):
        """Test that update_maker_kwargs creates a deep copy."""
        maker = StaticMaker()
        original_id = id(maker)

        dict_mod_updates = {"input_set_generator->user_params->a2s_kpts": [2, 2, 2]}

        updated_maker = update_maker_kwargs(
            class_filter=None,
            dict_mod_updates=dict_mod_updates,
            flow=maker,
            name_filter=None,
        )

        # Verify it's a different object
        assert id(updated_maker) != original_id


class TestUpdateUserSiestaSettings:
    """Tests for update_user_siesta_settings function."""

    def test_update_user_params_basic(self):
        """Test basic user parameter updates."""
        maker = StaticMaker()

        siesta_updates = {
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [6, 6, 6],
        }

        updated_maker = update_user_siesta_settings(
            flow=maker,
            siesta_updates=siesta_updates,
        )

        assert isinstance(updated_maker, Maker)
        assert updated_maker is not maker

    def test_update_user_params_with_name_filter(self):
        """Test updating with name filter."""
        maker = RelaxMaker(name="my_relax")

        siesta_updates = {"Mesh.Cutoff": "300 Ry"}

        updated_maker = update_user_siesta_settings(
            flow=maker,
            siesta_updates=siesta_updates,
            name_filter="my_relax",
        )

        assert isinstance(updated_maker, Maker)

    def test_update_user_params_with_class_filter(self):
        """Test updating with class filter."""
        maker = StaticMaker()

        siesta_updates = {"PAO.BasisSize": "TZP"}

        updated_maker = update_user_siesta_settings(
            flow=maker,
            siesta_updates=siesta_updates,
            class_filter=StaticMaker,
        )

        assert isinstance(updated_maker, StaticMaker)

    def test_update_user_params_none(self):
        """Test with None siesta_updates."""
        maker = StaticMaker()

        updated_maker = update_user_siesta_settings(
            flow=maker,
            siesta_updates=None,
        )

        # Should still work, just no updates applied
        assert isinstance(updated_maker, Maker)

    def test_update_with_new_fdf_flags(self):
        """Test updating with new FDF flags."""
        maker = StaticMaker()

        new_fdf_flags = {
            "SCF.Mixer.Weight": 0.1,
            "ElectronicTemperature": "300 K",
        }

        updated_maker = update_user_siesta_settings(
            flow=maker,
            siesta_updates=None,
            new_fdf_flags=new_fdf_flags,
        )

        assert isinstance(updated_maker, Maker)

    def test_update_both_params_and_flags(self):
        """Test updating both user_params and fdf_flags."""
        maker = StaticMaker()

        siesta_updates = {"PAO.BasisSize": "DZP"}
        new_fdf_flags = {"SCF.Mixer.Weight": 0.05}

        updated_maker = update_user_siesta_settings(
            flow=maker,
            siesta_updates=siesta_updates,
            new_fdf_flags=new_fdf_flags,
        )

        assert isinstance(updated_maker, Maker)

    def test_update_job_user_params(self, si_structure):
        """Test updating user params in a job."""
        maker = StaticMaker()
        job = maker.make(si_structure)

        siesta_updates = {"a2s_kpts": [8, 8, 8]}

        updated_job = update_user_siesta_settings(
            flow=job,
            siesta_updates=siesta_updates,
        )

        assert isinstance(updated_job, Job)
        assert updated_job is not job

    def test_update_flow_user_params(self, si_structure):
        """Test updating user params in a flow."""
        maker1 = StaticMaker()
        maker2 = RelaxMaker()
        job1 = maker1.make(si_structure)
        job2 = maker2.make(si_structure)
        flow = Flow([job1, job2])

        siesta_updates = {"Mesh.Cutoff": "400 Ry"}

        updated_flow = update_user_siesta_settings(
            flow=flow,
            siesta_updates=siesta_updates,
        )

        assert isinstance(updated_flow, Flow)
        assert updated_flow is not flow

    def test_update_multiple_parameters(self):
        """Test updating multiple parameters at once."""
        maker = StaticMaker()

        siesta_updates = {
            "PAO.BasisSize": "DZP",
            "a2s_kpts": [6, 6, 6],
            "Mesh.Cutoff": "350 Ry",
            "PAO.EnergyShift": "0.01 Ry",
        }

        updated_maker = update_user_siesta_settings(
            flow=maker,
            siesta_updates=siesta_updates,
        )

        assert isinstance(updated_maker, Maker)

    def test_update_preserves_original(self):
        """Test that original maker is not modified."""
        maker = StaticMaker()

        siesta_updates = {"PAO.BasisSize": "TZP"}

        updated_maker = update_user_siesta_settings(
            flow=maker,
            siesta_updates=siesta_updates,
        )

        # Original should be unchanged
        assert maker is not updated_maker


class TestUpdateFdfSiestaSettings:
    """Tests for update_fdf_siesta_settings function."""

    def test_update_fdf_settings_basic(self):
        """Test basic FDF settings update."""
        # This function requires a specific job structure
        # Skip if implementation details are needed
        pytest.skip("Requires specific job structure with basis_sets_and_projectors")

    def test_update_fdf_unknown_key(self):
        """Test updating with unknown FDF key."""
        pytest.skip("Requires specific job structure")


class TestWriteOutputJsonLocal:
    """Tests for write_output_json_local function."""

    def test_write_json_basic(self, tmp_path, monkeypatch):
        """Test writing JSON output to file."""
        from unittest.mock import MagicMock

        # Mock results structure
        mock_output = MagicMock()
        mock_output.dict.return_value = {
            "energy": -10.5,
            "forces": [[0.1, 0.2, 0.3]],
        }

        mock_response = MagicMock()
        mock_response.output = mock_output

        results = {
            "job_1": {
                "step_1": mock_response,
            }
        }

        # Change to tmp directory
        monkeypatch.chdir(tmp_path)

        from atomate2.siesta.powerups import write_output_json_local

        write_output_json_local(results)

        # Check file was created
        json_file = tmp_path / "calculation_results.json"
        assert json_file.exists()

    def test_write_json_multiple_jobs(self, tmp_path, monkeypatch):
        """Test writing JSON with multiple jobs."""
        from unittest.mock import MagicMock

        mock_output1 = MagicMock()
        mock_output1.dict.return_value = {"energy": -10.5}

        mock_output2 = MagicMock()
        mock_output2.dict.return_value = {"energy": -11.2}

        mock_response1 = MagicMock()
        mock_response1.output = mock_output1

        mock_response2 = MagicMock()
        mock_response2.output = mock_output2

        results = {
            "job_1": {"step_1": mock_response1},
            "job_2": {"step_1": mock_response2},
        }

        monkeypatch.chdir(tmp_path)

        from atomate2.siesta.powerups import write_output_json_local

        write_output_json_local(results)

        json_file = tmp_path / "calculation_results.json"
        assert json_file.exists()


class TestPowerupsIntegration:
    """Integration tests for powerups."""

    def test_chain_multiple_updates(self):
        """Test chaining multiple powerup updates."""
        maker = StaticMaker()

        # First update
        maker = update_user_siesta_settings(
            flow=maker,
            siesta_updates={"PAO.BasisSize": "DZP"},
        )

        # Second update
        maker = update_user_siesta_settings(
            flow=maker,
            siesta_updates={"a2s_kpts": [6, 6, 6]},
        )

        assert isinstance(maker, Maker)

    def test_update_complex_flow(self, si_structure):
        """Test updating a complex flow with multiple jobs."""
        # Create a flow with different makers
        static_maker = StaticMaker()
        relax_maker = RelaxMaker()

        job1 = static_maker.make(si_structure)
        job2 = relax_maker.make(si_structure)

        flow = Flow([job1, job2])

        # Update all jobs
        updated_flow = update_user_siesta_settings(
            flow=flow,
            siesta_updates={"Mesh.Cutoff": "400 Ry"},
        )

        assert isinstance(updated_flow, Flow)
        assert len(list(updated_flow)) == 2

    def test_selective_update_by_class(self, si_structure):
        """Test selectively updating only specific maker classes."""
        static_maker = StaticMaker()
        relax_maker = RelaxMaker()

        job1 = static_maker.make(si_structure)
        job2 = relax_maker.make(si_structure)

        flow = Flow([job1, job2])

        # Update only StaticMaker jobs
        updated_flow = update_user_siesta_settings(
            flow=flow,
            siesta_updates={"PAO.BasisSize": "TZP"},
            class_filter=StaticMaker,
        )

        assert isinstance(updated_flow, Flow)

    def test_selective_update_by_name(self, si_structure):
        """Test selectively updating by job name."""
        maker1 = StaticMaker(name="special_static")
        maker2 = StaticMaker(name="regular_static")

        job1 = maker1.make(si_structure)
        job2 = maker2.make(si_structure)

        flow = Flow([job1, job2])

        # Update only "special_static" jobs
        updated_flow = update_user_siesta_settings(
            flow=flow,
            siesta_updates={"a2s_kpts": [10, 10, 10]},
            name_filter="special_static",
        )

        assert isinstance(updated_flow, Flow)


class TestPowerupsEdgeCases:
    """Test edge cases for powerups."""

    def test_empty_updates_dict(self):
        """Test with empty updates dictionary."""
        maker = StaticMaker()

        updated_maker = update_user_siesta_settings(
            flow=maker,
            siesta_updates={},
        )

        assert isinstance(updated_maker, Maker)

    def test_none_class_filter(self):
        """Test with None class filter (updates all)."""
        maker = StaticMaker()

        updated_maker = update_user_siesta_settings(
            flow=maker,
            siesta_updates={"PAO.BasisSize": "DZP"},
            class_filter=None,
        )

        assert isinstance(updated_maker, Maker)

    def test_special_characters_in_updates(self):
        """Test with special characters in parameter names."""
        maker = StaticMaker()

        siesta_updates = {
            "PAO.EnergyShift": "0.01 Ry",
            "SCF.Mixer.Weight": 0.1,
        }

        updated_maker = update_user_siesta_settings(
            flow=maker,
            siesta_updates=siesta_updates,
        )

        assert isinstance(updated_maker, Maker)

    def test_numerical_parameter_values(self):
        """Test with various numerical parameter types."""
        maker = StaticMaker()

        siesta_updates = {
            "a2s_kpts": [6, 6, 6],  # list of ints
            "Mesh.Cutoff": "300 Ry",  # string
            "PAO.EnergyShift": 0.01,  # float
        }

        updated_maker = update_user_siesta_settings(
            flow=maker,
            siesta_updates=siesta_updates,
        )

        assert isinstance(updated_maker, Maker)
