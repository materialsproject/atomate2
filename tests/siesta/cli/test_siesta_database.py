"""Tests for siesta_database CLI module."""

from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import pytest
from click.testing import CliRunner

from atomate2.siesta.cli.database.siesta_database import (
    cli,
    test_mongodb_connection as mongodb_connection_helper,
    test_maggma_store as maggma_store_helper,
)


@pytest.fixture
def runner():
    """Create Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_mongo_client():
    """Mock MongoDB client."""
    with patch("atomate2.siesta.cli.database.siesta_database.MongoClient") as mock:
        client = MagicMock()
        mock.return_value = client

        # Mock admin.command for ping
        client.admin.command.return_value = "pong"

        # Mock database and collection
        db = MagicMock()
        client.__getitem__.return_value = db
        coll = MagicMock()
        db.__getitem__.return_value = coll

        yield mock, client, db, coll


class TestHelperFunctions:
    """Test helper functions."""

    def test_helper_functions_exist(self):
        """Test that helper functions can be imported."""
        # Just verify they exist and are callable
        assert callable(mongodb_connection_helper)
        assert callable(maggma_store_helper)


class TestTestCommand:
    """Test 'test' CLI command."""

    @patch("atomate2.siesta.cli.database.siesta_database.test_maggma_store")
    @patch("atomate2.siesta.cli.database.siesta_database.test_mongodb_connection")
    def test_successful_connection(self, mock_mongo_test, mock_maggma_test, runner):
        """Test successful connection."""
        # Mock successful connections
        client = MagicMock()
        db = MagicMock()
        coll = MagicMock()

        mock_mongo_test.return_value = (True, client, db, coll, None)
        mock_maggma_test.return_value = (True, MagicMock(), None)

        # Mock database stats
        coll.count_documents.return_value = 5
        db.command.return_value = {
            "collections": 2,
            "dataSize": 1024 * 1024,
            "storageSize": 2 * 1024 * 1024,
            "indexes": 3,
        }

        result = runner.invoke(cli, ["test"])

        assert result.exit_code == 0
        assert "PyMongo connection successful" in result.output
        assert "Maggma store connection successful" in result.output

    @patch("atomate2.siesta.cli.database.siesta_database.test_mongodb_connection")
    def test_connection_failed(self, mock_mongo_test, runner):
        """Test failed connection."""
        # Mock failed connection
        mock_mongo_test.return_value = (False, None, None, None, "Connection refused")

        result = runner.invoke(cli, ["test"])

        assert result.exit_code == 0
        assert "PyMongo connection failed" in result.output

    @patch("atomate2.siesta.cli.database.siesta_database.test_maggma_store")
    @patch("atomate2.siesta.cli.database.siesta_database.test_mongodb_connection")
    def test_empty_database(self, mock_mongo_test, mock_maggma_test, runner):
        """Test with empty database."""
        client = MagicMock()
        db = MagicMock()
        coll = MagicMock()

        mock_mongo_test.return_value = (True, client, db, coll, None)
        mock_maggma_test.return_value = (True, MagicMock(), None)

        # Mock empty database
        coll.count_documents.return_value = 0
        db.command.return_value = {"collections": 0}

        result = runner.invoke(cli, ["test"])

        assert result.exit_code == 0
        assert (
            "Database 'atomate2siesta' is NOT set up" in result.output
            or "not set up" in result.output.lower()
        )


class TestListCommand:
    """Test 'list' CLI command."""

    @patch("atomate2.siesta.cli.database.siesta_database.test_mongodb_connection")
    def test_list_documents(self, mock_mongo_test, runner):
        """Test listing documents."""
        client = MagicMock()
        db = MagicMock()
        coll = MagicMock()

        mock_mongo_test.return_value = (True, client, db, coll, None)

        # Mock document count
        coll.count_documents.return_value = 2

        # Mock documents
        mock_docs = [
            {
                "uuid": "123e4567-e89b-12d3-a456-426614174000",
                "output": {
                    "formula_pretty": "Si2",
                    "state": "successful",
                    "output": {"energy": -10.5},
                },
                "name": "relax",
            },
            {
                "uuid": "223e4567-e89b-12d3-a456-426614174001",
                "output": {
                    "formula": "Fe2O3",
                    "state": "failed",
                    "output": {},
                },
                "name": "static",
            },
        ]

        # Mock find().sort().limit()
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = iter(mock_docs)
        coll.find.return_value.sort.return_value.limit.return_value = mock_cursor

        result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        assert "Found 2 documents" in result.output

    @patch("atomate2.siesta.cli.database.siesta_database.test_mongodb_connection")
    def test_list_empty_collection(self, mock_mongo_test, runner):
        """Test listing with no documents."""
        client = MagicMock()
        db = MagicMock()
        coll = MagicMock()

        mock_mongo_test.return_value = (True, client, db, coll, None)
        coll.count_documents.return_value = 0

        result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        assert "No documents found" in result.output


class TestQueryCommand:
    """Test 'query' CLI command."""

    @patch("atomate2.siesta.cli.database.siesta_database.test_mongodb_connection")
    def test_query_formula(self, mock_mongo_test, runner):
        """Test querying by formula."""
        client = MagicMock()
        db = MagicMock()
        coll = MagicMock()

        mock_mongo_test.return_value = (True, client, db, coll, None)

        # Mock found documents
        mock_docs = [
            {
                "uuid": "123",
                "output": {
                    "formula_pretty": "Si",
                    "state": "successful",
                    "output": {"energy": -10.5},
                    "input": {"parameters": {"kpts": [4, 4, 4]}},
                },
                "name": "relax",
            }
        ]

        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = iter(mock_docs)
        coll.find.return_value = mock_cursor

        result = runner.invoke(cli, ["query", "--formula", "Si"])

        assert result.exit_code == 0
        assert "Found 1 matching documents" in result.output

    @patch("atomate2.siesta.cli.database.siesta_database.test_mongodb_connection")
    def test_query_no_results(self, mock_mongo_test, runner):
        """Test query with no matching results."""
        client = MagicMock()
        db = MagicMock()
        coll = MagicMock()

        mock_mongo_test.return_value = (True, client, db, coll, None)

        # Mock no documents found
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = iter([])
        coll.find.return_value = mock_cursor

        result = runner.invoke(cli, ["query", "--formula", "Unobtainium"])

        assert result.exit_code == 0
        assert "No matching documents found" in result.output


class TestStatsCommand:
    """Test 'stats' CLI command."""

    @patch("atomate2.siesta.cli.database.siesta_database.test_mongodb_connection")
    def test_stats_display(self, mock_mongo_test, runner):
        """Test displaying statistics."""
        client = MagicMock()
        db = MagicMock()
        coll = MagicMock()

        mock_mongo_test.return_value = (True, client, db, coll, None)

        # Mock collections
        db.list_collection_names.return_value = ["tasks", "task_data"]

        # Mock collection stats
        db.command.side_effect = [
            {"count": 10, "size": 1024 * 1024, "avgObjSize": 100 * 1024},
            {"count": 5, "size": 512 * 1024, "avgObjSize": 100 * 1024},
            {
                "dataSize": 1.5 * 1024 * 1024,
                "storageSize": 2 * 1024 * 1024,
                "indexes": 5,
                "indexSize": 0.5 * 1024 * 1024,
            },
        ]

        result = runner.invoke(cli, ["stats"])

        assert result.exit_code == 0
        assert "Collection Statistics" in result.output


class TestClearCommand:
    """Test 'clear' CLI command."""

    @patch("atomate2.siesta.cli.database.siesta_database.test_mongodb_connection")
    def test_clear_documents_with_force(self, mock_mongo_test, runner):
        """Test clearing documents with --force flag."""
        client = MagicMock()
        db = MagicMock()
        coll = MagicMock()

        mock_mongo_test.return_value = (True, client, db, coll, None)
        coll.count_documents.return_value = 5

        # Mock delete_many result
        delete_result = MagicMock()
        delete_result.deleted_count = 5
        coll.delete_many.return_value = delete_result

        result = runner.invoke(cli, ["clear", "--force"])

        assert result.exit_code == 0
        assert "Deleted 5 documents" in result.output

    @patch("atomate2.siesta.cli.database.siesta_database.test_mongodb_connection")
    def test_clear_without_force_abort(self, mock_mongo_test, runner):
        """Test clear without --force aborts when user declines."""
        client = MagicMock()
        db = MagicMock()
        coll = MagicMock()

        mock_mongo_test.return_value = (True, client, db, coll, None)
        coll.count_documents.return_value = 5

        # Simulate user declining confirmation
        result = runner.invoke(cli, ["clear"], input="n\n")

        # Click Abort raises exception which sets exit_code to 1, but handler catches it
        # so we check for "cancelled" in output instead
        assert "cancelled" in result.output.lower() or "abort" in result.output.lower()

    @patch("atomate2.siesta.cli.database.siesta_database.test_mongodb_connection")
    def test_drop_collection_with_force(self, mock_mongo_test, runner):
        """Test dropping collection with --force flag."""
        client = MagicMock()
        db = MagicMock()
        coll = MagicMock()

        mock_mongo_test.return_value = (True, client, db, coll, None)
        db.list_collection_names.return_value = ["tasks"]
        coll.count_documents.return_value = 10
        coll.list_indexes.return_value = iter([{"name": "_id_"}, {"name": "uuid_1"}])

        result = runner.invoke(cli, ["clear", "--drop-collection", "--force"])

        assert result.exit_code == 0
        coll.drop.assert_called_once()

    @patch("atomate2.siesta.cli.database.siesta_database.test_mongodb_connection")
    def test_drop_database_requires_confirmation(self, mock_mongo_test, runner):
        """Test dropping database requires typing database name."""
        client = MagicMock()
        db = MagicMock()
        coll = MagicMock()

        mock_mongo_test.return_value = (True, client, db, coll, None)
        db.command.return_value = {"dataSize": 1024 * 1024, "collections": 2}
        db.list_collection_names.return_value = ["tasks", "task_data"]

        # Simulate typing wrong database name
        result = runner.invoke(cli, ["clear", "--drop-database"], input="wrongname\n")

        assert result.exit_code == 0
        assert "doesn't match" in result.output or "cancelled" in result.output.lower()

    def test_conflicting_options_error(self, runner):
        """Test that --drop-collection and --drop-database cannot be used together."""
        result = runner.invoke(cli, ["clear", "--drop-collection", "--drop-database"])

        assert result.exit_code == 0
        assert "Cannot use both" in result.output


class TestCreateCommand:
    """Test 'create' CLI command."""

    @patch("atomate2.siesta.cli.database.siesta_database.test_mongodb_connection")
    def test_create_database_basic(self, mock_mongo_test, runner):
        """Test creating database without indexes."""
        client = MagicMock()
        db = MagicMock()
        coll = MagicMock()

        mock_mongo_test.return_value = (True, client, db, coll, None)
        db.list_collection_names.return_value = []

        # Mock insert/delete for collection creation
        coll.insert_one.return_value.inserted_id = "dummy_id"

        # Mock collection stats
        db.command.return_value = {"count": 0}
        coll.list_indexes.return_value = iter([{"name": "_id_"}])

        result = runner.invoke(cli, ["create"])

        assert result.exit_code == 0
        assert "Database setup complete" in result.output

    @patch("atomate2.siesta.cli.database.siesta_database.test_mongodb_connection")
    def test_create_with_indexes(self, mock_mongo_test, runner):
        """Test creating database with indexes."""
        client = MagicMock()
        db = MagicMock()
        coll = MagicMock()

        mock_mongo_test.return_value = (True, client, db, coll, None)
        db.list_collection_names.return_value = []

        coll.insert_one.return_value.inserted_id = "dummy_id"
        db.command.return_value = {"count": 0}
        coll.list_indexes.return_value = iter(
            [
                {"name": "_id_"},
                {"name": "uuid_1", "unique": True},
                {"name": "formula_1"},
            ]
        )

        result = runner.invoke(cli, ["create", "--create-indexes"])

        assert result.exit_code == 0
        # Check that create_index was called multiple times
        assert (
            coll.create_index.call_count >= 4
        )  # uuid, formula, formula_pretty, state, etc.

    @patch("atomate2.siesta.cli.database.siesta_database.test_mongodb_connection")
    def test_create_existing_collection(self, mock_mongo_test, runner):
        """Test creating when collection already exists."""
        client = MagicMock()
        db = MagicMock()
        coll = MagicMock()

        mock_mongo_test.return_value = (True, client, db, coll, None)
        db.list_collection_names.return_value = ["tasks"]
        coll.count_documents.return_value = 5

        # Simulate user declining to continue
        result = runner.invoke(cli, ["create"], input="n\n")

        assert result.exit_code == 0
        assert "already exists" in result.output


class TestConfigCommand:
    """Test 'config' CLI command."""

    def test_config_show_examples(self, runner):
        """Test config command shows examples."""
        result = runner.invoke(cli, ["config"])

        assert result.exit_code == 0
        assert (
            "Jobflow Configuration" in result.output or "jobflow.yaml" in result.output
        )
        assert "MongoStore" in result.output

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.home")
    @patch("pathlib.Path.exists")
    def test_config_generate_new_file(self, mock_exists, mock_home, mock_file, runner):
        """Test generating new config file."""
        # Mock home directory
        mock_home.return_value = Path("/home/user")
        mock_exists.return_value = False  # File doesn't exist

        result = runner.invoke(cli, ["config", "--generate"])

        assert result.exit_code == 0
        assert "Successfully created jobflow configuration" in result.output
        mock_file.assert_called_once()

    @patch("pathlib.Path.home")
    @patch("pathlib.Path.exists")
    def test_config_generate_existing_file_abort(self, mock_exists, mock_home, runner):
        """Test generating config when file exists and user aborts."""
        mock_home.return_value = Path("/home/user")
        mock_exists.return_value = True  # File exists

        # Simulate user declining to overwrite
        result = runner.invoke(cli, ["config", "--generate"], input="n\n")

        assert result.exit_code == 0
        assert (
            "Operation cancelled" in result.output
            or "File already exists" in result.output
        )

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.home")
    @patch("pathlib.Path.exists")
    def test_config_generate_with_force(
        self, mock_exists, mock_home, mock_file, runner
    ):
        """Test generating config with --force flag overwrites."""
        mock_home.return_value = Path("/home/user")
        mock_exists.return_value = True  # File exists

        result = runner.invoke(cli, ["config", "--generate", "--force"])

        assert result.exit_code == 0
        mock_file.assert_called_once()


class TestSetupCommand:
    """Test 'setup' CLI command."""

    @patch("subprocess.run")
    @patch("platform.system")
    def test_setup_check_only_installed(self, mock_system, mock_subprocess, runner):
        """Test --check-only when MongoDB is installed."""
        mock_system.return_value = "Darwin"

        # Mock mongod --version success
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "db version v7.0.0\n"
        mock_subprocess.return_value = mock_result

        result = runner.invoke(cli, ["setup", "--check-only"])

        assert result.exit_code == 0
        assert "MongoDB is already installed" in result.output

    @patch("subprocess.run")
    def test_setup_check_only_not_installed(self, mock_subprocess, runner):
        """Test --check-only when MongoDB is not installed."""
        # Mock mongod not found
        mock_subprocess.side_effect = FileNotFoundError

        result = runner.invoke(cli, ["setup", "--check-only"])

        assert result.exit_code != 0
        assert "MongoDB is not installed" in result.output


class TestInfoCommand:
    """Test 'info' CLI command."""

    def test_info_displays_help(self, runner):
        """Test info command displays comprehensive information."""
        result = runner.invoke(cli, ["info"])

        assert result.exit_code == 0
        assert (
            "atomate2siesta Database CLI" in result.output
            or "Database CLI" in result.output
        )
        # Check that all commands are listed
        assert "test" in result.output
        assert "create" in result.output
        assert "list" in result.output
        assert "query" in result.output
        assert "stats" in result.output


class TestCLIGroup:
    """Test CLI group functionality."""

    def test_cli_help(self, runner):
        """Test CLI help output."""
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "database" in result.output.lower()
        # Check that main commands are listed
        assert "test" in result.output
        assert "create" in result.output

    def test_cli_version(self, runner):
        """Test CLI version option."""
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "version" in result.output.lower() or "0.1.0" in result.output

    def test_command_with_custom_host_port(self, runner):
        """Test that commands accept custom host/port."""
        with patch(
            "atomate2.siesta.cli.database.siesta_database.test_mongodb_connection"
        ) as mock:
            mock.return_value = (False, None, None, None, "Connection failed")

            result = runner.invoke(
                cli,
                [
                    "test",
                    "--host",
                    "cluster.edu",
                    "--port",
                    "27018",
                    "--database",
                    "mydb",
                    "--collection",
                    "mycoll",
                ],
            )

            # Verify custom parameters were passed
            mock.assert_called_with("cluster.edu", 27018, "mydb", "mycoll")
            assert result.exit_code == 0
