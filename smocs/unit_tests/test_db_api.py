# test_dbmanager.py
import pytest
import decimal
import numpy as np
import pickle
from unittest.mock import MagicMock, patch
from smocs.db.mysql_api_v0 import DBManager


@pytest.fixture
def mock_db_objects():
    """Fixture to create a mock MySQL connection and cursor."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_conn.is_connected.return_value = True
    return mock_conn, mock_cursor


@pytest.fixture
def db_manager(mock_db_objects):
    mock_conn, mock_cursor = mock_db_objects
    db_cfg = {"host": "localhost", "user": "user", "pwd": "pwd", "database": "testdb"}
    with patch("smocs.db.mysql_api_v0.mysql.connect", return_value=mock_conn):
        manager = DBManager(db_cfg)
    # Inject mocks directly
    manager.mydb = mock_conn
    manager.db_cursor = mock_cursor
    return manager


def test_connect_always_uses_agent_database_name(mock_db_objects):
    """
    connect() must always connect to DBManager.AGENT_DATABASE_NAME, regardless of
    whether (or what) db_cfg supplies under 'database' - see AGENT_DATABASE_NAME's
    docstring for why a per-agent-configurable database name is never used.
    """
    mock_conn, _ = mock_db_objects
    db_cfg = {"host": "localhost", "user": "user", "pwd": "pwd", "database": "some-other-db"}
    with patch("smocs.db.mysql_api_v0.mysql.connect", return_value=mock_conn) as mock_connect:
        DBManager(db_cfg)
    assert mock_connect.call_args.kwargs["database"] == DBManager.AGENT_DATABASE_NAME
    assert mock_connect.call_args.kwargs["database"] != "some-other-db"


def test_is_connected(db_manager, mock_db_objects):
    mock_conn, _ = mock_db_objects
    mock_conn.is_connected.return_value = True
    assert db_manager.is_connected() is True
    mock_conn.is_connected.return_value = False
    assert db_manager.is_connected() is False


def test_execute_and_commit_success(db_manager):
    db_manager.db_cursor.execute.return_value = None
    db_manager.mydb.commit.return_value = None
    status = db_manager._DBManager__execute_and_commit("SELECT 1")
    assert status == 0
    db_manager.db_cursor.execute.assert_called_once_with("SELECT 1")


def test_execute_and_commit_failure(db_manager):
    db_manager.db_cursor.execute.side_effect = Exception("SQL error")
    with pytest.raises(Exception):
        db_manager._DBManager__execute_and_commit("BAD SQL")


def test_execute_query_success(db_manager):
    db_manager.db_cursor.fetchall.return_value = [{"id": 1}]
    result = db_manager._DBManager__execute_query("SELECT * FROM t")
    assert result == [{"id": 1}]
    db_manager.db_cursor.execute.assert_called_once()


def test_parse_results_converts_decimal_and_bytes(db_manager):
    results = [
        {
            "num": decimal.Decimal("3.14"),
            "data": np.array([1.0, 2.0]).tobytes(),
            "obj": pickle.dumps({"a": 1}),
        }
    ]
    parsed = db_manager.parse_results(results)
    assert parsed[0]["num"] == 3.14
    assert isinstance(parsed[0]["data"], np.ndarray)
    assert parsed[0]["obj"] == {"a": 1}


def test_get_timestamps_random_mode(db_manager):
    db_manager._DBManager__execute_query = MagicMock(
        return_value=[{"state_source_timestamp": "2025-08-01"}]
    )
    db_manager.parse_results = MagicMock(return_value=[{"state_source_timestamp": "2025-08-01"}])
    results = db_manager.get_timestamps(5, mode="random", n=1)
    assert results[0]["state_source_timestamp"] == "2025-08-01"


def test_get_timestamps_invalid_mode(db_manager):
    result = db_manager.get_timestamps(5, mode="unknown")
    assert result is None


def test_check_sample_feasibility(db_manager):
    db_manager.get_size = MagicMock(return_value=10)
    assert db_manager.check_sample_feasibility(5, "diagnostics") is True
    db_manager.get_size = MagicMock(side_effect=[3, 3])
    assert db_manager.check_sample_feasibility(5, "controls") is False


def test_record_sensor_data_inserts_correctly(db_manager):
    data = {"state_source_timestamp": "ts", "state": np.array([1.0, 2.0])}
    db_manager._DBManager__execute_and_commit = MagicMock(return_value=0)
    status = db_manager.record_sensor_data(data)
    assert status == 0
    db_manager._DBManager__execute_and_commit.assert_called_once()


def test_record_sensor_data_empty(db_manager):
    status = db_manager.record_sensor_data({})
    assert status == 0


def test_get_state_id_single_result(db_manager):
    db_manager._DBManager__execute_query = MagicMock(return_value=[{"id": 42}])
    assert db_manager.get_state_id("ts") == 42


def test_get_state_id_multiple_results(db_manager):
    db_manager._DBManager__execute_query = MagicMock(return_value=[{"id": 1}, {"id": 2}])
    ids = db_manager.get_state_id("ts")
    assert ids == [1, 2]


def test_get_state_id_none(db_manager):
    db_manager._DBManager__execute_query = MagicMock(return_value=[])
    assert db_manager.get_state_id("ts") is None


def test_record_prediction_success(db_manager):
    db_manager._DBManager__execute_and_commit = MagicMock(return_value=0)
    pred = np.array([1.0])
    status = db_manager.record_prediction(pred, "2025-08-01", "key")
    assert status == 0


def test_record_controls_tuple_success(db_manager):
    db_manager._DBManager__execute_and_commit = MagicMock(return_value=0)
    data = {"next_state": np.array([1.0]), "reward": np.array([1.0]), "terminate": True, "truncate": False}
    status = db_manager.record_controls_tuple(data, state_id=1)
    assert status == 0


def test_get_size_returns_count(db_manager):
    db_manager.db_cursor.fetchone.return_value = {"COUNT(*)": 5}
    assert db_manager.get_size("agent_inferences") == 5


def test_close_calls_close_methods(db_manager, mock_db_objects):
    mock_conn, mock_cursor = mock_db_objects
    db_manager.close()
    mock_cursor.close.assert_called_once()
    mock_conn.close.assert_called_once()
