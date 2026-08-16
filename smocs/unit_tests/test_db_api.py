# test_dbmanager.py
import pytest
import decimal
import logging
import numpy as np
import pickle
from datetime import datetime
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
    db_cfg = {"host": "localhost", "user": "user", "pwd": "pwd"}
    with patch("smocs.db.mysql_api_v0.mysql.connect", return_value=mock_conn):
        manager = DBManager(db_cfg)
    # Inject mocks directly
    manager.mydb = mock_conn
    manager.db_cursor = mock_cursor
    return manager


@pytest.fixture
def db_manager_ctx(mock_db_objects):
    """Same as db_manager, but configured with context columns + a gap threshold."""
    mock_conn, mock_cursor = mock_db_objects
    db_cfg = {
        "host": "localhost", "user": "user", "pwd": "pwd",
        "context_cols": ["ctx1", "ctx2"], "max_gap_seconds": 5.0,
    }
    with patch("smocs.db.mysql_api_v0.mysql.connect", return_value=mock_conn):
        manager = DBManager(db_cfg)
    manager.mydb = mock_conn
    manager.db_cursor = mock_cursor
    return manager


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


# ---------------------------------------------------------------------------
# max_gap_seconds defaulting
# ---------------------------------------------------------------------------

def test_max_gap_seconds_defaults_to_infinity_when_key_absent(mock_db_objects):
    # Callers (cores/*.py) must omit this key entirely when an agent's config doesn't
    # set it, rather than passing it through as None - dict.get's default only fires
    # when the key itself is absent, not when it's present with value None.
    mock_conn, _ = mock_db_objects
    with patch("smocs.db.mysql_api_v0.mysql.connect", return_value=mock_conn):
        manager = DBManager({"host": "h", "user": "u", "pwd": "p"})
    assert manager.max_gap_seconds == float('inf')


def test_compute_block_id_no_new_block_on_gap_when_max_gap_seconds_unset(db_manager):
    db_manager._latest_row = {"timestamp": datetime(2025, 8, 1, 0, 0, 0), "context": (), "block_id": 2}
    data = {"state_source_timestamp": "2026-01-01 00:00:00.000000"}  # huge gap
    assert db_manager._compute_block_id(data) == 2


# ---------------------------------------------------------------------------
# block_id computation
# ---------------------------------------------------------------------------

def test_compute_block_id_first_row_is_zero(db_manager_ctx):
    db_manager_ctx._latest_row = None
    data = {"state_source_timestamp": "2025-08-01 00:00:00.000000", "ctx1": 0.0, "ctx2": 1.0}
    assert db_manager_ctx._compute_block_id(data) == 0


def test_compute_block_id_new_block_on_gap(db_manager_ctx):
    db_manager_ctx._latest_row = {
        "timestamp": datetime(2025, 8, 1, 0, 0, 0),
        "context": (0.0, 1.0),
        "block_id": 3,
    }
    # 10s gap > max_gap_seconds (5.0), same context
    data = {"state_source_timestamp": "2025-08-01 00:00:10.000000", "ctx1": 0.0, "ctx2": 1.0}
    assert db_manager_ctx._compute_block_id(data) == 4


def test_compute_block_id_new_block_on_context_change(db_manager_ctx):
    db_manager_ctx._latest_row = {
        "timestamp": datetime(2025, 8, 1, 0, 0, 0),
        "context": (0.0, 1.0),
        "block_id": 3,
    }
    # within the gap, but context changed
    data = {"state_source_timestamp": "2025-08-01 00:00:01.000000", "ctx1": 1.0, "ctx2": 1.0}
    assert db_manager_ctx._compute_block_id(data) == 4


def test_compute_block_id_continues_block(db_manager_ctx):
    db_manager_ctx._latest_row = {
        "timestamp": datetime(2025, 8, 1, 0, 0, 0),
        "context": (0.0, 1.0),
        "block_id": 3,
    }
    # within the gap, same context
    data = {"state_source_timestamp": "2025-08-01 00:00:01.000000", "ctx1": 0.0, "ctx2": 1.0}
    assert db_manager_ctx._compute_block_id(data) == 3


def test_record_sensor_data_injects_block_id_and_updates_cache(db_manager_ctx):
    db_manager_ctx._latest_row = None
    db_manager_ctx._DBManager__execute_and_commit = MagicMock(return_value=0)
    data = {
        "state_source_timestamp": "2025-08-01 00:00:00.000000",
        "state": np.array([1.0, 2.0]),
        "ctx1": 0.0,
        "ctx2": 1.0,
    }
    status = db_manager_ctx.record_sensor_data(data)
    assert status == 0
    assert data["block_id"] == 0
    assert db_manager_ctx._latest_row["block_id"] == 0
    assert db_manager_ctx._latest_row["context"] == (0.0, 1.0)


def test_refresh_latest_row_cache_seeds_from_db(db_manager_ctx):
    db_manager_ctx._DBManager__execute_query = MagicMock(return_value=[
        {"state_source_timestamp": datetime(2025, 8, 1, 0, 0, 0), "block_id": 7, "ctx1": 0.0, "ctx2": 1.0}
    ])
    db_manager_ctx.refresh_latest_row_cache()
    assert db_manager_ctx._latest_row["block_id"] == 7
    assert db_manager_ctx._latest_row["context"] == (0.0, 1.0)


def test_refresh_latest_row_cache_empty_table(db_manager_ctx):
    db_manager_ctx._DBManager__execute_query = MagicMock(return_value=[])
    db_manager_ctx.refresh_latest_row_cache()
    assert db_manager_ctx._latest_row is None


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------

def test_migrate_inferences_schema_adds_missing_columns(db_manager_ctx):
    # block_id is declared directly in create_tables()'s CREATE TABLE statement (it's
    # universal, not per-agent) - only context columns are handled dynamically here.
    db_manager_ctx._get_existing_columns = MagicMock(return_value={"id", "state", "block_id"})
    db_manager_ctx._DBManager__execute_and_commit = MagicMock(return_value=0)
    db_manager_ctx._migrate_inferences_schema()
    executed_queries = [c.args[0] for c in db_manager_ctx._DBManager__execute_and_commit.call_args_list]
    assert not any("block_id" in q for q in executed_queries)
    assert any("`ctx1`" in q for q in executed_queries)
    assert any("`ctx2`" in q for q in executed_queries)


def test_migrate_inferences_schema_skips_existing_columns(db_manager_ctx):
    db_manager_ctx._get_existing_columns = MagicMock(return_value={"id", "state", "block_id", "ctx1", "ctx2"})
    db_manager_ctx._DBManager__execute_and_commit = MagicMock(return_value=0)
    db_manager_ctx._migrate_inferences_schema()
    db_manager_ctx._DBManager__execute_and_commit.assert_not_called()


def test_validate_identifier_rejects_backtick():
    with pytest.raises(ValueError):
        DBManager._validate_identifier("ctx1`; DROP TABLE agent_inferences; --")
    # Dotted PV-style names (e.g. "IPMK203.XPOS") are valid once backtick-quoted -
    # only a literal backtick is rejected.
    DBManager._validate_identifier("IPMK203.XPOS")  # should not raise
    DBManager._validate_identifier("valid_name_1")  # should not raise


def test_create_tables_does_not_switch_database(db_manager):
    # Regression test: create_tables() must operate on the DB connect() already
    # selected, not create/switch to a "SMOCS_Agent_*" database.
    db_manager.mydb.database = "original_db"
    db_manager._DBManager__execute_and_commit = MagicMock(return_value=0)
    db_manager._migrate_inferences_schema = MagicMock()
    db_manager.create_tables()
    executed_queries = [c.args[0] for c in db_manager._DBManager__execute_and_commit.call_args_list]
    assert not any("CREATE DATABASE" in q for q in executed_queries)
    assert db_manager.mydb.database == "original_db"


def test_create_tables_declares_block_id_directly(db_manager):
    # block_id is universal (not per-agent), so it's declared directly in the
    # agent_inferences CREATE TABLE statement rather than via dynamic migration.
    db_manager._DBManager__execute_and_commit = MagicMock(return_value=0)
    db_manager._migrate_inferences_schema = MagicMock()
    db_manager.create_tables()
    executed_queries = [c.args[0] for c in db_manager._DBManager__execute_and_commit.call_args_list]
    assert any("CREATE TABLE IF NOT EXISTS agent_inferences" in q and "block_id" in q for q in executed_queries)


# ---------------------------------------------------------------------------
# Sequence validity (block-homogeneity)
# ---------------------------------------------------------------------------

def test_sample_sequence_rejects_block_crossing_window(db_manager):
    db_manager.db_cursor.fetchall.return_value = [
        {"state_source_timestamp": "t1", "state": b"", "block_id": 1},
        {"state_source_timestamp": "t2", "state": b"", "block_id": 2},
    ]
    result = db_manager.sample_sequence("t1", "diagnostics", segment_length=2)
    assert result is None


def test_sample_sequence_accepts_block_homogeneous_window(db_manager):
    db_manager.db_cursor.fetchall.return_value = [
        {"state_source_timestamp": "t1", "state": b"", "block_id": 1},
        {"state_source_timestamp": "t2", "state": b"", "block_id": 1},
    ]
    result = db_manager.sample_sequence("t1", "diagnostics", segment_length=2)
    assert result is not None
    assert len(result) == 2


def test_sample_sequence_rejects_insufficient_rows(db_manager):
    db_manager.db_cursor.fetchall.return_value = [
        {"state_source_timestamp": "t1", "state": b"", "block_id": 1},
    ]
    result = db_manager.sample_sequence("t1", "diagnostics", segment_length=2)
    assert result is None


# ---------------------------------------------------------------------------
# sample_batch - modes / stratified_groups
# ---------------------------------------------------------------------------

def test_sample_batch_mode_latest_ignores_context(db_manager_ctx):
    db_manager_ctx.check_sample_feasibility = MagicMock(return_value=True)
    db_manager_ctx.get_size = MagicMock(return_value=100)
    db_manager_ctx._get_distinct_context_tuples = MagicMock()
    db_manager_ctx._get_latest_context_tuple = MagicMock()
    fake_seq = [{"state_source_timestamp": "t", "state": np.zeros(2), "ctx1": 0.0, "ctx2": 1.0}]
    db_manager_ctx._collect_group_sequences = MagicMock(return_value=[fake_seq])

    batch = db_manager_ctx.sample_batch(batch_size=1, segment_length=1, agent_type="diagnostics", mode="latest")

    db_manager_ctx._get_distinct_context_tuples.assert_not_called()
    db_manager_ctx._get_latest_context_tuple.assert_not_called()
    kwargs = db_manager_ctx._collect_group_sequences.call_args.kwargs
    assert kwargs["context_filter"] is None
    assert batch is not None


def test_sample_batch_stratified_all_equal_weight(db_manager_ctx):
    db_manager_ctx.check_sample_feasibility = MagicMock(return_value=True)
    db_manager_ctx.get_size = MagicMock(return_value=100)
    db_manager_ctx._get_distinct_context_tuples = MagicMock(return_value=[(0.0, 1.0), (1.0, 1.0)])
    fake_seq = [{"state_source_timestamp": "t", "state": np.zeros(2), "ctx1": 0.0, "ctx2": 1.0}]
    db_manager_ctx._collect_group_sequences = MagicMock(return_value=[fake_seq])

    db_manager_ctx.sample_batch(batch_size=10, segment_length=1, agent_type="diagnostics",
                                 mode="stratified", stratified_groups="all")

    assert db_manager_ctx._collect_group_sequences.call_count == 2
    target_ns = sorted(c.kwargs["target_n"] for c in db_manager_ctx._collect_group_sequences.call_args_list)
    assert target_ns == [5, 5]


def test_sample_batch_stratified_latest(db_manager_ctx):
    db_manager_ctx.check_sample_feasibility = MagicMock(return_value=True)
    db_manager_ctx.get_size = MagicMock(return_value=100)
    db_manager_ctx._get_latest_context_tuple = MagicMock(return_value=(1.0, 1.0))
    fake_seq = [{"state_source_timestamp": "t", "state": np.zeros(2), "ctx1": 1.0, "ctx2": 1.0}]
    db_manager_ctx._collect_group_sequences = MagicMock(return_value=[fake_seq])

    db_manager_ctx.sample_batch(batch_size=10, segment_length=1, agent_type="diagnostics",
                                 mode="stratified", stratified_groups="latest")

    db_manager_ctx._collect_group_sequences.assert_called_once()
    kwargs = db_manager_ctx._collect_group_sequences.call_args.kwargs
    assert kwargs["target_n"] == 10
    assert kwargs["context_filter"] == {"ctx1": 1.0, "ctx2": 1.0}


def test_sample_batch_stratified_dict_weights(db_manager_ctx):
    db_manager_ctx.check_sample_feasibility = MagicMock(return_value=True)
    db_manager_ctx.get_size = MagicMock(return_value=100)
    db_manager_ctx._collect_group_sequences = MagicMock(return_value=[])

    db_manager_ctx.sample_batch(batch_size=100, segment_length=1, agent_type="diagnostics",
                                 mode="stratified",
                                 stratified_groups={(0.0, 1.0): 0.7, (1.0, 1.0): 0.3})

    target_ns = sorted(c.kwargs["target_n"] for c in db_manager_ctx._collect_group_sequences.call_args_list)
    assert target_ns == [30, 70]


def test_sample_batch_stratified_dict_weights_normalizes_and_logs_info(db_manager_ctx, caplog):
    db_manager_ctx.check_sample_feasibility = MagicMock(return_value=True)
    db_manager_ctx.get_size = MagicMock(return_value=100)
    db_manager_ctx._collect_group_sequences = MagicMock(return_value=[])

    with caplog.at_level(logging.INFO):
        db_manager_ctx.sample_batch(batch_size=100, segment_length=1, agent_type="diagnostics",
                                     mode="stratified",
                                     stratified_groups={(0.0, 1.0): 0.25, (1.0, 1.0): 0.25})  # sums to 0.5

    assert any("normaliz" in rec.message.lower() for rec in caplog.records)
    target_ns = sorted(c.kwargs["target_n"] for c in db_manager_ctx._collect_group_sequences.call_args_list)
    assert target_ns == [50, 50]


def test_sample_batch_stratified_dict_zero_matches_logs_info(db_manager_ctx, caplog):
    db_manager_ctx.check_sample_feasibility = MagicMock(return_value=True)
    db_manager_ctx.get_size = MagicMock(return_value=100)
    fake_seq = [{"state_source_timestamp": "t", "state": np.zeros(2), "ctx1": 0.0, "ctx2": 1.0}]

    def fake_collect(target_n, segment_length, agent_type, mode, context_filter, candidate_pool_size):
        if context_filter == {"ctx1": 0.0, "ctx2": 1.0}:
            return [fake_seq] * target_n
        return []

    db_manager_ctx._collect_group_sequences = MagicMock(side_effect=fake_collect)

    with caplog.at_level(logging.INFO):
        batch = db_manager_ctx.sample_batch(batch_size=100, segment_length=1, agent_type="diagnostics",
                                             mode="stratified",
                                             stratified_groups={(0.0, 1.0): 0.5, (1.0, 1.0): 0.5})

    assert any("no valid block-homogeneous windows" in rec.message for rec in caplog.records)
    # only the matching group contributed - the other's shortfall was not redistributed
    assert len(batch["state"]) == 50


def test_sample_batch_stratified_invalid_tuple_length_raises(db_manager_ctx):
    db_manager_ctx.check_sample_feasibility = MagicMock(return_value=True)
    with pytest.raises(ValueError):
        db_manager_ctx.sample_batch(batch_size=10, segment_length=1, agent_type="diagnostics",
                                     mode="stratified",
                                     stratified_groups={(0.0, 1.0, 1.0, 1.0): 1.0})


def test_sample_batch_short_batch_logs_info(db_manager_ctx, caplog):
    db_manager_ctx.check_sample_feasibility = MagicMock(return_value=True)
    db_manager_ctx.get_size = MagicMock(return_value=100)
    fake_seq = [{"state_source_timestamp": "t", "state": np.zeros(2), "ctx1": 0.0, "ctx2": 1.0}]
    db_manager_ctx._collect_group_sequences = MagicMock(return_value=[fake_seq] * 3)  # fewer than requested

    with caplog.at_level(logging.INFO):
        batch = db_manager_ctx.sample_batch(batch_size=10, segment_length=1, agent_type="diagnostics", mode="latest")

    assert len(batch["state"]) == 3
    assert any("returning partial batch" in rec.message for rec in caplog.records)


def test_sample_batch_invalid_mode_returns_none(db_manager_ctx):
    db_manager_ctx.check_sample_feasibility = MagicMock(return_value=True)
    result = db_manager_ctx.sample_batch(batch_size=10, segment_length=1, agent_type="diagnostics", mode="random")
    assert result is None


def test_build_batch_dict_shapes(db_manager_ctx):
    seq = [
        {"state_source_timestamp": "t1", "state": np.array([1.0, 2.0]), "ctx1": 0.0, "ctx2": 1.0},
        {"state_source_timestamp": "t2", "state": np.array([3.0, 4.0]), "ctx1": 0.0, "ctx2": 1.0},
    ]
    batch = db_manager_ctx._build_batch_dict([seq, seq])
    assert batch["state"].shape == (2, 2, 2)  # (batch, segment_length, n_channels)
    assert batch["ctx1"].shape == (2, 2)      # (batch, segment_length)
