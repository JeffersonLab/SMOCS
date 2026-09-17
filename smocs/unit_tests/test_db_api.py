# test_dbmanager.py
import pytest
import decimal
import logging
import numpy as np
import pickle
from datetime import datetime, timedelta
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
    """
    Same as db_manager, but configured with a gap threshold. agent_inferences'
    context column is fixed and always present regardless of configuration - see
    DBManager.__init__ - so, unlike max_gap_seconds, there is no separate
    context-related config key to set up here; tests exercise context behavior by
    including (or omitting) a 'context' entry directly in their test data.
    """
    mock_conn, mock_cursor = mock_db_objects
    db_cfg = {
        "host": "localhost", "user": "user", "pwd": "pwd", "max_gap_seconds": 5.0,
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
    db_manager._latest_row = {"timestamp": datetime(2025, 8, 1, 0, 0, 0), "context": None, "block_id": 2}
    data = {"state_source_timestamp": "2026-01-01 00:00:00.000000"}  # huge gap
    assert db_manager._compute_block_id(data, None) == 2


# ---------------------------------------------------------------------------
# block_id computation
# ---------------------------------------------------------------------------

def test_compute_block_id_first_row_is_zero(db_manager_ctx):
    db_manager_ctx._latest_row = None
    data = {"state_source_timestamp": "2025-08-01 00:00:00.000000"}
    assert db_manager_ctx._compute_block_id(data, (0.0, 1.0)) == 0


def test_compute_block_id_new_block_on_gap(db_manager_ctx):
    db_manager_ctx._latest_row = {
        "timestamp": datetime(2025, 8, 1, 0, 0, 0),
        "context": (0.0, 1.0),
        "block_id": 3,
    }
    # 10s gap > max_gap_seconds (5.0), same context
    data = {"state_source_timestamp": "2025-08-01 00:00:10.000000"}
    assert db_manager_ctx._compute_block_id(data, (0.0, 1.0)) == 4


def test_compute_block_id_new_block_on_context_change(db_manager_ctx):
    db_manager_ctx._latest_row = {
        "timestamp": datetime(2025, 8, 1, 0, 0, 0),
        "context": (0.0, 1.0),
        "block_id": 3,
    }
    # within the gap, but context changed
    data = {"state_source_timestamp": "2025-08-01 00:00:01.000000"}
    assert db_manager_ctx._compute_block_id(data, (1.0, 1.0)) == 4


def test_compute_block_id_continues_block(db_manager_ctx):
    db_manager_ctx._latest_row = {
        "timestamp": datetime(2025, 8, 1, 0, 0, 0),
        "context": (0.0, 1.0),
        "block_id": 3,
    }
    # within the gap, same context
    data = {"state_source_timestamp": "2025-08-01 00:00:01.000000"}
    assert db_manager_ctx._compute_block_id(data, (0.0, 1.0)) == 3


def test_record_sensor_data_injects_block_id_and_updates_cache(db_manager_ctx):
    db_manager_ctx._latest_row = None
    db_manager_ctx._DBManager__execute_and_commit = MagicMock(return_value=0)
    data = {
        "state_source_timestamp": "2025-08-01 00:00:00.000000",
        "state": np.array([1.0, 2.0]),
        "context": np.array([0.0, 1.0], dtype=np.float32),
    }
    status = db_manager_ctx.record_sensor_data(data)
    assert status == 0
    assert data["block_id"] == 0
    # 'context', like 'state', is serialized to bytes in place before the INSERT -
    # see record_sensor_data - so the caller's own dict reflects that afterward too.
    assert isinstance(data["context"], bytes)
    assert db_manager_ctx._latest_row["block_id"] == 0
    assert db_manager_ctx._latest_row["context"] == (0.0, 1.0)


def test_record_sensor_data_without_context_leaves_cache_context_none(db_manager):
    # An agent with no notion of context (for example, the plain autoencoder) simply
    # never includes a 'context' entry at all - this must not raise or otherwise be
    # treated as an error.
    db_manager._latest_row = None
    db_manager._DBManager__execute_and_commit = MagicMock(return_value=0)
    data = {"state_source_timestamp": "2025-08-01 00:00:00.000000", "state": np.array([1.0, 2.0])}
    status = db_manager.record_sensor_data(data)
    assert status == 0
    assert db_manager._latest_row["context"] is None


def test_refresh_latest_row_cache_seeds_from_db(db_manager_ctx):
    db_manager_ctx._DBManager__execute_query = MagicMock(return_value=[
        {"state_source_timestamp": datetime(2025, 8, 1, 0, 0, 0), "block_id": 7,
         "context": np.array([0.0, 1.0], dtype=np.float32).tobytes()}
    ])
    db_manager_ctx.refresh_latest_row_cache()
    assert db_manager_ctx._latest_row["block_id"] == 7
    assert db_manager_ctx._latest_row["context"] == (0.0, 1.0)


def test_refresh_latest_row_cache_seeds_none_context_when_null(db_manager_ctx):
    # agent_inferences' context column is nullable - a row written by an agent with
    # no notion of context (or written before context was ever introduced) has NULL
    # there, which parse_results leaves as None rather than decoding as bytes.
    db_manager_ctx._DBManager__execute_query = MagicMock(return_value=[
        {"state_source_timestamp": datetime(2025, 8, 1, 0, 0, 0), "block_id": 3, "context": None}
    ])
    db_manager_ctx.refresh_latest_row_cache()
    assert db_manager_ctx._latest_row["context"] is None


def test_refresh_latest_row_cache_empty_table(db_manager_ctx):
    db_manager_ctx._DBManager__execute_query = MagicMock(return_value=[])
    db_manager_ctx.refresh_latest_row_cache()
    assert db_manager_ctx._latest_row is None


def test_validate_identifier_rejects_backtick():
    with pytest.raises(ValueError):
        DBManager._validate_identifier("block_id`; DROP TABLE agent_inferences; --")
    # Dotted PV-style names (e.g. "IPMK203.XPOS") are valid once backtick-quoted -
    # only a literal backtick is rejected.
    DBManager._validate_identifier("IPMK203.XPOS")  # should not raise
    DBManager._validate_identifier("valid_name_1")  # should not raise


# ---------------------------------------------------------------------------
# Window validity (block-homogeneity)
# ---------------------------------------------------------------------------

def test_sample_window_rejects_block_crossing_window(db_manager):
    db_manager.db_cursor.fetchall.return_value = [
        {"state_source_timestamp": "t1", "state": b"", "block_id": 1},
        {"state_source_timestamp": "t2", "state": b"", "block_id": 2},
    ]
    result = db_manager.sample_window("t1", "diagnostics", window_size=2)
    assert result is None


def test_sample_window_accepts_block_homogeneous_window(db_manager):
    db_manager.db_cursor.fetchall.return_value = [
        {"state_source_timestamp": "t1", "state": b"", "block_id": 1},
        {"state_source_timestamp": "t2", "state": b"", "block_id": 1},
    ]
    result = db_manager.sample_window("t1", "diagnostics", window_size=2)
    assert result is not None
    assert len(result) == 2


def test_sample_window_rejects_insufficient_rows(db_manager):
    db_manager.db_cursor.fetchall.return_value = [
        {"state_source_timestamp": "t1", "state": b"", "block_id": 1},
    ]
    result = db_manager.sample_window("t1", "diagnostics", window_size=2)
    assert result is None


# ---------------------------------------------------------------------------
# sampling_lookback parsing
# ---------------------------------------------------------------------------

def test_parse_lookback_accepts_timedelta_unchanged():
    td = timedelta(hours=3)
    assert DBManager._parse_lookback(td) is td


def test_parse_lookback_parses_strings():
    assert DBManager._parse_lookback("24h") == timedelta(hours=24)
    assert DBManager._parse_lookback("90m") == timedelta(minutes=90)
    assert DBManager._parse_lookback("3d") == timedelta(days=3)
    assert DBManager._parse_lookback("30s") == timedelta(seconds=30)


def test_parse_lookback_rejects_invalid_string():
    with pytest.raises(ValueError):
        DBManager._parse_lookback("not-a-duration")


# ---------------------------------------------------------------------------
# "now" anchoring and block discovery
# ---------------------------------------------------------------------------

def test_get_latest_timestamp_returns_none_when_empty(db_manager):
    db_manager._DBManager__execute_query = MagicMock(return_value=[])
    assert db_manager._get_latest_timestamp() is None


def test_get_latest_timestamp_returns_parsed_value(db_manager):
    db_manager._DBManager__execute_query = MagicMock(
        return_value=[{"state_source_timestamp": datetime(2025, 8, 1, 0, 0, 0)}]
    )
    assert db_manager._get_latest_timestamp() == datetime(2025, 8, 1, 0, 0, 0)


def test_get_block_row_counts_returns_counts_per_block(db_manager):
    db_manager._DBManager__execute_query = MagicMock(
        return_value=[{"block_id": 1, "row_count": 7}, {"block_id": 3, "row_count": 2}]
    )
    assert db_manager._get_block_row_counts(datetime(2025, 8, 1)) == {1: 7, 3: 2}


def test_get_timestamps_diagnostics_feasibility_check_is_block_scoped(db_manager):
    # Regression test: the inner feasibility count must be scoped to the same
    # block_id as the candidate row, not to the whole table - otherwise a
    # candidate near the end of one block could look feasible only because a
    # later, different block's rows pad the count, when sample_window would
    # go on to reject it as block-crossing.
    db_manager._DBManager__execute_query = MagicMock(return_value=[])
    db_manager.get_timestamps(window_size=1, mode="random", n=1, agent_type="diagnostics")
    query = db_manager._DBManager__execute_query.call_args.args[0]
    assert "ai2.block_id = agent_inferences.block_id" in query


# ---------------------------------------------------------------------------
# _allocate_with_redistribution (water-filling)
# ---------------------------------------------------------------------------

def test_allocate_no_shortfall_splits_evenly():
    alloc = DBManager._allocate_with_redistribution(100, {0: 50, 1: 50})
    assert alloc == {0: 50, 1: 50}


def test_allocate_redistributes_shortfall_to_blocks_with_room():
    # block 0 can only supply 10; its shortfall (against an equal 25 share) is
    # redistributed evenly across the other three blocks, which have room.
    alloc = DBManager._allocate_with_redistribution(100, {0: 10, 1: 50, 2: 50, 3: 50})
    assert alloc[0] == 10
    assert sum(alloc.values()) == 100
    assert alloc[1] == alloc[2] == alloc[3] == 30


def test_allocate_caps_at_total_available_when_insufficient():
    alloc = DBManager._allocate_with_redistribution(100, {0: 5, 1: 5})
    assert alloc == {0: 5, 1: 5}
    assert sum(alloc.values()) == 10


def test_allocate_handles_zero_availability_blocks():
    alloc = DBManager._allocate_with_redistribution(10, {0: 0, 1: 10})
    assert alloc == {0: 0, 1: 10}


# ---------------------------------------------------------------------------
# sample_batch - block stratification over the lookback window
# ---------------------------------------------------------------------------

def test_sample_batch_stratifies_equally_across_blocks_in_window(db_manager_ctx):
    db_manager_ctx.check_sample_feasibility = MagicMock(return_value=True)
    db_manager_ctx._get_latest_timestamp = MagicMock(return_value=datetime(2025, 8, 2, 0, 0, 0))
    # window_size=1 below, so each block's exact capacity equals its row count.
    db_manager_ctx._get_block_row_counts = MagicMock(return_value={1: 25, 2: 25, 3: 25, 4: 25})
    fake_window = [{"state_source_timestamp": "t", "state": np.zeros(2), "context": np.array([0.0, 1.0])}]
    db_manager_ctx._collect_windows_for_block = MagicMock(
        side_effect=lambda block_id, target_n, window_size, agent_type, mode, min_timestamp:
            [fake_window] * target_n
    )

    batch = db_manager_ctx.sample_batch(batch_size=100, window_size=1, agent_type="diagnostics")

    # four equally-sized blocks split batch_size evenly with no shortfall, so
    # each block is asked for (and supplies) exactly its 25-window share, in
    # a single call each - no exploratory or top-up rounds.
    assert db_manager_ctx._collect_windows_for_block.call_count == 4
    for call in db_manager_ctx._collect_windows_for_block.call_args_list:
        assert call.kwargs["target_n"] == 25
    assert len(batch["state"]) == 100


def test_sample_batch_redistributes_shortfall_from_thin_block(db_manager_ctx):
    db_manager_ctx.check_sample_feasibility = MagicMock(return_value=True)
    db_manager_ctx._get_latest_timestamp = MagicMock(return_value=datetime(2025, 8, 2, 0, 0, 0))
    # window_size=1, so block 1's true capacity is only 10, versus 50 each
    # for blocks 2-4.
    db_manager_ctx._get_block_row_counts = MagicMock(return_value={1: 10, 2: 50, 3: 50, 4: 50})
    fake_window = [{"state_source_timestamp": "t", "state": np.zeros(2), "context": np.array([0.0, 1.0])}]
    db_manager_ctx._collect_windows_for_block = MagicMock(
        side_effect=lambda block_id, target_n, window_size, agent_type, mode, min_timestamp:
            [fake_window] * target_n
    )

    batch = db_manager_ctx.sample_batch(batch_size=100, window_size=1, agent_type="diagnostics")

    # block 1's true capacity (10) falls short of an equal 25-share, so its
    # 15-window shortfall is redistributed across blocks 2-4 by
    # _allocate_with_redistribution *before* any fetching happens - every
    # block is then asked for its final allocation in exactly one call: block
    # 1 for 10, blocks 2-4 for 30 each (25 base share + a 5 share of the
    # redistributed shortfall) - never for the full batch_size of 100.
    calls_by_block = {call.kwargs["block_id"]: call.kwargs["target_n"]
                      for call in db_manager_ctx._collect_windows_for_block.call_args_list}
    assert calls_by_block == {1: 10, 2: 30, 3: 30, 4: 30}
    assert db_manager_ctx._collect_windows_for_block.call_count == 4
    assert len(batch["state"]) == 100


def test_sample_batch_uses_lookback_window_as_min_timestamp(db_manager_ctx):
    db_manager_ctx.check_sample_feasibility = MagicMock(return_value=True)
    db_manager_ctx._get_latest_timestamp = MagicMock(return_value=datetime(2025, 8, 2, 0, 0, 0))
    db_manager_ctx._get_block_row_counts = MagicMock(return_value={1: 10})
    db_manager_ctx._collect_windows_for_block = MagicMock(return_value=[])

    db_manager_ctx.sample_batch(batch_size=10, window_size=1, agent_type="diagnostics",
                                 sampling_lookback="1h")

    expected_min_ts = datetime(2025, 8, 1, 23, 0, 0)
    db_manager_ctx._get_block_row_counts.assert_called_once_with(expected_min_ts)
    assert db_manager_ctx._collect_windows_for_block.call_args.kwargs["min_timestamp"] == expected_min_ts


def test_sample_batch_no_blocks_in_window_returns_empty_batch(db_manager_ctx, caplog):
    db_manager_ctx.check_sample_feasibility = MagicMock(return_value=True)
    db_manager_ctx._get_latest_timestamp = MagicMock(return_value=datetime(2025, 8, 2, 0, 0, 0))
    db_manager_ctx._get_block_row_counts = MagicMock(return_value={})

    with caplog.at_level(logging.INFO):
        batch = db_manager_ctx.sample_batch(batch_size=10, window_size=1, agent_type="diagnostics")

    assert len(batch["state"]) == 0
    assert any("no blocks found" in rec.message for rec in caplog.records)


def test_sample_batch_short_batch_logs_info(db_manager_ctx, caplog):
    db_manager_ctx.check_sample_feasibility = MagicMock(return_value=True)
    db_manager_ctx._get_latest_timestamp = MagicMock(return_value=datetime(2025, 8, 2, 0, 0, 0))
    # only 3 rows exist in the single block in the window, so only 3 valid
    # windows exist in total - short of the requested batch_size of 10.
    db_manager_ctx._get_block_row_counts = MagicMock(return_value={1: 3})
    fake_window = [{"state_source_timestamp": "t", "state": np.zeros(2), "context": np.array([0.0, 1.0])}]
    db_manager_ctx._collect_windows_for_block = MagicMock(return_value=[fake_window] * 3)

    with caplog.at_level(logging.INFO):
        batch = db_manager_ctx.sample_batch(batch_size=10, window_size=1, agent_type="diagnostics")

    assert len(batch["state"]) == 3
    assert any("returning partial batch" in rec.message for rec in caplog.records)


def test_sample_batch_invalid_sampling_strategy_returns_none(db_manager_ctx):
    db_manager_ctx.check_sample_feasibility = MagicMock(return_value=True)
    result = db_manager_ctx.sample_batch(batch_size=10, window_size=1, agent_type="diagnostics",
                                          sampling_strategy="stratified")
    assert result is None


def test_sample_batch_sampling_strategy_passed_through_to_block_collection(db_manager_ctx):
    db_manager_ctx.check_sample_feasibility = MagicMock(return_value=True)
    db_manager_ctx._get_latest_timestamp = MagicMock(return_value=datetime(2025, 8, 2, 0, 0, 0))
    db_manager_ctx._get_block_row_counts = MagicMock(return_value={1: 10})
    db_manager_ctx._collect_windows_for_block = MagicMock(return_value=[])

    db_manager_ctx.sample_batch(batch_size=10, window_size=1, agent_type="diagnostics",
                                 sampling_strategy="latest")

    assert db_manager_ctx._collect_windows_for_block.call_args.kwargs["mode"] == "latest"


def test_sample_batch_controls_path_unaffected(db_manager):
    db_manager.check_sample_feasibility = MagicMock(return_value=True)
    db_manager._sample_batch_controls = MagicMock(return_value={"state": np.zeros((1, 1))})

    db_manager.sample_batch(batch_size=1, window_size=1, agent_type="controls")

    db_manager._sample_batch_controls.assert_called_once_with(1, 1)


# ---------------------------------------------------------------------------
# get_timestamps - equality_filter / min_timestamp
# ---------------------------------------------------------------------------

def test_get_timestamps_applies_equality_filter_and_min_timestamp(db_manager):
    db_manager._DBManager__execute_query = MagicMock(return_value=[])
    db_manager.get_timestamps(window_size=1, mode="random", n=5, agent_type="diagnostics",
                               equality_filter={"block_id": 3},
                               min_timestamp=datetime(2025, 8, 1))

    query, kwargs = db_manager._DBManager__execute_query.call_args.args[0], db_manager._DBManager__execute_query.call_args.kwargs
    assert "`block_id` = %s" in query
    assert "state_source_timestamp >= %s" in query
    assert kwargs["values"] == (3, datetime(2025, 8, 1))


# ---------------------------------------------------------------------------
# _collect_windows_for_block
# ---------------------------------------------------------------------------

def test_collect_windows_for_block_stops_at_target_n(db_manager_ctx):
    # get_timestamps' real return value is np.array(list_of_dicts) (see
    # parse_results), never a plain list - deliberately mocked as a numpy array
    # here, rather than a list, so this test would catch a regression like
    # `if not candidates:` on that array raising "The truth value of an array
    # with more than one element is ambiguous" once more than one candidate
    # comes back, which a list-mocked candidates value cannot expose.
    db_manager_ctx.get_timestamps = MagicMock(return_value=np.array([
        {"state_source_timestamp": f"t{i}"} for i in range(5)
    ]))
    db_manager_ctx.sample_window = MagicMock(return_value=[{"state_source_timestamp": "t", "block_id": 1}])

    result = db_manager_ctx._collect_windows_for_block(
        block_id=1, target_n=2, window_size=1, agent_type="diagnostics",
        mode="random", min_timestamp=datetime(2025, 8, 1)
    )

    assert len(result) == 2
    kwargs = db_manager_ctx.get_timestamps.call_args.kwargs
    assert kwargs["equality_filter"] == {"block_id": 1}
    assert kwargs["n"] == 2  # never requests more than target_n


def test_collect_windows_for_block_zero_target_returns_immediately(db_manager_ctx):
    db_manager_ctx.get_timestamps = MagicMock()
    result = db_manager_ctx._collect_windows_for_block(
        block_id=1, target_n=0, window_size=1, agent_type="diagnostics",
        mode="random", min_timestamp=datetime(2025, 8, 1)
    )
    assert result == []
    db_manager_ctx.get_timestamps.assert_not_called()


def test_build_batch_dict_shapes(db_manager_ctx):
    window = [
        {"state_source_timestamp": "t1", "state": np.array([1.0, 2.0]), "context": np.array([0.0, 1.0])},
        {"state_source_timestamp": "t2", "state": np.array([3.0, 4.0]), "context": np.array([0.0, 1.0])},
    ]
    batch = db_manager_ctx._build_batch_dict([window, window])
    assert batch["state"].shape == (2, 2, 2)    # (batch_size_eff, window_size, n_input_channels)
    assert batch["context"].shape == (2, 2, 2)  # (batch_size_eff, window_size, n_context_channels)
