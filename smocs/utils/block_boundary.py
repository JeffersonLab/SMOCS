# Copyright 2024, Jefferson Science Associates, LLC.
# Subject to the terms in the License.txt file found in the top-level directory.

"""
Shared "block boundary" rule: what it means for one sample, arriving after
another, to begin a new block rather than continue the current one.

This single rule is relied on from two independent places that must never be
allowed to drift apart:

  - DBManager._compute_block_id (smocs/db/mysql_api_v0.py) uses it to assign
    the persisted block_id column to each row written to agent_inferences.

  - AutoencoderMLInferenceThread's streaming sliding-window buffer
    (smocs/agents/autoencoder_agent.py) uses the identical rule, purely
    in-memory and with no database involved, to decide when to reset its
    buffer of recent samples. Resetting on every detected boundary guarantees
    that any window later sliced from that buffer is block-homogeneous - the
    streaming analogue of the guarantee DBManager.sample_window enforces,
    after the fact, for windows sampled from the database (see that method's
    docstring). Querying the database for each streaming sample's actual
    block_id, instead of recomputing the rule locally, was deliberately
    rejected: it would race the ingest thread's own asynchronous write for
    that same message (there is no guarantee that write has landed by the
    time inference processes it - see _fetch_and_preprocess_batch's docstring
    for a previously-observed bug from exactly this kind of race), and would
    add a database round-trip to every streamed message solely for
    bookkeeping.
"""

from datetime import datetime


def parse_timestamp(ts):
    """
    Normalizes a timestamp value into a datetime object, regardless of which
    of three forms it currently takes: it may already be a datetime object;
    it may be a string in the exact format store_message() writes when
    constructing a new row prior to insertion ('%Y-%m-%d %H:%M:%S.%f'); or it
    may be a numeric POSIX epoch timestamp in seconds, as carried in
    message_data['timestamp'] / inference_request['timestamp'] throughout the
    Kafka message pipeline. This normalization allows callers to perform
    datetime arithmetic - specifically, subtraction, to compute an elapsed gap
    in seconds - uniformly, without needing to know which of the three
    original forms a given timestamp happened to arrive in.

    Args:
        ts: A datetime object, a string formatted as '%Y-%m-%d %H:%M:%S.%f',
            or a numeric (int/float) POSIX epoch timestamp in seconds.

    Returns:
        datetime: The equivalent datetime object.
    """
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts)
    return datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f')


def is_new_block(prev_timestamp, prev_context, new_timestamp, new_context, max_gap_seconds):
    """
    Determines whether a new sample, arriving at new_timestamp with
    new_context, should begin a new block relative to the immediately
    preceding sample (prev_timestamp, prev_context).

    A new block begins - this function returns True - under either of two
    conditions: first, if the gap between new_timestamp and prev_timestamp
    exceeds max_gap_seconds, which is taken to indicate that data collection
    was interrupted in the interim; or second, if new_context differs, under
    exact equality, from prev_context, which is taken to indicate that the
    agent's operating context has changed. Context values are compared using
    exact equality, rather than any numerical tolerance, because they are
    assumed to already be categorical or discretized by the time they reach
    this comparison. If neither condition holds, this function returns False,
    and the new sample is considered to continue the same block as the
    preceding one.

    Comparing already-normalized plain tuples (or None) for context here,
    rather than raw numpy arrays, is deliberate: a numpy array's `!=` is
    elementwise, returning another array rather than a plain bool, which is
    exactly the kind of expression that raises "truth value of an array is
    ambiguous" if used directly in the boolean `or` below. Callers are
    responsible for that normalization before calling here.

    Args:
        prev_timestamp (datetime or None): The immediately preceding sample's
            timestamp, or None if there is no preceding sample yet (for
            example, the very first sample of a fresh stream or a freshly
            reset buffer) - in which case this function always returns False,
            since a first sample cannot itself begin a "new" block relative
            to nothing.
        prev_context (tuple or None): The immediately preceding sample's
            context value, already normalized to a plain tuple, or None if
            this caller has no notion of context.
        new_timestamp (datetime): The new sample's timestamp.
        new_context (tuple or None): The new sample's context value, already
            normalized to a plain tuple, or None if this caller has no notion
            of context.
        max_gap_seconds (float): The maximum permitted gap, in seconds,
            between prev_timestamp and new_timestamp before a new block is
            begun on that basis alone.

    Returns:
        bool: True if the new sample should begin a new block, False if it
            continues the same block as the preceding sample.
    """
    if prev_timestamp is None:
        return False

    gap_seconds = (new_timestamp - prev_timestamp).total_seconds()
    gap_triggered = gap_seconds > max_gap_seconds
    context_changed = new_context != prev_context

    return gap_triggered or context_changed
