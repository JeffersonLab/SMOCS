# Author: Kishansingh Rajput (Kishan)
# Organization: Thomas Jefferson National Acceleratory Facility
# Script: Utility script to establish connection to the Database

#Copyright 2024, Jefferson Science Associates, LLC.
#Subject to the terms in the License.txt file found in the top-level directory.

import mysql.connector as mysql
import numpy as np
import logging
import decimal
import time
import pickle
import re
from datetime import datetime, timedelta

class DBManager:

    # Every agent connects to one single, fixed database name, rather than a
    # per-agent-configurable one. This is intentional, not an oversight: init.sql
    # unconditionally creates a database with this exact name (via
    # "CREATE DATABASE IF NOT EXISTS agentdb") when the containing MySQL server is
    # first provisioned, so any other value supplied here would simply fail to
    # connect, since no database by that alternate name would exist. Furthermore,
    # each agent is deployed with its own independent MySQL server process and its
    # own dedicated storage volume (see the orchestration/docker-compose.yml service
    # definitions), so no two agents ever share a single server in the first place;
    # consequently, there has never been a genuine need to distinguish between
    # agents by database name. Should per-agent database names become necessary in
    # the future, that should be undertaken as a dedicated, separate change, since it
    # would additionally require modifying init.sql to accept a configurable name.
    AGENT_DATABASE_NAME = "agentdb"

    # Function to connect with db
    def __init__(self, db_cfg_dict):
        """
        Initializes the DBConnector by loading the database configuration from a YAML file
        and establishing a connection to the MySQL database.

        Args:
            db_cfg_filepath (str): The path of the YAML configuration file containing the database connection details.

        Raises:
            Exception: If there is an error connecting to the MySQL database, the exception is raised
                    and the process is terminated.

        Attributes:
            mydb (MySQLConnection): A MySQL connection object used to interact with the database.
            db_cursor (MySQLCursor): A cursor object for executing SQL queries on the database.

        Example:
            >>> db_connector = DBManager("db_config.yaml")
            Initializing DBConnector
            Connected to DB: my_database
        """

        logging.debug("initializing DBConnector")

        # agent_inferences has a single, fixed 'context' BLOB column (declared
        # directly in init.sql, always present, NULL by default) rather than any
        # per-agent-configurable set of named columns - DBManager itself carries no
        # config or notion of what "context" means for a given agent; it simply
        # stores whatever numpy array, if any, a caller supplies under data['context']
        # to record_sensor_data, and compares it against the previous row's value to
        # help decide when to advance block_id (see _compute_block_id, below, for the
        # complete rule, which also accounts for max_gap_seconds). An agent with no
        # notion of context (for example, the plain, non-contextual autoencoder, or
        # the RL control agent) simply never supplies a 'context' entry at all, in
        # which case every row's context stays None and never contributes to block_id
        # advancement - only max_gap_seconds-based gap detection remains active.

        # The maximum permitted gap, in seconds, between the timestamps of two
        # consecutively written rows before a new block_id is begun. If the calling
        # agent's configuration does not supply this value, it defaults to positive
        # infinity, which has the effect of disabling gap-based block splitting
        # entirely - under that condition, a new block can then only ever be started
        # by a change in the row's context value, as described above.
        self.max_gap_seconds = db_cfg_dict.get('max_gap_seconds', float('inf'))

        # An in-memory cache holding the timestamp, context values, and block_id of
        # the most recently written row, consulted by _compute_block_id to determine
        # the block_id of each subsequent row without needing to re-query the
        # database on every single write. This cache is meaningful only for the one
        # DBManager instance that is actually responsible for writing sensor data
        # (that is, the ingest thread's own instance); it is initialized to None here
        # and is expected to be populated from the database's existing history via an
        # explicit call to refresh_latest_row_cache() before any writing begins.
        self._latest_row = None

        self.connect(db_cfg_dict)

    def is_connected(self):
        """
        Checks if the database connection is established.

        Returns:
            bool: True if the connection is established, False otherwise.
        """
        return self.mydb.is_connected()

    def __execute_and_commit(self, query, values=None):
        """
        Executes a SQL query and commits the changes to the database.

        Args:
            query (str): The SQL query to be executed.

        Returns:
            None

        Raises:
            Exception: If there is an error executing the query, it will logging.debug the error message.
        """
        status = 1
        try:
            if values is not None:
                self.db_cursor.execute(query, values)
            else:
                self.db_cursor.execute(query)

            self.mydb.commit()
            status = 0
        except Exception as e:
            logging.debug("DB Error: ", e)
            logging.debug("with query: ", query)
            raise e

        return status

    def __execute_query(self, query, values=None):
        """
        Executes a SQL query and returns the results.

        Args:
            query (str): The SQL query to be executed.
            values (tuple, optional): The parameter values to be substituted into a
                parameterized query - that is, a query containing %s placeholders.
                If omitted, the query is executed with no parameter substitution.

        Returns:
            list: A list of dictionaries containing the results of the query.

        Raises:
            Exception: If there is an error executing the query, it will logging.debug the error message.
        """
        try:
            if values is None:
                self.db_cursor.execute(query)
            else:
                self.db_cursor.execute(query, values)
            results = self.db_cursor.fetchall()
        except Exception as e:
            logging.debug("DB Error: ", e)
            logging.debug("with query: ", query)
            raise e

        return results

    def connect(self, db_config, n_connection_trials=5):

        """
        Establishes a connection to the MySQL database using the provided configuration.

        The database connected to is always AGENT_DATABASE_NAME; it is not read from
        db_config, and no value supplied within db_config can override it. See the
        explanatory comment beside the AGENT_DATABASE_NAME class attribute, above, for
        the complete rationale.

        Args:
            db_config (dict): A dictionary containing the database connection
                parameters - specifically, the host, user, and password. This
                dictionary is not expected to, and need not, contain the database
                name itself.
            n_connection_trials (int): The number of attempts to connect to the database before giving up.
        Raises:
            Exception: If the connection fails after the specified number of trials, an exception is raised and
                        the process is terminated.
        Returns:
            None
        """

        for i in range(n_connection_trials):
            try:
                logging.debug("Connecting to DB - trial ", i)
                self.mydb = mysql.connect(host=db_config["host"],
                                username=db_config["user"],
                                password=db_config["pwd"],
                                database=self.AGENT_DATABASE_NAME,
                                autocommit=True)

                self.db_cursor = self.mydb.cursor(dictionary=True)
                break
            except Exception as e:
                logging.debug(e)
                if i == n_connection_trials-1:
                    logging.debug(f"Could not connect to DATABASE in {n_connection_trials} attempts, exiting.")
                    exit(1)
                logging.debug("ERROR: CANNOT CONNECT TO DATABASE, WAITING 5 sec...")
                time.sleep(5)

        logging.debug("CONNECTED TO DB: ", self.AGENT_DATABASE_NAME)

    @staticmethod
    def _validate_identifier(name):
        """
        Rejects a proposed SQL identifier (a column name to be interpolated
        directly into a query, such as an equality_filter key in get_timestamps)
        if it contains a backtick character.

        Because column names cannot be supplied as parameterized query values in the
        way that ordinary data values can, any identifier interpolated into a query
        in this file is conventionally enclosed within a pair of backticks (for
        example, as `` `{name}` ``). A backtick occurring within the name itself
        would prematurely terminate that quoting and allow whatever text follows it
        to be interpreted as additional, unintended SQL, rather than as part of the
        identifier. This check exists solely to close off that possibility; no other
        character restriction is imposed, since backtick-quoted identifiers in MySQL
        may otherwise contain a wide range of characters without issue.

        Args:
            name (str): The proposed identifier to validate.

        Raises:
            ValueError: If `name` contains one or more backtick characters.
        """
        if '`' in name:
            raise ValueError(f"Invalid identifier (contains a backtick): {name!r}")

    def refresh_latest_row_cache(self):
        """
        Seeds, or refreshes, the in-memory "latest row" cache (self._latest_row) from
        the database's actual most recent row.

        This method must be called exactly once, at startup, by the single DBManager
        instance responsible for writing sensor data (that is, the ingest thread's
        own instance), before that instance's first call to record_sensor_data(). Its
        purpose is to ensure that block_id assignment continues correctly across
        process restarts: absent this call, a freshly constructed DBManager would
        have no memory of any row written by a prior run, and would therefore assign
        block_id 0 to the very next row it wrote, as though no data had ever been
        recorded for this agent - even though rows from earlier runs may already
        occupy that same block_id value. Calling this method first ensures that the
        block_id sequence instead resumes correctly from wherever it had previously
        left off.
        """
        self._latest_row = self._load_latest_row()

    def _load_latest_row(self):
        """
        Queries agent_inferences for the single row with the highest id - that is,
        the most recently inserted row - and returns the subset of its fields
        relevant to block_id computation.

        Returns:
            dict or None: A dictionary with keys 'timestamp', 'context' (a tuple of
                that row's decoded context values, or None if that row's context
                column was NULL), and 'block_id', describing the most recently
                inserted row; or None if agent_inferences currently contains no
                rows at all.
        """
        query = "SELECT state_source_timestamp, block_id, context FROM agent_inferences ORDER BY id DESC LIMIT 1"
        results = self._DBManager__execute_query(query)
        if not results:
            return None
        row = self.parse_results(results)[0]
        context_value = row.get('context')
        return {
            'timestamp': self._parse_timestamp(row['state_source_timestamp']),
            'context': tuple(context_value) if context_value is not None else None,
            'block_id': int(row['block_id']),
        }

    @staticmethod
    def _parse_timestamp(ts):
        """
        Normalizes a timestamp value into a datetime object, regardless of which of
        two forms it currently takes: it may already be a datetime object, as
        returned directly by the database cursor when a row is read back from the
        database, or it may instead be a string in the exact format that
        store_message() writes when constructing a new row prior to insertion
        ('%Y-%m-%d %H:%M:%S.%f'). This normalization allows subsequent code to
        perform datetime arithmetic - specifically, subtraction, to compute an
        elapsed gap in seconds - uniformly, without needing to know which of the two
        original forms a given timestamp happened to arrive in.

        Args:
            ts: Either a datetime object or a string formatted as
                '%Y-%m-%d %H:%M:%S.%f'.

        Returns:
            datetime: The equivalent datetime object.
        """
        return datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f') if isinstance(ts, str) else ts

    def _compute_block_id(self, data, new_context):
        """
        Determines the block_id that should be assigned to a new row about to be
        inserted, by comparing that row's timestamp and context value against those
        of the most recently written row, as recorded in self._latest_row.

        A new block is begun - that is, the returned block_id is one greater than
        that of the latest row - under either of two conditions: first, if the gap
        between the new row's timestamp and the latest row's timestamp exceeds
        self.max_gap_seconds, which is taken to indicate that data collection was
        interrupted in the interim; or second, if the new row's context value
        differs, under exact equality, from the latest row's context value, which is
        taken to indicate that the agent's operating context has changed. Context
        values are compared using exact equality, rather than any numerical
        tolerance, because they are assumed to already be categorical or discretized
        by the time they reach the database. If neither condition holds, the new row
        is considered to continue the same block as the latest row, and the latest
        row's block_id is returned unchanged.

        Comparing tuples here, rather than the numpy arrays context values actually
        arrive as, is deliberate: a numpy array's `!=` is elementwise, returning
        another array rather than a plain bool, which is exactly the kind of
        expression that raises "truth value of an array is ambiguous" if used
        directly in the boolean `or` below - see record_sensor_data, which converts
        data['context'] to a tuple, once, before calling here.

        Args:
            data (dict): The row about to be inserted. This dictionary must contain
                a 'state_source_timestamp' entry.
            new_context (tuple or None): This row's context value, already
                normalized to a plain tuple (or None, if this row carries no
                context at all) by record_sensor_data.

        Returns:
            int: The block_id to be assigned to this row - either zero, if no prior
                row exists yet for this agent; the latest row's existing block_id
                unchanged, if this row continues the same block; or the latest row's
                block_id incremented by one, if this row begins a new block.
        """
        if self._latest_row is None:
            return 0

        new_ts = self._parse_timestamp(data['state_source_timestamp'])

        last = self._latest_row
        gap_seconds = (new_ts - last['timestamp']).total_seconds()
        gap_triggered = gap_seconds > self.max_gap_seconds
        context_changed = new_context != last['context']

        return last['block_id'] + 1 if (gap_triggered or context_changed) else last['block_id']

    def register_agent(self, agent_id, agent_name, config=None, info=None):
        """
        Register an agent in the database.

        Args:
            agent_id (str): The unique identifier for the agent
            agent_name (str): The name of the agent
            config (dict, optional): Configuration dictionary for the agent. Defaults to empty dict.
            info (dict, optional): Additional information dictionary for the agent. Defaults to empty dict.

        Returns:
            int: Status code (0 for success, 1 for error)

        Raises:
            Exception: If there is an error executing the database query
        """
        try:
            # Set defaults if not provided
            if config is None:
                config = {}
            if info is None:
                info = {
                    'startup_time': time.time(),
                    'status': 'starting'
                }

            # Prepare the data for insertion
            query = """INSERT INTO agent_information
                      (registered_id, agent_name, config, info)
                      VALUES (%s, %s, %s, %s)"""
            values = (
                agent_id,
                agent_name,
                pickle.dumps(config),
                pickle.dumps(info)
            )

            status = self.__execute_and_commit(query, values)

            if status == 0:
                logging.info(f"Agent {agent_id} ({agent_name}) registered successfully in database")
            else:
                logging.error(f"Failed to register agent {agent_id} in database")

            return status

        except Exception as e:
            logging.error(f"Error registering agent {agent_id}: {e}")
            raise e

    def update_agent_info(self, agent_id, info_updates):
        """
        Update agent information in the database.

        Args:
            agent_id (str): The unique identifier for the agent
            info_updates (dict): Dictionary containing the updates to merge with existing info

        Returns:
            int: Status code (0 for success, 1 for error)
        """
        try:
            # First, get the existing info using parameterized query
            query = "SELECT info FROM agent_information WHERE registered_id = %s"
            self.db_cursor.execute(query, (agent_id,))
            results = self.db_cursor.fetchall()

            if not results:
                logging.error(f"Agent {agent_id} not found in database")
                return 1

            # Deserialize existing info
            existing_info = pickle.loads(results[0]['info']) if results[0]['info'] else {}

            # Merge with updates
            existing_info.update(info_updates)

            # Update the database
            update_query = "UPDATE agent_information SET info = %s WHERE registered_id = %s"
            values = (pickle.dumps(existing_info), agent_id)

            status = self.__execute_and_commit(update_query, values)

            if status == 0:
                logging.info(f"Agent {agent_id} info updated successfully")
            else:
                logging.error(f"Failed to update agent {agent_id} info")

            return status

        except Exception as e:
            logging.error(f"Error updating agent {agent_id} info: {e}")
            raise e

    def get_agent_info(self, agent_id):
        """
        Retrieve agent information from the database.

        Args:
            agent_id (str): The unique identifier for the agent

        Returns:
            dict: Agent information including config and info, or None if not found
        """
        try:
            query = "SELECT * FROM agent_information WHERE registered_id = %s"
            self.db_cursor.execute(query, (agent_id,))
            result = self.db_cursor.fetchone()

            if not result:
                logging.warning(f"Agent {agent_id} not found in database")
                return None

            # Deserialize pickled data
            agent_info = {
                'id': result['id'],
                'registered_id': result['registered_id'],
                'agent_name': result['agent_name'],
                'config': pickle.loads(result['config']) if result['config'] else {},
                'info': pickle.loads(result['info']) if result['info'] else {}
            }

            return agent_info

        except Exception as e:
            logging.error(f"Error retrieving agent {agent_id} info: {e}")
            raise e

    def parse_results(self, results):
        """
        Parses the results from the database query, converting any decimal.Decimal or bytes types to appropriate formats.
        This function is used to ensure that the data retrieved from the database is in a format that can be easily used in Python, such as converting decimal.Decimal to float and bytes to numpy arrays or unpickled objects.
        Args:
            results (list): A list of dictionaries containing the results from the database query.
        Returns:
            np.ndarray: A numpy array containing the parsed results, with decimal.Decimal converted to float and bytes converted to numpy arrays or unpickled objects.
        Raises:
            None: If the results are empty or if there are no decimal.Decimal or bytes types in the results, it simply returns an empty numpy array.
        Example:
            >>> results = [{'id': 1, 'value': decimal.Decimal('3.14'), 'data': b'\\x00\\x01\\x02\\x03'}]
            >>> parsed_results = self.parse_results(results)
            >>> logging.debug(parsed_results)
            [{'id': 1, 'value': 3.14, 'data': array([0., 1., 2., 3.])}]
        """
        parsed_results = []
        for result in results:
            for key in result:
                if type(result[key]) is decimal.Decimal:
                    result[key] = float(result[key])
                if type(result[key]) is bytes:
                    try:
                        result[key] = np.frombuffer(result[key], dtype=np.float32)
                    except ValueError:
                        try:
                            result[key] = pickle.loads(result[key])
                        except pickle.UnpicklingError:
                            logging.debug(f"Could not unpickle data for key: {key}")

            parsed_results.append(result)
        return np.array(parsed_results)

    def get_timestamps(self, window_size, mode="random", n=1, agent_type="diagnostics",
                        equality_filter=None, min_timestamp=None):
        """
        Retrieves a list of timestamps from the database that have at least `window_size` number of records.

        Args:
            window_size (int): The minimum number of records required for each timestamp.
            mode (str): The order in which candidate timestamps are to be sampled,
                either "random" or "latest". Defaults to "random". This is a
                lower-level, internal ordering primitive used by sample_batch (by
                way of _collect_windows_for_block); it is unrelated to
                sample_batch's own sampling_strategy argument.
            n (int): The number of timestamps to retrieve. Defaults to 1.
            agent_type (str): The type of agent - "controls" or "diagnostics". Defaults to "diagnostics".
            equality_filter (dict, optional): A mapping from column name to the
                exact value that column must hold - for example, {"block_id": 3}.
                When supplied, only timestamps belonging to rows matching every
                entry in this mapping exactly are considered as candidates. This
                parameter applies to diagnostics agents only, and has no effect
                when agent_type is "controls".
            min_timestamp (datetime, optional): When supplied, only timestamps at
                or after this value are considered as candidates. This parameter
                applies to diagnostics agents only, and has no effect when
                agent_type is "controls".

        Returns:
            list: A list of dictionaries containing the timestamps that meet the criteria.
        """

        if agent_type.lower() == "controls":
            # Control agents require data in both agent_inferences and agent_replay
            query = f"""
            SELECT ai.state_source_timestamp
            FROM agent_inferences ai
            WHERE EXISTS (
                SELECT 1
                FROM agent_replay ar
                WHERE ar.state_id = ai.id
            )
            AND (
                SELECT COUNT(*)
                FROM agent_inferences ai2
                WHERE ai2.state_source_timestamp >= ai.state_source_timestamp
            ) >= {window_size}
            """

            if mode.lower() == "random":
                query += " ORDER BY RAND()"
            elif mode.lower() == "latest":
                query += " ORDER BY ai.state_source_timestamp DESC"
            else:
                logging.debug("mode not understood in get_timestamps function...")
                return None

        else:  # diagnostics
            # Diagnostic agents only need data in agent_inferences. The inner count
            # is scoped to ai2.block_id = agent_inferences.block_id - that is, it
            # only counts rows belonging to the same block as the candidate row
            # itself - because a candidate is only actually usable as a window's
            # starting point if enough rows remain within its own block to fill
            # that window; sample_window rejects any window that crosses into a
            # different block_id regardless of how much data exists further along
            # in the table as a whole. Scoping this check by block_id up front
            # means every candidate this query returns is guaranteed to produce a
            # valid, block-homogeneous window - see _collect_windows_for_block's
            # docstring for why this exactness matters to sample_batch.
            query = f"""
            SELECT state_source_timestamp
            FROM agent_inferences
            WHERE (
                SELECT COUNT(*)
                FROM agent_inferences ai2
                WHERE ai2.state_source_timestamp >= agent_inferences.state_source_timestamp
                AND ai2.block_id = agent_inferences.block_id
            ) >= {window_size}
            """

            values = []
            if equality_filter:
                for col, val in equality_filter.items():
                    self._validate_identifier(col)
                    query += f" AND `{col}` = %s"
                    values.append(val)
            if min_timestamp is not None:
                query += " AND state_source_timestamp >= %s"
                values.append(min_timestamp)

            if mode.lower() == "random":
                query += " ORDER BY RAND()"
            elif mode.lower() == "latest":
                query += " ORDER BY state_source_timestamp DESC"
            else:
                logging.debug("mode not understood in get_timestamps function...")
                return None

            query += f" LIMIT {n}"
            results = self._DBManager__execute_query(query, values=tuple(values) if values else None)
            return self.parse_results(results)

        query += f" LIMIT {n}"

        results = self._DBManager__execute_query(query)
        parsed_results = self.parse_results(results)
        return parsed_results

    def sample_window(self, window_time_seed, agent_type, window_size):
        """
        Samples a window of data from the database starting from a given timestamp.

        For diagnostics agents, the candidate window is subjected to validation
        before being returned: it is returned to the caller only if it contains
        exactly `window_size` rows, and, in addition, all of those rows share an
        identical block_id, meaning that the window does not span across a block
        boundary. A window failing either of these conditions is considered
        invalid and causes this method to return None instead, so that no caller can
        ever receive a sampled window whose timesteps mix two distinct operating
        contexts or regimes. This validation is applied unconditionally, regardless
        of which sampling mode requested the window.

        Args:
            window_time_seed (str): The starting timestamp for the window.
            agent_type (str): The type of agent for which the window is being sampled.
            window_size (int): The length of the window to be sampled.
        Returns:
            list or None: A list of dictionaries containing the sampled data for the
                specified agent type; or None, if the candidate window proved too
                short, spanned more than one block_id, or a database error occurred
                while retrieving it.
        Example:
            >>> window = self.sample_window(window_time_seed='2023-10-01 12 :00:00', agent_type='diagnostics', window_size=10)
            >>> logging.debug(window)
            [{'state_source_timestamp': '2023-10-01 12:00:00', 'state': array([0.1, 0.2])},
             {'state_source_timestamp': '2023-10-01 12:00:01', 'state': array([0.3, 0.4])},
             ...]
        """
        if agent_type.lower() != "controls":
            query = (f"SELECT state_source_timestamp, state, block_id, context "
                     f"FROM agent_inferences WHERE state_source_timestamp >= '{window_time_seed}' "
                     f"ORDER BY state_source_timestamp LIMIT {window_size}")
        else:
            query = f"SELECT ai.state_source_timestamp, ai.state, ai.prediction, ar.next_state, ar.reward, ar.truncate, ar.terminate FROM agent_inferences ai JOIN agent_replay ar ON ai.Id = ar.state_id WHERE ai.state_source_timestamp >= '{window_time_seed}' ORDER BY ai.state_source_timestamp LIMIT {window_size}"

        try:
            self.db_cursor.execute(query)
            results = self.db_cursor.fetchall()
        except Exception as e:
            logging.debug("DB Error: ", e)
            logging.debug("with query: ", query)
            return None

        parsed_results = self.parse_results(results)

        if agent_type.lower() != "controls":
            if len(parsed_results) != window_size:
                return None
            if len({row['block_id'] for row in parsed_results}) != 1:
                logging.debug(f"sample_window: rejected block-crossing window seeded at {window_time_seed}")
                return None

        return parsed_results

    def check_sample_feasibility(self, window_size, agent_type):
        """
        Checks if there are enough samples in the database to sample a batch of the specified window size.

        Args:
            window_size (int): The length of the window to be sampled.
            agent_type (str): The type of agent for which the samples are being checked.

        Returns:
            bool: True if there are enough samples, False otherwise.

        Raises:
            None: If the agent_type is not recognized, it returns False.
        Example:
            >>> success = self.check_sample_feasibility(window_size=10, agent_type='diagnostics')
            >>> logging.debug(success)
            True
        """
        success = True
        number_of_records_prediction_table = self.get_size(table_name="agent_inferences")
        if number_of_records_prediction_table < window_size:
            logging.debug("Number of records in prediction table is less than window size. Cannot sample batch, waiting for more data to be recorded...")
            success = False

        if agent_type.lower() == "controls":
            number_of_records_replay_table = self.get_size(table_name="agent_replay")
            if number_of_records_replay_table < window_size:
                logging.debug("Number of records in replay table is less than window size. Cannot sample batch, waiting for more data to be recorded...")
                success = False

        return success

    @staticmethod
    def _parse_lookback(value):
        """
        Normalizes sample_batch's sampling_lookback argument into a timedelta.

        Accepts a timedelta directly (returned unchanged), or a string consisting
        of a number immediately followed by a single unit suffix - one of 's'
        (seconds), 'm' (minutes), 'h' (hours), 'd' (days), or 'w' (weeks) - for
        example '24h', '90m', or '3d'. There is no dependency on a third-party
        duration-parsing library (such as pandas.Timedelta) here, since this file
        must remain usable from every agent's container, and pandas is not a
        dependency of every one of them (only the RL control agent's container
        currently installs it).

        Args:
            value: A timedelta, or a string of the form '<number><unit>'.

        Returns:
            timedelta: The equivalent duration.

        Raises:
            ValueError: If value is a string that does not match the expected format.
        """
        if isinstance(value, timedelta):
            return value
        unit_seconds = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400, 'w': 604800}
        match = re.fullmatch(r'(\d+(?:\.\d+)?)([smhdw])', str(value).strip().lower())
        if not match:
            raise ValueError(
                f"Invalid sampling_lookback value: {value!r}. Expected a timedelta, "
                f"or a string like '24h', '90m', '3d' (a number followed by one of "
                f"s/m/h/d/w)."
            )
        amount, unit = match.groups()
        return timedelta(seconds=float(amount) * unit_seconds[unit])

    def _get_latest_timestamp(self):
        """
        Determines the state_source_timestamp of the single most recently
        inserted row of agent_inferences, used by sample_batch to anchor its
        sampling_lookback window to the data's own timeline rather than to
        wall-clock time. Anchoring to the latest row rather than to
        datetime.now() keeps the window meaningful even if ingestion has been
        paused for a while (the window still covers the most recent real data,
        rather than an increasingly stale slice of it), and sidesteps any
        question of whether the sampling and database processes' clocks agree.

        Returns:
            datetime or None: The most recently inserted row's timestamp, or
                None if agent_inferences currently contains no rows at all.
        """
        results = self._DBManager__execute_query(
            "SELECT state_source_timestamp FROM agent_inferences ORDER BY id DESC LIMIT 1"
        )
        if not results:
            return None
        return self._parse_timestamp(self.parse_results(results)[0]['state_source_timestamp'])

    def _get_block_row_counts(self, min_timestamp):
        """
        Determines, in a single query, every distinct block_id value that
        appears among rows of agent_inferences whose state_source_timestamp is
        at or after min_timestamp, together with how many such rows each one
        has - used by sample_batch to decide which blocks to stratify sampling
        across, and exactly how many valid windows each one can supply.

        A row count alone is enough to determine a block's true capacity
        without querying for its windows first: because a block is, by
        construction, a contiguous run of rows sharing one block_id, any
        window_size consecutive rows entirely within it are automatically
        block-homogeneous, so the number of valid starting positions is exactly
        row_count - window_size + 1 (or zero, if row_count is smaller than
        window_size) - see sample_batch, which performs that arithmetic on
        the counts this method returns.

        Args:
            min_timestamp (datetime): The lower bound of the sampling window.

        Returns:
            dict: A mapping from block_id (int) to the number of rows (int)
                that block has at or after min_timestamp. Empty if no rows are
                found in the window at all.
        """
        results = self._DBManager__execute_query(
            "SELECT block_id, COUNT(*) AS row_count FROM agent_inferences "
            "WHERE state_source_timestamp >= %s GROUP BY block_id",
            values=(min_timestamp,)
        )
        return {int(row['block_id']): int(row['row_count']) for row in self.parse_results(results)}

    def _collect_windows_for_block(self, block_id, target_n, window_size, agent_type, mode, min_timestamp):
        """
        Collects exactly target_n valid, block-homogeneous windows belonging
        to a single block_id, with candidate starting timestamps restricted to
        at or after min_timestamp, on behalf of sample_batch's per-block
        sampling.

        A single pool of exactly target_n candidate starting timestamps is
        requested via one call to get_timestamps, ordered according to mode;
        that pool is then walked once, retrieving and validating a window via
        sample_window for each candidate in turn. sample_batch always derives
        target_n from _get_block_row_counts's exact per-block row counts (see
        its own docstring), so target_n never exceeds this block's true number
        of valid windows, and get_timestamps' own candidate-selection query is
        itself scoped to this same block (see its docstring) - so, barring a
        genuine database error, every one of the target_n candidates returned
        is expected to pass sample_window's validation, and this method
        should return exactly target_n windows rather than fewer.

        Args:
            block_id (int): The block whose windows are being collected.
            target_n (int): The number of valid windows to collect. If zero
                or negative, this method returns immediately with an empty list.
            window_size (int): The number of consecutive rows each window
                must contain.
            agent_type (str): The type of agent for which windows are being
                sampled, as passed through to get_timestamps and sample_window.
            mode (str): The candidate-ordering mode to pass through to
                get_timestamps - either "random" or "latest".
            min_timestamp (datetime): The lower bound of the sampling window, as
                passed through to get_timestamps.

        Returns:
            list: target_n valid windows belonging to this block, each itself
                a list of row dictionaries as returned by sample_window,
                ordered consistently with mode (most-recent-first for "latest";
                in the database's own RAND() order for "random"). Only contains
                fewer than target_n windows if the caller requested more than
                this block's true capacity, or a candidate unexpectedly failed
                validation.
        """
        if target_n <= 0:
            return []
        candidates = self.get_timestamps(window_size=window_size,
                                          mode=mode,
                                          n=target_n,
                                          agent_type=agent_type,
                                          equality_filter={"block_id": block_id},
                                          min_timestamp=min_timestamp)
        if len(candidates) == 0:
            return []
        collected = []
        for candidate in candidates:
            window = self.sample_window(window_time_seed=candidate['state_source_timestamp'],
                                         agent_type=agent_type,
                                         window_size=window_size)
            if window is not None:
                collected.append(window)
                if len(collected) >= target_n:
                    break
        return collected

    @staticmethod
    def _allocate_with_redistribution(batch_size, available_counts):
        """
        Distributes batch_size units of quota as evenly as possible across a set
        of blocks, subject to each block's own maximum available count, and
        redistributes whatever quota a block cannot use to the remaining blocks
        that still have room - a water-filling allocation. This is what
        implements sample_batch's redistribution policy: a block unable to
        supply its equal share does not, by itself, cause the returned batch to
        fall short by that amount, as long as some other block has spare
        valid windows to make up the difference.

        Allocation proceeds in rounds: each round divides whatever quota
        remains evenly across the blocks still short of their available count,
        removes any block that reaches its own count from further
        consideration, and repeats. This terminates within at most
        len(available_counts) rounds, since every round both makes progress
        (allocates at least one unit per active block) and either exhausts the
        remaining quota or removes at least one block from consideration.

        Within each round, blocks are always visited in descending block_id
        order - largest block_id first. This matters whenever a round's quota
        does not divide evenly across the still-active blocks (share =
        remaining // len(active) can leave a remainder, and a batch_size of 1
        is the extreme case of this: share is forced up to 1 by the max(1, ...)
        below, but only one block can actually receive it): whichever block is
        visited first within that round claims the leftover unit(s), so
        visiting largest-to-smallest means that leftover always goes to the
        largest block_id still in contention, rather than to whichever block_id
        happened to occupy an arbitrary position in available_counts's own
        iteration order (available_counts is ultimately built from a GROUP BY
        query with no ORDER BY - see _get_block_row_counts - so that order
        cannot itself be relied on to prioritize anything). This ordering is
        unrelated to, and unaffected by, sample_batch's own sampling_strategy
        argument, which only controls how windows are chosen *within* a single
        already-allocated block, never the order in which different block_ids
        are allocated to.

        Args:
            batch_size (int): The total quota to distribute.
            available_counts (dict): A mapping from block_id to the number of
                valid windows actually available for that block - each
                block's maximum possible allocation.

        Returns:
            dict: A mapping from block_id to the number of windows allocated
                to it. Every value is at most that block's entry in
                available_counts, and the values sum to
                min(batch_size, sum(available_counts.values())).
        """
        alloc = {b: 0 for b in available_counts}
        remaining = batch_size
        active = sorted([b for b, n in available_counts.items() if n > 0], reverse=True)
        while remaining > 0 and active:
            share = max(1, remaining // len(active))
            progressed = False
            for b in list(active):
                room = available_counts[b] - alloc[b]
                take = min(share, room, remaining)
                if take > 0:
                    alloc[b] += take
                    remaining -= take
                    progressed = True
                if alloc[b] >= available_counts[b]:
                    active.remove(b)
                if remaining <= 0:
                    break
            if not progressed:
                break
        return alloc

    def _build_batch_dict(self, windows):
        """
        Assembles a list of previously collected, individually valid windows into
        the single dictionary of numpy arrays that sample_batch returns to its
        caller, for the diagnostics case.

        The resulting dictionary contains the key 'state_source_timestamp', an
        array of shape (batch_size_eff, window_size); the key 'state', an array of shape
        (batch_size_eff, window_size, n_input_channels), containing input channel values
        exclusively; and the key 'context', an array of shape
        (batch_size_eff, window_size, n_context_channels) if this agent's rows carry a
        context value, or an array of Nones of shape (batch_size_eff, window_size)
        otherwise - callers with no notion of context (for example, the plain
        autoencoder) simply never look at this key.

        Args:
            windows (list): A list of windows, each itself a list of row
                dictionaries of the form returned by sample_window, all sharing
                the same window_size.

        Returns:
            dict: A dictionary of numpy arrays, keyed as described above, with
                'batch_size_eff' in every shape description referring to len(windows).
        """
        keys = ['state_source_timestamp', 'state', 'context']
        batch = {k: [] for k in keys}
        for window in windows:
            for k in keys:
                batch[k].append([row[k] for row in window])
        for k in batch:
            batch[k] = np.array(batch[k])
        return batch

    def _sample_batch_controls(self, batch_size, window_size):
        """Legacy controls sampling path - unchanged, out of scope for context/block_id support."""
        mode = "random"
        batch = {'state_source_timestamp': [],
                 'state': [],
                 'prediction': [],
                 'next_state': [],
                 'reward': [],
                 'terminate': [],
                 'truncate': []}

        required_samples = batch_size

        while required_samples > 0:
            timestamps = self.get_timestamps(window_size=window_size,
                                            mode=mode,
                                            n=required_samples,
                                            agent_type="controls")
            for result in timestamps:
                window_seed = result['state_source_timestamp']
                results = self.sample_window(window_time_seed=window_seed,
                                              agent_type="controls",
                                              window_size=window_size)

                if results is None:
                    raise ValueError("No results found for the given window seed.")

                for key in batch:
                    batch[key].append([results[i][key] for i in range(len(results))])
            key = list(batch.keys())[0]
            required_samples = required_samples - len(batch[key])

        for key in batch:
            batch[key] = np.array(batch[key])

        return batch

    def sample_batch(self, batch_size, window_size, agent_type,
                      sampling_lookback="24h", sampling_strategy="latest"):
        """
        Samples a batch of block-homogeneous windows from the database,
        stratified equally across every block that occurred within a recent
        lookback window.

        Every returned window is validated to be block-homogeneous (see
        sample_window), and every window's starting timestamp is restricted
        to the most recent sampling_lookback duration, measured back from the
        timestamp of the most recently inserted row - not wall-clock time, see
        _get_latest_timestamp. Within that window, sampling is stratified
        equally across every distinct block_id present, in three steps. First,
        _get_block_row_counts determines, via a single grouped query, exactly
        how many rows each block has in the window; since a block is a
        contiguous run of same-block_id rows by construction, this row count
        alone is enough to compute that block's exact number of valid
        window_size-row windows (row_count - window_size + 1, or zero),
        with no need to query for candidate windows merely to discover
        availability. Second, those exact counts are passed to
        _allocate_with_redistribution, which divides batch_size as evenly as
        possible across however many blocks are found, redistributing whatever
        share a block cannot use - for instance, because it only just began, or
        was itself too short to contain many valid windows - to the other
        blocks that do have room for it, via a water-filling allocation. Ties
        and remainders in that division always favor larger block_id values
        first (see _allocate_with_redistribution's docstring) - for example, a
        batch_size of 1 is always drawn from the single largest block_id
        present, regardless of sampling_strategy, which only governs how
        windows are chosen *within* a block, not the order blocks themselves
        are considered in. Third,
        exactly the allocated number of windows is fetched from each block via
        _collect_windows_for_block; because get_timestamps' own candidate
        query is itself scoped to a single block (see its docstring), this
        fetch is exact rather than exploratory - it neither wastes candidates
        that would only be rejected downstream, nor needs a second round to
        make up an unexpected shortfall. Only if the total number of valid
        windows available across every block in the window falls short of
        batch_size in the first place does the returned batch itself fall short
        (logged at INFO).

        Args:
            batch_size (int): The number of samples to be included in the batch.
            window_size (int): The length of each window to be sampled.
            agent_type (str): The type of agent for which the batch is being
                sampled. Valid values are 'controls' or 'diagnostics'. Everything
                described above applies to 'diagnostics' only; 'controls' is an
                unrelated, unmodified legacy sampling path - see
                _sample_batch_controls.
            sampling_lookback (timedelta or str): How far back from the most
                recent row's timestamp the sampling window extends. Accepts a
                timedelta directly, or a string such as '24h', '90m', '3d' - see
                _parse_lookback. Defaults to 24 hours.
            sampling_strategy (str): The order in which each block's candidate
                windows are considered - 'latest' (the block's most recent
                valid windows) or 'random' (a uniformly random selection from
                the block's valid windows). Defaults to 'latest'.

        Returns:
            dict or None: None if agent_type or sampling_strategy is invalid, or
            if there isn't enough data at all (see check_sample_feasibility).
            Otherwise a dict with 'state_source_timestamp', 'state', 'context'
            (diagnostics - see _build_batch_dict), or the controls-specific keys
            ('prediction', 'next_state', 'reward', 'terminate', 'truncate') for
            controls.
        """
        if agent_type.lower() not in ["controls", "diagnostics"]:
            logging.debug(f"Invalid agent_type: {agent_type}. Valid values are 'controls' or 'diagnostics'.")
            return None

        if not self.check_sample_feasibility(window_size, agent_type):
            logging.debug("Not enough samples in the database to sample a batch.")
            return None

        if agent_type.lower() == "controls":
            return self._sample_batch_controls(batch_size, window_size)

        if sampling_strategy.lower() not in ("latest", "random"):
            logging.debug(f"Invalid sampling_strategy: {sampling_strategy}. Valid values are 'latest' or 'random'.")
            return None

        lookback_td = self._parse_lookback(sampling_lookback)
        # check_sample_feasibility above already confirmed at least window_size
        # rows exist, so agent_inferences is guaranteed non-empty here.
        min_timestamp = self._get_latest_timestamp() - lookback_td

        # Step 1: determine every block's exact capacity from row counts alone -
        # see _get_block_row_counts's docstring for why a count is sufficient,
        # with no need to fetch candidate windows merely to learn availability.
        row_counts = self._get_block_row_counts(min_timestamp)
        if not row_counts:
            logging.info(f"sample_batch: no blocks found within the last {lookback_td} of data; returning empty batch")
            return self._build_batch_dict([])
        available_counts = {block_id: max(0, count - window_size + 1)
                             for block_id, count in row_counts.items()}

        # Step 2: allocate batch_size across blocks in a single pass, using
        # those exact counts - no exploratory request-then-redistribute rounds
        # are needed, since every block's true ceiling is already known.
        alloc = self._allocate_with_redistribution(batch_size, available_counts)

        # Step 3: fetch exactly the allocated number of windows from each
        # block - see _collect_windows_for_block's docstring for why this
        # fetch is expected to return exactly what was asked for, rather than
        # needing to over-request as a safety margin.
        windows = []
        for block_id, n in alloc.items():
            windows.extend(self._collect_windows_for_block(block_id=block_id,
                                                             target_n=n,
                                                             window_size=window_size,
                                                             agent_type=agent_type,
                                                             mode=sampling_strategy.lower(),
                                                             min_timestamp=min_timestamp))

        if len(windows) < batch_size:
            logging.info(f"sample_batch: requested batch_size={batch_size} but only found "
                         f"{len(windows)} valid block-homogeneous windows across "
                         f"{len(row_counts)} block(s) within the last {lookback_td}; returning partial batch")

        return self._build_batch_dict(windows)

    def record_sensor_data(self, data):
        """
        Records sensor data into the database.

        block_id is computed and injected here (never supplied by the caller) by comparing
        this row's timestamp/context value against the latest previously written row -
        see _compute_block_id. Callers (ingest threads) stay fully agnostic of block_id.

        Args:
            data (dict): A dictionary containing the sensor data to be stored. The
                keys should match the columns in the `agent_inferences` table. An
                agent with a notion of context should include a 'context' entry
                here, as a numpy array; an agent with none should simply omit it
                (agent_inferences' context column then stores NULL for this row,
                and this row can never trigger a context-based block_id advance).

        Returns:
            int: The status of the operation, 0 if successful, 1 if there was an error.

        Raises:
            AssertionError: If the data is not a dictionary or if it is empty.
        """
        assert isinstance(data, dict), "Data must be a dictionary"
        if len(data) == 0:
            logging.debug("No data to store, exiting...")
            return 0

        # Captured once, here, before the tobytes() conversion below overwrites
        # data['context'] with its serialized form - both _compute_block_id and the
        # latest-row cache update further down need this same plain-tuple form (see
        # _compute_block_id's docstring for why a tuple, rather than the raw numpy
        # array, is what must actually be compared).
        context_value = data.get('context')
        new_context = tuple(context_value) if context_value is not None else None
        data['block_id'] = self._compute_block_id(data, new_context)

        query = f"INSERT INTO agent_inferences "
        query_columns = "("
        query_values = "("
        values = []
        for key in data:
            if key in ['state', 'prediction', 'context'] and data[key] is not None:
                data[key] = data[key].tobytes()

            query_columns+= f"{key}, "
            query_values += f"%s, "
            values.append(data[key])
        query += query_columns[:-2] + ") VALUES " + query_values[:-2] + ")"

        status = self.__execute_and_commit(query, values=tuple(values))

        if status == 0:
            try:
                self._latest_row = {
                    'timestamp': self._parse_timestamp(data['state_source_timestamp']),
                    'context': new_context,
                    'block_id': data['block_id'],
                }
            except (KeyError, ValueError) as e:
                logging.warning(f"DBManager: could not update latest-row cache after write: {e}")

        return status

    def get_state_id(self, source_timestamp):
        """
        Retrieves the state ID based on the source timestamp.

        Args:
            source_timestamp (str): The source timestamp to search for.

        Returns:
            int: The state ID corresponding to the given source timestamp.
        """
        query = f"SELECT id FROM agent_inferences WHERE state_source_timestamp = '{source_timestamp}'"
        results = self.__execute_query(query)

        if len(results) == 0:
            logging.debug(f"No state found for source timestamp: {source_timestamp}")
            return None
        elif len(results) > 1:
            logging.debug(f"Multiple states found for source timestamp: {source_timestamp}, returning the first one.")
            return [int(results[i]['id']) for i in range(len(results))]

        return int(results[0]['id'])

    def record_prediction(self, prediction, prediction_timestamp, key_value, key="state_source_timestamp"):
        """
        Records a prediction in the database.

        Args:
            prediction (np.ndarray): The prediction data to be stored.
            prediction_timestamp (str): The timestamp of the prediction.
            key_value (str): The value of the key to identify the record to update.
            key (str): The key to identify the record, either 'state_source_timestamp' or 'state_id'. Defaults to 'state_source_timestamp'.

        Returns:
            int: The status of the operation, 0 if successful, 1 if there was an error.
        Raises:
            AssertionError: If the prediction is not a numpy array or if the key is not one of 'state_source_timestamp' or 'state_id'.
        """
        assert isinstance(prediction, np.ndarray), "Prediction must be a numpy array"
        assert key in ['state_source_timestamp', 'state_id'], f"Key must be one of 'state_source_timestamp' or 'state_id', got {key}"

        query = f"UPDATE agent_inferences set prediction = %s, prediction_timestamp = '{prediction_timestamp}' WHERE {key} = %s"
        values = (prediction.tobytes(), key_value)

        status = self.__execute_and_commit(query, values=values)

        return status

    def record_controls_tuple(self, data, state_id):
        """
        Records a controls tuple in the database.

        Args:
            data (dict): A dictionary containing the controls tuple data to be stored.
                         It must contain keys such as 'next_state', 'reward', 'terminate', and 'truncate'.
            state_id (int): The ID of the state to which this controls tuple belongs.

        Returns:
            int: The status of the operation, 0 if successful, 1 if there was an error.

        Raises:
            AssertionError: If the data is not a dictionary or if it does not contain the required keys.
            AssertionError: If the state_id is None, indicating that the controls tuple cannot be stored without a valid state ID.
        """
        assert isinstance(data, dict), "controls tuple must be passed as a dictionary"
        assert 'next_state' in data, "controls tuple must contain 'next_state'"
        assert 'reward' in data, "controls tuple must contain 'reward'"
        assert 'terminate' in data, "controls tuple must terminate"
        assert 'truncate' in data, "controls tuple must contain truncate"

        if state_id is None:
            logging.debug("State ID is None. Cannot store controls tuple.")
            return 1

        query = f"INSERT INTO agent_replay (state_id, "
        query_columns = ""
        query_values = f"({state_id}, "
        values = []
        for key in data:
            if key in ['reward', 'next_state']:
                data[key] = data[key].tobytes()
            elif key in ['info']:
                data[key] = pickle.dumps(data[key])

            query_columns+= f"{key}, "
            query_values += f"%s, "
            values.append(data[key])

        query += query_columns[:-2] + ") VALUES " + query_values[:-2] + ")"


        status = self.__execute_and_commit(query, values=tuple(values))

        return status

    def get_size(self, table_name):
        """
        Returns the number of records in the specified table.

        Args:
            table_name (str): The name of the table for which the record count is to be retrieved.

        Returns:
            int: The number of records in the specified table.
        """
        query = f"SELECT COUNT(*) FROM {table_name}"
        self.db_cursor.execute(query)
        rowcount = self.db_cursor.fetchone()
        return rowcount['COUNT(*)']

    def close(self):
        """
        Closes the database cursor and connection, terminating the current session.

        This method is used to gracefully close the database connection and cursor
        once all operations are complete. It ensures that resources are released properly.

        Args:
            None

        Returns:
            None

        Example:
            >>> db_connector.close()
            closing cursor
            closing connection
            db connection closed
        """
        logging.info("closing cursor")
        self.db_cursor.close()
        logging.info("closing connection")
        self.mydb.close()
        logging.info("db connection closed")


# DOCUMENTATION: notifies of database error if connection fails
