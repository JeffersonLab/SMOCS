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
from datetime import datetime

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

        # The set of column names, on the agent_inferences table, whose values are
        # used to stratify and group sampling - for example, columns derived from
        # switch-state or operating-mode channels. This defaults to an empty list, so
        # that agents which have no notion of context are entirely unaffected by this
        # mechanism. As described further in _compute_block_id, below, block_id
        # advances to a new value whenever any of these columns' values changes
        # between one row and the next, in addition to advancing whenever the gap
        # between consecutive timestamps exceeds max_gap_seconds.
        self.context_cols = db_cfg_dict.get('context_cols', [])

        # The maximum permitted gap, in seconds, between the timestamps of two
        # consecutively written rows before a new block_id is begun. If the calling
        # agent's configuration does not supply this value, it defaults to positive
        # infinity, which has the effect of disabling gap-based block splitting
        # entirely - under that condition, a new block can then only ever be started
        # by a change in one of the context_cols values described above.
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

    def create_tables(self):
        """
        Creates the necessary tables in the database if they do not already exist.
        This method is called during the initialization of the DBManager to ensure that the required tables
        for storing agent information, inferences, and replay data are present in the database.
        It executes SQL queries to create the following tables:
            - agent_information: Stores information about the agents.
            - agent_inferences: Stores the inferences made by the agents.
            - agent_replay: Stores the replay data for the agents.

        These tables are created within AGENT_DATABASE_NAME - the single, fixed
        database that connect() connects to, and which init.sql independently
        provisions when the MySQL server itself is first started. This method does
        not, itself, create or switch to any database; it only creates tables within
        whichever database the existing connection is already using.
        """
        query = f"CREATE TABLE IF NOT EXISTS agent_information (id INT AUTO_INCREMENT PRIMARY KEY, registered_id VARCHAR(50) NOT NULL, agent_name VARCHAR(50), config BLOB, info BLOB)"
        self.__execute_and_commit(query)

        query = """
        CREATE TABLE IF NOT EXISTS agent_inferences (id INT AUTO_INCREMENT PRIMARY KEY, 
                                                                state_source_timestamp DATETIME(6) NOT NULL, 
                                                                state_received_timestamp DATETIME(6) NOT NULL,
                                                                state BLOB NOT NULL, 
                                                                prediction_timestamp DATETIME(6), 
                                                                prediction BLOB,
                                                                block_id INT NOT NULL DEFAULT 0)
        """
        self.__execute_and_commit(query)

        query = """
        CREATE TABLE IF NOT EXISTS agent_replay (
                                                id INT AUTO_INCREMENT PRIMARY KEY,
                                                state_id INT NOT NULL,
                                                action_success BOOL,
                                                reward FLOAT NOT NULL,
                                                next_state_source_timestamp DATETIME(6) NOT NULL,
                                                next_state_received_timestamp DATETIME(6) NOT NULL,
                                                next_state BLOB NOT NULL,
                                                terminate BOOL NOT NULL,
                                                truncate BOOL NOT NULL,
                                                info, BLOB,
                                                FOREIGN KEY (state_id) REFERENCES agent_inferences(id)
                                            )
        """
        self.__execute_and_commit(query)

        # Perform an additive, idempotent migration to ensure that agent_inferences
        # also has a column for each of this agent's configured context_cols. Note
        # that block_id itself is not handled by this migration step; it is declared
        # directly within the CREATE TABLE statement above, since, unlike the
        # context columns, it is not specific to any individual agent - see
        # _migrate_inferences_schema's docstring, for the complete explanation of
        # this distinction.
        self._migrate_inferences_schema()

    def _get_existing_columns(self, table_name):
        """
        Determines which columns currently exist on the given table by querying
        INFORMATION_SCHEMA.COLUMNS, MySQL's built-in metadata table describing the
        structure of every table in every database the connected user can see.

        INFORMATION_SCHEMA is queried directly here, rather than relying on the
        `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` syntax, because that syntax's
        availability and exact behavior differ across MySQL versions; querying
        INFORMATION_SCHEMA and then conditionally issuing a plain `ADD COLUMN`
        achieves the same idempotent effect in a manner that is portable across
        versions.

        Args:
            table_name (str): The name of the table whose columns are to be listed.

        Returns:
            set: The set of column names currently present on the given table.
        """
        query = "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s"
        self.db_cursor.execute(query, (self.mydb.database, table_name))
        return {row['COLUMN_NAME'] for row in self.db_cursor.fetchall()}

    @staticmethod
    def _validate_identifier(name):
        """
        Rejects a proposed context column name if it contains a backtick character.

        Because column names cannot be supplied as parameterized query values in the
        way that ordinary data values can, every context column name that appears in
        this file is instead interpolated directly into the text of a SQL statement,
        conventionally enclosed within a pair of backticks (for example, as
        `` `{name}` ``). A backtick occurring within the name itself would prematurely
        terminate that quoting and allow whatever text follows it to be interpreted
        as additional, unintended SQL, rather than as part of the identifier. This
        check exists solely to close off that possibility; no other character
        restriction is imposed, since backtick-quoted identifiers in MySQL may
        otherwise contain a wide range of characters (including, for instance, the
        periods commonly found in PV-style channel names) without issue.

        Args:
            name (str): The proposed context column name to validate.

        Raises:
            ValueError: If `name` contains one or more backtick characters.
        """
        if '`' in name:
            raise ValueError(f"Invalid context column name (contains a backtick): {name!r}")

    def _migrate_inferences_schema(self):
        """
        Ensures that agent_inferences has one column corresponding to each entry in
        self.context_cols, adding whichever such columns are not yet present. This
        migration is both additive (it only ever adds columns, never removes or
        alters existing ones) and idempotent (repeated invocations against a table
        that already has every required column have no further effect), and is
        therefore safe to invoke unconditionally every time a DBManager instance
        establishes its connection.

        Note that block_id is deliberately not among the columns handled by this
        method. Unlike the context columns, block_id is a single column whose name
        and type are identical for every agent, and it is therefore declared
        directly within the CREATE TABLE statement in create_tables(), above, rather
        than being added dynamically here. The context columns cannot be declared
        that way, for two reasons: first, which columns are required varies from one
        agent's configuration to another, and second, the set of required columns
        for a single agent may grow over that agent's lifetime - for instance, if an
        additional entry is later added to that agent's context_channels
        configuration. This dynamic, idempotent "add whatever is currently missing"
        approach is what accommodates both of those forms of variability.
        """
        existing = self._get_existing_columns("agent_inferences")

        for col in self.context_cols:
            self._validate_identifier(col)
            if col not in existing:
                logging.info(f"DBManager: adding context column `{col}` to agent_inferences")
                self.__execute_and_commit(f"ALTER TABLE agent_inferences ADD COLUMN `{col}` FLOAT")

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
                that row's context_cols values, in self.context_cols order), and
                'block_id', describing the most recently inserted row; or None if
                agent_inferences currently contains no rows at all.
        """
        context_select = "".join(f", `{c}`" for c in self.context_cols)
        query = f"SELECT state_source_timestamp, block_id{context_select} FROM agent_inferences ORDER BY id DESC LIMIT 1"
        results = self._DBManager__execute_query(query)
        if not results:
            return None
        row = self.parse_results(results)[0]
        return {
            'timestamp': self._parse_timestamp(row['state_source_timestamp']),
            'context': tuple(row[c] for c in self.context_cols),
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

    def _compute_block_id(self, data):
        """
        Determines the block_id that should be assigned to a new row about to be
        inserted, by comparing that row's timestamp and context values against those
        of the most recently written row, as recorded in self._latest_row.

        A new block is begun - that is, the returned block_id is one greater than
        that of the latest row - under either of two conditions: first, if the gap
        between the new row's timestamp and the latest row's timestamp exceeds
        self.max_gap_seconds, which is taken to indicate that data collection was
        interrupted in the interim; or second, if the new row's context values
        differ, under exact equality, from the latest row's context values, which is
        taken to indicate that the agent's operating context has changed. Context
        values are compared using exact equality, rather than any numerical
        tolerance, because they are assumed to already be categorical or discretized
        by the time they reach the database. If neither condition holds, the new row
        is considered to continue the same block as the latest row, and the latest
        row's block_id is returned unchanged.

        Args:
            data (dict): The row about to be inserted. This dictionary must contain
                a 'state_source_timestamp' entry, and, if any context columns are
                configured via self.context_cols, an entry for each of those columns
                as well.

        Returns:
            int: The block_id to be assigned to this row - either zero, if no prior
                row exists yet for this agent; the latest row's existing block_id
                unchanged, if this row continues the same block; or the latest row's
                block_id incremented by one, if this row begins a new block.
        """
        if self._latest_row is None:
            return 0

        new_ts = self._parse_timestamp(data['state_source_timestamp'])
        new_context = tuple(data.get(c) for c in self.context_cols)

        if self.context_cols and any(data.get(c) is None for c in self.context_cols):
            missing = [c for c in self.context_cols if data.get(c) is None]
            logging.warning(f"DBManager: record_sensor_data missing configured context_cols: {missing}")

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
            >>> results = [{'id': 1, 'value': decimal.Decimal('3.14'), 'data': b'\x00\x01\x02\x03'}]
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

    def get_timestamps(self, window_size, mode="random", n=1, agent_type="diagnostics", context_filter=None):
        """
        Retrieves a list of timestamps from the database that have at least `window_size` number of records.

        Args:
            window_size (int): The minimum number of records required for each timestamp.
            mode (str): The order in which candidate timestamps are to be sampled,
                either "random" or "latest". Defaults to "random". Note that this
                parameter governs only the ordering applied within this method; it
                is a lower-level, internal primitive employed by sample_batch's
                stratified-sampling logic, and is distinct from, and should not be
                confused with, sample_batch's own, higher-level mode argument.
            n (int): The number of timestamps to retrieve. Defaults to 1.
            agent_type (str): The type of agent - "controls" or "diagnostics". Defaults to "diagnostics".
            context_filter (dict, optional): A mapping from context column name to
                the exact value that column must hold. When supplied, only
                timestamps belonging to rows whose context_cols values match this
                mapping exactly are considered as candidates. This parameter applies
                to diagnostics agents only, and has no effect when agent_type is
                "controls".

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
            # Diagnostic agents only need data in agent_inferences
            query = f"""
            SELECT state_source_timestamp
            FROM agent_inferences
            WHERE (
                SELECT COUNT(*)
                FROM agent_inferences ai2
                WHERE ai2.state_source_timestamp >= agent_inferences.state_source_timestamp
            ) >= {window_size}
            """

            values = []
            if context_filter:
                for col, val in context_filter.items():
                    self._validate_identifier(col)
                    query += f" AND `{col}` = %s"
                    values.append(val)

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
      
    def sample_sequence(self, window_time_seed, agent_type, segment_length):
        """
        Samples a sequence of data from the database starting from a given timestamp.

        For diagnostics agents, the candidate sequence is subjected to validation
        before being returned: it is returned to the caller only if it contains
        exactly `segment_length` rows, and, in addition, all of those rows share an
        identical block_id, meaning that the sequence does not span across a block
        boundary. A sequence failing either of these conditions is considered
        invalid and causes this method to return None instead, so that no caller can
        ever receive a sampled window whose timesteps mix two distinct operating
        contexts or regimes. This validation is applied unconditionally, regardless
        of which sampling mode requested the sequence.

        Args:
            window_time_seed (str): The starting timestamp for the sequence.
            agent_type (str): The type of agent for which the sequence is being sampled.
            segment_length (int): The length of the segment to be sampled.
        Returns:
            list or None: A list of dictionaries containing the sampled data for the
                specified agent type; or None, if the candidate sequence proved too
                short, spanned more than one block_id, or a database error occurred
                while retrieving it.
        Example:
            >>> sequence = self.sample_sequence(window_time_seed='2023-10-01 12 :00:00', agent_type='diagnostics', segment_length=10)
            >>> logging.debug(sequence)
            [{'state_source_timestamp': '2023-10-01 12:00:00', 'state': array([0.1, 0.2])},
             {'state_source_timestamp': '2023-10-01 12:00:01', 'state': array([0.3, 0.4])},
             ...]
        """
        if agent_type.lower() != "controls":
            context_select = "".join(f", `{c}`" for c in self.context_cols)
            query = (f"SELECT state_source_timestamp, state, block_id{context_select} "
                     f"FROM agent_inferences WHERE state_source_timestamp >= '{window_time_seed}' "
                     f"ORDER BY state_source_timestamp LIMIT {segment_length}")
        else:
            query = f"SELECT ai.state_source_timestamp, ai.state, ai.prediction, ar.next_state, ar.reward, ar.truncate, ar.terminate FROM agent_inferences ai JOIN agent_replay ar ON ai.Id = ar.state_id WHERE ai.state_source_timestamp >= '{window_time_seed}' ORDER BY ai.state_source_timestamp LIMIT {segment_length}"

        try:
            self.db_cursor.execute(query)
            results = self.db_cursor.fetchall()
        except Exception as e:
            logging.debug("DB Error: ", e)
            logging.debug("with query: ", query)
            return None

        parsed_results = self.parse_results(results)

        if agent_type.lower() != "controls":
            if len(parsed_results) != segment_length:
                return None
            if len({row['block_id'] for row in parsed_results}) != 1:
                logging.debug(f"sample_sequence: rejected block-crossing window seeded at {window_time_seed}")
                return None

        return parsed_results
    
    def check_sample_feasibility(self, segment_length, agent_type):
        """
        Checks if there are enough samples in the database to sample a batch of the specified segment length.

        Args:
            segment_length (int): The length of the segment to be sampled.
            agent_type (str): The type of agent for which the samples are being checked.

        Returns:
            bool: True if there are enough samples, False otherwise.

        Raises:
            None: If the agent_type is not recognized, it returns False.
        Example:
            >>> success = self.check_sample_feasibility(segment_length=10, agent_type='diagnostics')
            >>> logging.debug(success)
            True
        """
        success = True
        number_of_records_prediction_table = self.get_size(table_name="agent_inferences")
        if number_of_records_prediction_table < segment_length:
            logging.debug("Number of records in prediction table is less than segment length. Cannot sample batch, waiting for more data to be recorded...")
            success = False
        
        if agent_type.lower() == "controls":
            number_of_records_replay_table = self.get_size(table_name="agent_replay")
            if number_of_records_replay_table < segment_length:
                logging.debug("Number of records in replay table is less than segment length. Cannot sample batch, waiting for more data to be recorded...")
                success = False
        
        return success
 
    def _get_distinct_context_tuples(self):
        """
        Determines every distinct combination of context_cols values that currently
        appears among the rows of agent_inferences, used by _resolve_stratified_groups
        to implement the stratified_groups="all" case of sample_batch.

        Returns:
            list: A list of tuples, each of the same length as self.context_cols and
                in that same column order, one for each distinct combination of
                values observed. Returns an empty list if no context columns are
                configured for this agent at all.
        """
        if not self.context_cols:
            return []
        cols = ", ".join(f"`{c}`" for c in self.context_cols)
        results = self._DBManager__execute_query(f"SELECT DISTINCT {cols} FROM agent_inferences")
        parsed = self.parse_results(results)
        return [tuple(row[c] for c in self.context_cols) for row in parsed]

    def _get_latest_context_tuple(self):
        """
        Determines the context_cols values belonging to the single most recently
        inserted row of agent_inferences, used by _resolve_stratified_groups to
        implement the stratified_groups="latest" case of sample_batch.

        Returns:
            tuple or None: A tuple of context values, of the same length as
                self.context_cols and in that same column order, describing the most
                recent row; or None, if no context columns are configured for this
                agent, or if agent_inferences currently contains no rows at all.
        """
        if not self.context_cols:
            return None
        cols = ", ".join(f"`{c}`" for c in self.context_cols)
        results = self._DBManager__execute_query(f"SELECT {cols} FROM agent_inferences ORDER BY id DESC LIMIT 1")
        if not results:
            return None
        row = self.parse_results(results)[0]
        return tuple(row[c] for c in self.context_cols)

    def _resolve_stratified_groups(self, mode, stratified_groups):
        """
        Interprets sample_batch's mode and stratified_groups arguments and resolves
        them into a single, normalized dictionary mapping each context tuple to be
        sampled to the fraction of the batch that should be drawn from it, with all
        such fractions summing to exactly 1.0.

        When mode is "latest", context is disregarded entirely, and this method
        returns a dictionary containing a single entry, keyed by None, with weight
        1.0; a key of None signifies "impose no context filter" throughout the rest
        of the sampling logic. When mode is "stratified", the interpretation instead
        depends on the value of stratified_groups: the string "all" resolves to an
        equal weight for every context tuple currently present in the data, as
        determined by _get_distinct_context_tuples; the string "latest" resolves to
        a single weight-1.0 entry for whichever context tuple belongs to the most
        recently written row, as determined by _get_latest_context_tuple; and a
        dictionary mapping context tuples directly to explicit weights is validated
        - each tuple's length must equal len(self.context_cols), and the sum of all
        weights must be positive - and then normalized so that its weights sum to
        1.0, with an explanatory message logged at the INFO level whenever
        normalization was actually required (that is, whenever the weights, as
        supplied, did not already sum to 1.0).

        Args:
            mode (str): sample_batch's own mode argument - either "latest" or
                "stratified".
            stratified_groups: sample_batch's own stratified_groups argument - one
                of the string "all", the string "latest", or a dictionary mapping
                context tuples to float weights. This argument is disregarded
                entirely whenever mode is "latest".

        Returns:
            dict: A dictionary mapping each context tuple (or None, signifying no
                context filter) to the fraction, between 0.0 and 1.0, of the batch
                that should be drawn from sequences matching that context, with all
                fractions summing to 1.0.

        Raises:
            ValueError: If stratified_groups is a dictionary containing a tuple
                whose length does not match len(self.context_cols), if its weights
                do not sum to a positive number, or if stratified_groups is none of
                the recognized forms described above.
        """
        if mode.lower() == "latest":
            return {None: 1.0}

        if stratified_groups == "all":
            tuples = self._get_distinct_context_tuples()
            if not tuples:
                return {None: 1.0}
            return {t: 1.0 / len(tuples) for t in tuples}

        if stratified_groups == "latest":
            t = self._get_latest_context_tuple()
            return {t: 1.0} if t is not None else {None: 1.0}

        if isinstance(stratified_groups, dict):
            for key in stratified_groups:
                if len(key) != len(self.context_cols):
                    raise ValueError(
                        f"stratified_groups tuple {key} has length {len(key)}, "
                        f"expected {len(self.context_cols)} (len(context_cols))"
                    )
            total = sum(stratified_groups.values())
            if total <= 0:
                raise ValueError("stratified_groups weights must sum to a positive number")
            if abs(total - 1.0) > 1e-9:
                logging.info(
                    f"sample_batch: stratified_groups percentages summed to {total}, "
                    f"not 1.0 - normalizing automatically to sum to 1.0"
                )
            return {tuple(k): v / total for k, v in stratified_groups.items()}

        raise ValueError(f"Invalid stratified_groups value: {stratified_groups!r}")

    def _collect_group_sequences(self, target_n, segment_length, agent_type, mode, context_filter, candidate_pool_size):
        """
        Collects as many as target_n valid, block-homogeneous sequences matching the
        given context_filter, on behalf of a single group within sample_batch's
        stratified-sampling logic.

        A single, bounded pool of candidate starting timestamps is requested via one
        call to get_timestamps, sized to the larger of candidate_pool_size and
        target_n; that pool is then walked exactly once, retrieving and validating a
        sequence via sample_sequence for each candidate in turn, and stopping as
        soon as target_n valid sequences have been collected. This method
        deliberately never loops indefinitely in search of more candidates: if the
        available candidates are exhausted before target_n valid sequences have
        been found - for instance, because many candidates were rejected by
        sample_sequence's block-homogeneity check - this method simply returns
        however many valid sequences it did manage to collect, which may be fewer
        than target_n, or even zero.

        Args:
            target_n (int): The desired number of valid sequences to collect for
                this group. If zero or negative, this method returns immediately
                with an empty list.
            segment_length (int): The number of consecutive rows each sequence must
                contain.
            agent_type (str): The type of agent for which sequences are being
                sampled, as passed through to get_timestamps and sample_sequence.
            mode (str): The candidate-ordering mode to pass through to
                get_timestamps - either "random" or "latest".
            context_filter (dict or None): The context column values, if any, that
                candidate rows must match exactly, as passed through to
                get_timestamps.
            candidate_pool_size (int): A sizing hint for how many candidate
                timestamps to request from get_timestamps in a single call; the
                actual number requested is the larger of this value and target_n.

        Returns:
            list: A list of as many as target_n valid sequences, each itself a list
                of row dictionaries as returned by sample_sequence. May contain
                fewer than target_n sequences, or be empty, if insufficiently many
                valid candidates were available.
        """
        if target_n <= 0:
            return []
        candidates = self.get_timestamps(window_size=segment_length,
                                          mode=mode,
                                          n=max(candidate_pool_size, target_n),
                                          agent_type=agent_type,
                                          context_filter=context_filter)
        if not candidates:
            return []
        collected = []
        for candidate in candidates:
            seq = self.sample_sequence(window_time_seed=candidate['state_source_timestamp'],
                                        agent_type=agent_type,
                                        segment_length=segment_length)
            if seq is not None:
                collected.append(seq)
                if len(collected) >= target_n:
                    break
        return collected

    def _build_batch_dict(self, sequences):
        """
        Assembles a list of previously collected, individually valid sequences into
        the single dictionary of numpy arrays that sample_batch returns to its
        caller, for the diagnostics case.

        The resulting dictionary contains the key 'state_source_timestamp', an
        array of shape (batch, segment_length); the key 'state', an array of shape
        (batch, segment_length, n_input_channels), containing input channel values
        exclusively; and one further key for each entry in self.context_cols, each
        such array being of shape (batch, segment_length) and containing that
        context column's value at every timestep of every sequence.

        Args:
            sequences (list): A list of sequences, each itself a list of row
                dictionaries of the form returned by sample_sequence, all sharing
                the same segment_length.

        Returns:
            dict: A dictionary of numpy arrays, keyed as described above, with
                'batch' in every shape description referring to len(sequences).
        """
        keys = ['state_source_timestamp', 'state'] + list(self.context_cols)
        batch = {k: [] for k in keys}
        for seq in sequences:
            for k in keys:
                batch[k].append([row[k] for row in seq])
        for k in batch:
            batch[k] = np.array(batch[k])
        return batch

    def _sample_batch_controls(self, batch_size, segment_length, mode):
        """Legacy controls sampling path - unchanged, out of scope for context/block_id support."""
        batch = {'state_source_timestamp': [],
                 'state': [],
                 'prediction': [],
                 'next_state': [],
                 'reward': [],
                 'terminate': [],
                 'truncate': []}

        required_samples = batch_size

        while required_samples > 0:
            timestamps = self.get_timestamps(window_size=segment_length,
                                            mode=mode,
                                            n=required_samples,
                                            agent_type="controls")
            for result in timestamps:
                window_seed = result['state_source_timestamp']
                results = self.sample_sequence(window_time_seed=window_seed,
                                               agent_type="controls",
                                               segment_length=segment_length)

                if results is None:
                    raise ValueError("No results found for the given window seed.")

                for key in batch:
                    batch[key].append([results[i][key] for i in range(len(results))])
            key = list(batch.keys())[0]
            required_samples = required_samples - len(batch[key])

        for key in batch:
            batch[key] = np.array(batch[key])

        return batch

    def sample_batch(self, batch_size, segment_length, agent_type, mode="latest", stratified_groups="all"):
        """
        Samples a batch of data from the database based on the specified parameters.

        Every returned sequence is validated to be block-homogeneous (see sample_sequence) -
        this applies in every mode, including "latest". If fewer valid sequences exist than
        batch_size, whatever was found is returned (logged at INFO) rather than hanging or
        raising.

        Args:
            batch_size (int): The number of samples to be included in the batch.
            segment_length (int): The length of each segment to be sampled.
            agent_type (str): The type of agent for which the batch is being sampled. Valid
                values are 'controls' or 'diagnostics'.
            mode (str): 'latest' (most recent valid sequences, context-agnostic) or
                'stratified' (see stratified_groups). Defaults to 'latest'.
            stratified_groups: Only used when mode='stratified'. One of:
                - "all" (default): sample uniformly across every distinct context_cols
                  combination present in the data.
                - "latest": sample only from sequences whose context matches the single
                  most recent row's context.
                - dict[tuple, float]: e.g. {(0,1,1,1): 0.7, (1,1,1,1): 0.3} - fraction of
                  batch_size to sample from sequences matching each context tuple. Weights
                  that don't sum to 1.0 are normalized (logged at INFO); a tuple with no
                  matching valid sequences contributes 0 examples (logged at INFO, not
                  redistributed to other groups).

        Returns:
            dict or None: None if agent_type is invalid or there isn't enough data at all
            (see check_sample_feasibility). Otherwise a dict with 'state_source_timestamp',
            'state', one array per context column (diagnostics), or the controls-specific
            keys ('prediction', 'next_state', 'reward', 'terminate', 'truncate') for controls.
        """
        if agent_type.lower() not in ["controls", "diagnostics"]:
            logging.debug(f"Invalid agent_type: {agent_type}. Valid values are 'controls' or 'diagnostics'.")
            return None

        if not self.check_sample_feasibility(segment_length, agent_type):
            logging.debug("Not enough samples in the database to sample a batch.")
            return None

        if agent_type.lower() == "controls":
            return self._sample_batch_controls(batch_size, segment_length, mode)

        if mode.lower() not in ("latest", "stratified"):
            logging.debug(f"Invalid mode: {mode}. Valid values are 'latest' or 'stratified'.")
            return None

        groups = self._resolve_stratified_groups(mode, stratified_groups)
        total_rows = self.get_size("agent_inferences")

        sequences = []
        for context_tuple, weight in groups.items():
            target_n = round(weight * batch_size)
            context_filter = dict(zip(self.context_cols, context_tuple)) if context_tuple is not None else None
            group_mode = "latest" if mode.lower() == "latest" else "random"
            group_sequences = self._collect_group_sequences(target_n=target_n,
                                                              segment_length=segment_length,
                                                              agent_type=agent_type,
                                                              mode=group_mode,
                                                              context_filter=context_filter,
                                                              candidate_pool_size=total_rows)
            if target_n > 0 and len(group_sequences) == 0 and context_tuple is not None:
                logging.info(f"sample_batch: no valid block-homogeneous windows found for context "
                             f"group {context_filter}; contributing 0 examples")
            sequences.extend(group_sequences)

        if len(sequences) < batch_size:
            logging.info(f"sample_batch: requested batch_size={batch_size} but only found "
                         f"{len(sequences)} valid block-homogeneous sequences; returning partial batch")

        return self._build_batch_dict(sequences)

    def record_sensor_data(self, data):
        """
        Records sensor data into the database.

        block_id is computed and injected here (never supplied by the caller) by comparing
        this row's timestamp/context_cols values against the latest previously written row -
        see _compute_block_id. Callers (ingest threads) stay fully agnostic of block_id.

        Args:
            data (dict): A dictionary containing the sensor data to be stored. The keys should match the columns in the `agent_inferences` table.

        Returns:
            int: The status of the operation, 0 if successful, 1 if there was an error.

        Raises:
            AssertionError: If the data is not a dictionary or if it is empty.
        """
        assert isinstance(data, dict), "Data must be a dictionary"
        if len(data) == 0:
            logging.debug("No data to store, exiting...")
            return 0

        data['block_id'] = self._compute_block_id(data)

        query = f"INSERT INTO agent_inferences "
        query_columns = "("
        query_values = "("
        values = []
        for key in data:
            if key in ['state', 'prediction'] and data[key] is not None:
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
                    'context': tuple(data.get(c) for c in self.context_cols),
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