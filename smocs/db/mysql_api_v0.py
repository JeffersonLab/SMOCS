# Author: Kishansingh Rajput (Kishan)
# Organization: Thomas Jefferson National Acceleratory Facility
# Script: Utility script to establish connection to the Database

#Copyright 2024, Jefferson Science Associates, LLC.
#Subject to the terms in the License.txt file found in the top-level directory.

import mysql.connector as mysql
import numpy as np
import decimal
import time
import pickle

class DBManager:

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
        
        self.db_name = f"SMOCS_Agent_{db_cfg_dict['database']}"
        print("initializing DBConnector")
        
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
            Exception: If there is an error executing the query, it will print the error message.
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
            print("DB Error: ", e)
            print("with query: ", query)
            raise e
        
        return status
    
    def __execute_query(self, query):
        """
        Executes a SQL query and returns the results.

        Args:
            query (str): The SQL query to be executed.

        Returns:
            list: A list of dictionaries containing the results of the query.

        Raises:
            Exception: If there is an error executing the query, it will print the error message.
        """
        try:
            self.db_cursor.execute(query)
            results = self.db_cursor.fetchall()
        except Exception as e:
            print("DB Error: ", e)
            print("with query: ", query)
            raise e
        
        return results

    def connect(self, db_config, n_connection_trials=5):

        """
        Establishes a connection to the MySQL database using the provided configuration.
        Args:
            db_config (dict): A dictionary containing the database connection parameters such as host, user,
                                password, and database name.
            n_connection_trials (int): The number of attempts to connect to the database before giving up.
        Raises:
            Exception: If the connection fails after the specified number of trials, an exception is raised and
                        the process is terminated.
        Returns:
            None
        """
        
        for i in range(n_connection_trials):
            try:
                print("Connecting to DB - trial ", i)
                self.mydb = mysql.connect(host=db_config["host"], 
                                username=db_config["user"], 
                                password=db_config["pwd"], 
                                database=db_config["database"],
                                autocommit=True)

                self.db_cursor = self.mydb.cursor(dictionary=True)
                break
            except Exception as e:            
                print(e)
                if i == n_connection_trials-1:
                    print(f"Could not connect to DATABASE in {n_connection_trials} attempts, exiting.")
                    exit(1)
                print("ERROR: CANNOT CONNECT TO DATABASE, WAITING 5 sec...")
                time.sleep(5)

        print("CONNECTED TO DB: ", db_config['database'])


    def create_tables(self):
        """
        Creates the necessary tables in the database if they do not already exist.
        This method is called during the initialization of the DBManager to ensure that the required tables
        for storing agent information, inferences, and replay data are present in the database.
        It executes SQL queries to create the following tables:
            - agent_information: Stores information about the agents.
            - agent_inferences: Stores the inferences made by the agents.
            - agent_replay: Stores the replay data for the agents.
        If the database does not exist, it will be created first.
        """
        query = f"CREATE DATABASE IF NOT EXISTS {self.db_name}"
        self.__execute_and_commit(query)
        self.mydb.database = self.db_name

        query = f"CREATE TABLE IF NOT EXISTS agent_information (id INT AUTO_INCREMENT PRIMARY KEY, registered_id VARCHAR(50) NOT NULL, agent_name VARCHAR(50), config BLOB, info BLOB)"
        self.__execute_and_commit(query)

        query = """
        CREATE TABLE IF NOT EXISTS agent_inferences (id INT AUTO_INCREMENT PRIMARY KEY, 
                                                                state_source_timestamp DATETIME(6) NOT NULL, 
                                                                state_received_timestamp DATETIME(6) NOT NULL,
                                                                state BLOB NOT NULL, 
                                                                prediction_timestamp DATETIME(6), 
                                                                prediction BLOB)
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
            >>> print(parsed_results)
            [{'id': 1, 'value': 3.14, 'data': array([0., 1., 2., 3.])}]
        """
        parsed_results = []
        for result in results:
            for key in result:
                if type(result[key]) is decimal.Decimal:
                    result[key] = float(result[key])
                if type(result[key]) is bytes:
                    try:
                        result[key] = np.frombuffer(result[key], dtype=np.float64)
                    except ValueError:
                        try:
                            result[key] = pickle.loads(result[key])
                        except pickle.UnpicklingError:
                            print(f"Could not unpickle data for key: {key}")
                
            parsed_results.append(result)
        return np.array(parsed_results)

    def get_timestamps(self, window_size, mode="random", n=1):
        """
        Retrieves a list of timestamps from the database that have at least `window_size` number of records in the `agent_inferences` table.
        Args:
            window_size (int): The minimum number of records required for each timestamp.
            mode (str): The mode of sampling, either "random" or "latest". Defaults to "random".
            n (int): The number of timestamps to retrieve. Defaults to 1.
        Returns:
            list: A list of dictionaries containing the timestamps that meet the criteria.
        Raises:
            None: If the mode is not understood, it returns None.
        Example:
            >>> timestamps = self.get_timestamps(window_size=5, mode="random", n=3)
            >>> print(timestamps)
            [{'state_source_timestamp': '2023-10-01 12:00:00'}, 
             {'state_source_timestamp': '2023-10-01 12:05:00'}, 
             {'state_source_timestamp': '2023-10-01 12:10:00'}]
        """

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
        ORDER BY
        """
        if mode.lower() == "random":
            query += f" RAND() LIMIT {n}"
        elif mode.lower() == "latest":
            query += f" DESC LIMIT {n}"
        else:
            print("mode not understood in get_timestamps function...most likely coming from invalid mode argument to sample_batch function. Valid values are 'random' or 'latest' ")
            return None
        
        results = self.__execute_query(query)
        parsed_results = self.parse_results(results)

        return parsed_results
    
    
    def sample_sequence(self, window_time_seed, agent_type, segment_length):
        """
        Samples a sequence of data from the database starting from a given timestamp.
        Args:
            window_time_seed (str): The starting timestamp for the sequence.
            agent_type (str): The type of agent for which the sequence is being sampled.
            segment_length (int): The length of the segment to be sampled.
        Returns:
            list: A list of dictionaries containing the sampled data for the specified agent type.
        Raises:
            Exception: If there is an error executing the query, it will print the error message and return None.
        Example:
            >>> sequence = self.sample_sequence(window_time_seed='2023-10-01 12 :00:00', agent_type='diagnostics', segment_length=10)
            >>> print(sequence)
            [{'state_source_timestamp': '2023-10-01 12:00:00', 'state': array([0.1, 0.2])},
             {'state_source_timestamp': '2023-10-01 12:00:01', 'state': array([0.3, 0.4])},
             ...]
        """
        if agent_type.lower() != "controls":
            # query = f"SELECT ai.state_source_timestamp, ai.state, ar.next_state FROM agent_inferences ai JOIN agent_replay ar ON ai.Id = ar.state_id WHERE ai.state_source_timestamp >= '{window_time_seed}' ORDER BY ai.state_source_timestamp LIMIT {segment_length}"
            query = f"SELECT state_source_timestamp, state FROM agent_inferences WHERE state_source_timestamp >= '{window_time_seed}' ORDER BY state_source_timestamp LIMIT {segment_length}"
        else:
            query = f"SELECT ai.state_source_timestamp, ai.state, ai.prediction, ar.next_state, ar.reward, ar.truncate, ar.terminate FROM agent_inferences ai JOIN agent_replay ar ON ai.Id = ar.state_id WHERE ai.state_source_timestamp >= '{window_time_seed}' ORDER BY ai.state_source_timestamp LIMIT {segment_length}"

        try:
            self.db_cursor.execute(query)
            results = self.db_cursor.fetchall()
        except Exception as e:
            print("DB Error: ", e)
            print("with query: ", query)
            return None
            
        parsed_results = self.parse_results(results)
            
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
            >>> print(success)
            True
        """
        success = True
        number_of_records_prediction_table = self.get_size(table_name="agent_inferences")
        if number_of_records_prediction_table < segment_length:
            print("Number of records in prediction table is less than segment length. Cannot sample batch, waiting for more data to be recorded...")
            success = False
        
        if agent_type.lower() == "controls":
            number_of_records_replay_table = self.get_size(table_name="agent_replay")
            if number_of_records_replay_table < segment_length:
                print("Number of records in replay table is less than segment length. Cannot sample batch, waiting for more data to be recorded...")
                success = False
        
        return success

    
    def sample_batch(self, batch_size, segment_length, agent_type, mode="random"):
        # Select n random timestamps as starting point for sequences
        """
        Samples a batch of data from the database based on the specified parameters.

        Args:
            batch_size (int): The number of samples to be included in the batch.
            segment_length (int): The length of each segment to be sampled.
            agent_type (str): The type of agent for which the batch is being sampled. Valid
                values are 'controls' or 'diagnostics'.
            mode (str): The mode of sampling, either 'random' or 'latest'. Defaults to 'random'.   


        Raises:
            ValueError: If the agent_type is not 'controls' or 'diagnostics', or if there are not enough samples in the database to sample a batch.

        Returns:
            dict: A dictionary containing the sampled batch data, with keys such as 'state_source_timestamp', 'state', 'prediction', 'next_state', 'reward', 'terminate', and 'truncate'.
        """
        if agent_type.lower() not in ["controls", "diagnostics"]:
            print(f"Invalid agent_type: {agent_type}. Valid values are 'controls' or 'diagnostics'.")
            return None
        
        if not self.check_sample_feasibility(segment_length, agent_type):
            print("Not enough samples in the database to sample a batch.")
            return None
        
        batch = {'state_source_timestamp': [],
                              'state': []}
        if agent_type.lower() == "controls":
            batch['prediction'] = []
            batch['next_state'] = []
            batch['reward'] = []
            batch['terminate'] = []
            batch['truncate'] = []

        required_samples = batch_size
        
        while required_samples > 0:
            
            timestamps = self.get_timestamps(window_size=segment_length,
                                             mode=mode,
                                             n=required_samples)
            # print("parsed results after get timestamps: ", parsed_results)
            for result in timestamps:
                window_seed = result['state_source_timestamp']
                results = self.sample_sequence(window_time_seed=window_seed,
                                               agent_type=agent_type,
                                               segment_length=segment_length)
                
                if results is None:
                    raise ValueError("No results found for the given window seed.")
                
                print("results: ", results)
                
                for key in batch:
                    batch[key].append([results[i][key] for i in range(len(results))])
            key = list(batch.keys())[0]
            required_samples = required_samples - len(batch[key])
            
        for key in batch:
            batch[key] = np.array(batch[key])
        
        return batch

    
    def record_sensor_data(self, data):
        """
        Records sensor data into the database.

        Args:
            data (dict): A dictionary containing the sensor data to be stored. The keys should match the columns in the `agent_inferences` table.

        Returns:
            int: The status of the operation, 0 if successful, 1 if there was an error.

        Raises:
            AssertionError: If the data is not a dictionary or if it is empty.
        """
        assert isinstance(data, dict), "Data must be a dictionary"
        if len(data) == 0:
            print("No data to store, exiting...")
            return 0
        
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
            print(f"No state found for source timestamp: {source_timestamp}")
            return None
        elif len(results) > 1:
            print(f"Multiple states found for source timestamp: {source_timestamp}, returning the first one.")
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
            print("State ID is None. Cannot store controls tuple.")
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
        print("closing cursor")
        self.db_cursor.close()
        print("closing connection")
        self.mydb.close()
        print("db connection closed")


# DOCUMENTATION: notifies of database error if connection fails 
