# Author: Kishansingh Rajput (Kishan)
# Organization: Thomas Jefferson National Acceleratory Facility
# Script: Utility script to establish connection to the Database

#Copyright 2024, Jefferson Science Associates, LLC.
#Subject to the terms in the License.txt file found in the top-level directory.

import mysql.connector as mysql
import numpy as np
import decimal
import time

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
            >>> db_connector = DBConnector("db_config.yaml")
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
        

    def execute_and_commit(self, query):
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
            self.db_cursor.execute(query)
            self.mydb.commit()
            status = 0
        except Exception as e:
            print("DB Error: ", e)
            print("with query: ", query)
            raise e
        
        return status
    
    def execute_query(self, query):
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
        query = f"CREATE DATABASE IF NOT EXISTS {self.db_name}"
        self.execute_and_commit(query)
        self.mydb.database = self.db_name

        query = f"CREATE TABLE IF NOT EXISTS agent_information (id INT AUTO_INCREMENT PRIMARY KEY, registered_id VARCHAR(50) NOT NULL, agent_name VARCHAR(50), config BLOB, info BLOB)"
        self.execute_and_commit(query)

        query = """
        CREATE TABLE IF NOT EXISTS agent_inferences (id INT AUTO_INCREMENT PRIMARY KEY, 
                                                                state_source_timestamp DATETIME(6) NOT NULL, 
                                                                state_received_timestamp DATETIME(6) NOT NULL,
                                                                prediction_timestamp DATETIME(6) NOT NULL, 
                                                                state BLOB NOT NULL, 
                                                                prediction BLOB NOT NULL)
        """
        self.execute_and_commit(query)

        query = """
        CREATE TABLE IF NOT EXISTS agent_replay (
                                                id INT AUTO_INCREMENT PRIMARY KEY,
                                                state_id INT NOT NULL,
                                                action_success BOOL,
                                                reward FLOAT,
                                                next_state_source_timestamp DATETIME(6) NOT NULL,
                                                next_state_received_timestamp DATETIME(6) NOT NULL,
                                                next_state BLOB NOT NULL,
                                                done BOOL,
                                                FOREIGN KEY (state_id) REFERENCES agent_inferences(id)
                                            )
        """
        self.execute_and_commit(query)

    
    def parse_results(self, results):
        parsed_results = []
        for result in results:
            for key in result:
                if type(result[key]) is decimal.Decimal:
                    result[key] = float(result[key])
                # TODO: add more types
                
            parsed_results.append(result)
        return np.array(parsed_results)

    def get_timestamps(self, window_size, mode="random", n=1):

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
        
        results = self.execute_query(query)
        parsed_results = self.parse_results(results)

        return parsed_results
    
    
    def sample_sequence(self, window_time_seed, agent_type, segment_length):
        """
        
        """
        if agent_type.lower() != "controls":
            query = f"SELECT ai.state, ar.next_state FROM agent_inferences ai JOIN agent_replay ar ON ai.Id = ar.state_id WHERE ai.state_source_timestamp >= '{window_time_seed}' ORDER BY ai.state_source_timestamp LIMIT {segment_length}"
        else:
            query = f"SELECT ai.state, ai.prediction, ar.next_state, ar.reward, ar.truncate, ar.terminate FROM agent_inferences ai JOIN agent_replay ar ON ai.Id = ar.state_id WHERE ai.state_source_timestamp >= '{window_time_seed}' ORDER BY ai.state_source_timestamp LIMIT {segment_length}"

        try:
            self.db_cursor.execute(query)
            results = self.db_cursor.fetchall()
        except Exception as e:
            print("DB Error: ", e)
            print("with query: ", query)
            return None
            
        parsed_results = self.parse_results(results)
            
        return parsed_results

    
    def sample_batch(self, batch_size, segment_length, agent_type, mode="random"):
        # Select n random timestamps as starting point for sequences
        """

        """
        
        batch = {'state_source_timestamp': [],
                              'state': [],
                              'prediction': [],
                              'next_state': []}
        if agent_type.lower() == "controls":
            batch['reward'] = []
            batch['done'] = []

        required_samples = batch_size
        
        while required_samples > 0:
            
            timestamps = self.get_timestamps(window_size=segment_length,
                                             mode=mode,
                                             n=batch_size)
            # print("parsed results after get timestamps: ", parsed_results)
            for result in timestamps:
                window_seed = result['ai.state_source_timestamp']
                results = self.sample_sequence(window_time_seed=window_seed,
                                               agent_type=agent_type,
                                               segment_length=segment_length)
                
                if results is None:
                    raise ValueError("No results found for the given window seed.")
                
                for result in results:
                    for key in batch:
                        batch[key].append(result[key])
                    
                

            required_samples = required_samples - len(batch)
            
        for key in batch:
            batch[key] = np.array(batch[key])
        
        return batch

    
    def record_sensor_data(self, data):
        assert isinstance(data, dict), "Data must be a dictionary"
        if len(data) == 0:
            print("No data to store, exiting...")
            return 0
        
        query = f"INSERT INTO agent_inferences "
        query_columns = "("
        query_values = "("
        for key in data:
            if not isinstance(data[key], float):
                print(f"Data for {key} is not a float, skipping...")
                continue
            query_columns+= f"{key}, "
            query_values += f"'{data[key]}', "
        query += query_columns[:-2] + ") VALUES " + query_values[:-2] + ")"
            
        status = self.execute_and_commit(query)    
            
        return status
    
    def get_size(self):
        query = f"SELECT COUNT(*) FROM {self.table_name}"
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
