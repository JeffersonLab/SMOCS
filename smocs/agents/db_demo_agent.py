# This Agent is only meant to be used for demonstration purposes to interact with the database.
# It should be removed once actual agents are implemented.


import numpy as np
import time
import os
from smocs.db.mysql_api_v0 import DBManager

class DemoDBAgent:
    def __init__(self):
        db_config = {
            'agent_id': 'demo_agent',
            'host': os.environ.get('MYSQL_HOST', 'localhost'),
            'port': int(os.environ.get('MYSQL_PORT', 3307)),
            'user': os.environ.get('MYSQL_USER', 'root'),
            'pwd': os.environ['MYSQL_ROOT_PASSWORD'],
            'database': os.environ.get('MYSQL_DATABASE', 'agentdb')  # Ensure the database is created
        }
        print("Connecting to database with config:", db_config)
        self.db_manager = DBManager(db_config)
        print("Connection to database ", self.db_manager.is_connected())
        self.observation_shape = (10, 2)
        self.batch_size = 3
        self.agent_type = "diagnostic"
        


    def store_data(self, data):
        status = self.db_manager.record_sensor_data(data)
        if status != 0:
            print("Failed to store data in the database.")
            print("data: ", data)
            
    def store_inference(self, prediction_data, key="state_source_timestamp"):
        assert key in prediction_data, f"Key '{key}' not found in prediction data."
        status = self.db_manager.update_record(prediction_data, key)
        if status != 0:
            print("Failed to store inference in the database.")
            print("prediction: ", prediction)
            print("key: ", key)

    def sample_training_batch(self, mode="random"):
        results = self.db_manager.sample_batch(segment_length=self.observation_shape[0],
                                               batch_size=self.batch_size,
                                               agent_type=self.agent_type,
                                               mode=mode)
        if results is None:
            print("No results found for the given parameters.")
        else:
            print("*"*30)
            print("Sampled Training Batch:")
            print(results)
            print("*"*30)
        
        return None
    

if __name__ == "__main__":
    agent = DemoDBAgent()
    print(f"DB Connection: {agent.db_manager.is_connected()}")
    # for i in range(100):
    #     source_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    #     data = {
    #         "state_source_timestamp": source_timestamp,
    #         "state_received_timestamp" : time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    #         "prediction_timestamp" : time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    #         "state": np.random.normal(0, 1, size=(2)), 
    #         "prediction": np.random.normal(0, 1, size=(2)),  
    #     }

    #     prediction_data = {"state_source_timestamp": source_timestamp,
    #                        state_id INT NOT NULL,
    #                         action_success BOOL,
    #                         reward FLOAT,
    #                         next_state_source_timestamp DATETIME(6) NOT NULL,
    #                         next_state_received_timestamp DATETIME(6) NOT NULL,
    #                         next_state BLOB NOT NULL,
    #                         done BOOL,
    #                         FOREIGN KEY (state_id) REFERENCES agent_inferences(id)
    #                        }

    
    while True:
        print("Agent initialized. Sampling training batch...")
        time.sleep(1)  # Simulate some delay