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
            
    def store_prediction(self, pred, pred_timestamp, key_value, key="state_source_timestamp"):
        
        status = self.db_manager.record_prediction(pred, pred_timestamp, key_value, key)
        if status != 0:
            print("Failed to store inference in the database.")
            print("prediction: ", prediction)
            print("key: ", key)
        
        return status
    
    def store_controls_tuple(self, prediction_data, state_id):
        if state_id is None:
            print("State ID is None. Cannot store controls tuple.")
            return
        
        status = self.db_manager.record_controls_tuple(prediction_data, state_id)
        if status != 0:
            print("Failed to store controls tuple in the database.")
            print("prediction_data: ", prediction_data)
            print("state_id: ", state_id)
        
        return status

    def sample_training_batch_diagnostic(self, mode="random"):
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
        
        return results
    
    def sample_training_batch_controls(self, mode="random"):
        results = self.db_manager.sample_batch(segment_length=1,
                                               batch_size=self.batch_size,
                                               agent_type='controls',
                                               mode=mode)
        if results is None:
            print("No results found for the given parameters.")
        else:
            print("*"*30)
            print("Sampled Training Batch:")
            print(results)
            print("*"*30)
        
        return results
    

if __name__ == "__main__":
    agent = DemoDBAgent()
    print(f"DB Connection: {agent.db_manager.is_connected()}")
    for i in range(5):
        source_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        state = np.random.normal(0, 1, size=(2))
        data = {
            "state_source_timestamp": source_timestamp,
            "state_received_timestamp" : time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "state": state
        }
        agent.store_data(data)


        pred = np.random.normal(0, 1, size=(2))
        pred_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        agent.store_prediction(pred, pred_timestamp, source_timestamp, key="state_source_timestamp")


        next_state = state + pred
        prediction_data = {"action_success": True,
                            "reward": np.array([0.3, 0.9]),
                            "next_state_source_timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                            "next_state_received_timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                            "next_state": next_state,
                            "terminate": False,
                            "truncate": False,
                            "info": {"env_property": "env_property_value", "env version": 1.0}  # Assuming info is not used in this demo
                           }
        
        state_id = agent.db_manager.get_state_id(source_timestamp)
        print(f"state_id: {state_id}")
        agent.store_controls_tuple(prediction_data, state_id)
        print(f"Stored data for iteration {i+1}: {data['state_source_timestamp']}")
        time.sleep(1)  # Simulate some delay

    data = agent.sample_training_batch_diagnostic(mode="random")
    print("Sampled Data for Diagnostics: ", data)
    print("keys: ", data.keys())
    for key in data:
        print(f"{key} Shape: {data[key].shape}")

    data = agent.sample_training_batch_controls(mode="random")
    print("Sampled Data for Controls: ", data)
    print("keys: ", data.keys())
    for key in data:
        print(f"{key} Shape: {data[key].shape}")