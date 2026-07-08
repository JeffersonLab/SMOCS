import json
import os
from unittest.mock import MagicMock, patch

import numpy as np


os.environ.setdefault('MYSQL_HOST', 'localhost')
os.environ.setdefault('MYSQL_PORT', '3307')
os.environ.setdefault('MYSQL_USER', 'root')
os.environ.setdefault('MYSQL_ROOT_PASSWORD', 'test_password')
os.environ.setdefault('MYSQL_DATABASE', 'test_db')
os.environ.setdefault('KAFKA_BROKER_URL', 'test-broker:9092')


class MockDBManager:
    def __init__(self, db_config):
        self.db_name = f"SMOCS_Agent_{db_config['database']}"

    def is_connected(self):
        return True

    def record_prediction(self, prediction, prediction_timestamp, key_value, key='state_source_timestamp'):
        return 0

    def close(self):
        pass


patcher_db = patch('smocs.db.mysql_api_v0.DBManager', MockDBManager)
patcher_kafka_consumer = patch('kafka.KafkaConsumer')
patcher_kafka_producer = patch('kafka.KafkaProducer')
patcher_kafka_admin = patch('kafka.KafkaAdminClient')

patcher_db.start()
patcher_kafka_consumer.start()
patcher_kafka_producer.start()
patcher_kafka_admin.start()

from smocs.agents.linear_regression_agent import LinearRegressionMLInferenceThread


def teardown_module(module):
    patcher_db.stop()
    patcher_kafka_consumer.stop()
    patcher_kafka_producer.stop()
    patcher_kafka_admin.stop()


def test_linear_regression_inference_runs_for_each_message():
    config = {
        'model_input': {
            'channel': 'feature_a',
        },
        'model_output': {
            'name': 'predicted_value',
        },
        'regression_parameters': {
            'intercept': 1.0,
            'coefficient': 2.0,
        },
        'kafka_topics': {
            'input': 'CEBAF',
            'output': 'linear-regression-results',
        },
    }

    thread = LinearRegressionMLInferenceThread('agent-123', config)
    thread.db_manager.record_prediction = MagicMock(return_value=0)

    first_message = {
        'timestamp': 1000.0,
        'channels': {
            'feature_a': 1.0,
        },
    }
    success, outputs = thread.process_message(json.dumps(first_message), 'CEBAF', 0, 1)
    assert success is True
    assert len(outputs) == 1

    second_message = {
        'timestamp': 1001.0,
        'channels': {
            'feature_a': 1.2,
        },
    }
    success, outputs = thread.process_message(json.dumps(second_message), 'CEBAF', 0, 2)
    assert success is True
    assert len(outputs) == 1

    topic, payload = outputs[0]
    assert topic == 'linear-regression-results'

    parsed_payload = json.loads(payload)
    channels = parsed_payload['channels']

    assert channels['input_channel'] == 'feature_a'
    assert channels['output_name'] == 'predicted_value'
    assert np.isclose(channels['input_value'], 1.2)
    assert np.isclose(channels['feature_a_input'], 1.2)
    assert np.isclose(channels['predicted_value'], 3.4)
    assert channels['status'] == 'success'
    assert thread.db_manager.record_prediction.call_count == 2


def test_linear_regression_supports_multi_input_multi_output_models():
    config = {
        'model_input': {
            'channels': ['pressure', 'gas_density'],
        },
        'model_output': {
            'channels': ['calib_const_a', 'calib_const_b'],
        },
        'regression_parameters': {
            'calib_const_a': {
                'intercept': 1.0,
                'coefficients': {
                    'pressure': 2.0,
                    'gas_density': -0.5,
                },
            },
            'calib_const_b': {
                'intercept': -2.0,
                'coefficients': {
                    'pressure': 0.25,
                    'gas_density': 4.0,
                },
            },
        },
        'kafka_topics': {
            'input': 'CEBAF',
            'output': 'linear-regression-results',
        },
    }

    thread = LinearRegressionMLInferenceThread('agent-456', config)
    thread.db_manager.record_prediction = MagicMock(return_value=0)

    message = {
        'timestamp': 2000.0,
        'channels': {
            'pressure': 10.0,
            'gas_density': 2.0,
        },
    }
    success, outputs = thread.process_message(json.dumps(message), 'CEBAF', 0, 1)
    assert success is True
    assert len(outputs) == 1

    topic, payload = outputs[0]
    assert topic == 'linear-regression-results'

    parsed_payload = json.loads(payload)
    channels = parsed_payload['channels']

    assert np.isclose(channels['pressure_input'], 10.0)
    assert np.isclose(channels['gas_density_input'], 2.0)
    assert np.isclose(channels['calib_const_a'], 20.0)
    assert np.isclose(channels['calib_const_b'], 8.5)
    assert channels['status'] == 'success'
    thread.db_manager.record_prediction.assert_called_once()