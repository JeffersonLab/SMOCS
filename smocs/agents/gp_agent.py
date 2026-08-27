import argparse
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from smocs.cores import AgentBase, MLInferenceThreadBase
from smocs.utils import ConfigLoader, setup_logging


class GaussianProcessMLInferenceThread(MLInferenceThreadBase):
    """Run inference with a pre-trained Gaussian-process model."""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.input_features = self._resolve_input_features(config.get('model_input', {}))
        self.output_names = self._resolve_output_names(config.get('model_output', {}))
        self.model_path = Path(config.get('model', {}).get('path', ''))
        self.model_format = config.get('model', {}).get('format', 'pickle')
        self.return_std = config.get('model', {}).get('return_std', True)
        self.model_version = config.get('model', {}).get('version', self.model_path.name)
        self.model = None
        super().__init__(agent_id, config)

    @staticmethod
    def _resolve_input_features(model_input_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        features = model_input_config.get('features')
        if features is not None:
            if not isinstance(features, list) or not all(isinstance(feature, dict) for feature in features):
                raise ValueError('GPInferenceThread: model_input.features must be a list of mappings')
            return features
        return [
            {'name': channel, 'type': 'channel', 'channel': channel}
            for channel in model_input_config.get('channels', [])
        ]

    @staticmethod
    def _resolve_output_names(model_output_config: Dict[str, Any]) -> List[str]:
        output_name = model_output_config.get('name')
        if output_name:
            return [output_name]
        return list(model_output_config.get('channels', []))

    def load_model(self) -> None:
        """Load the configured pre-trained pickle or joblib model."""
        if not self.input_features:
            raise ValueError('GPInferenceThread: model_input.features or model_input.channels must be configured')
        if not self.output_names:
            raise ValueError('GPInferenceThread: model_output.name or model_output.channels must be configured')
        if not self.model_path.is_file():
            raise FileNotFoundError(f'GPInferenceThread: Model file does not exist: {self.model_path}')

        if self.model_format == 'pickle':
            with self.model_path.open('rb') as model_file:
                self.model = pickle.load(model_file)
        elif self.model_format == 'joblib':
            import joblib

            self.model = joblib.load(self.model_path)
        else:
            raise ValueError(f'GPInferenceThread: Unsupported model format {self.model_format!r}')

        if not callable(getattr(self.model, 'predict', None)):
            raise TypeError('GPInferenceThread: Loaded model has no predict() method')
        logging.info('GPInferenceThread: Loaded model %s', self.model_path)

    def _calculate_input_features(self, channels: Dict[str, Any]) -> Optional[Tuple[List[float], Dict[str, float]]]:
        input_values = {}
        for feature in self.input_features:
            name = feature.get('name')
            feature_type = feature.get('type', 'channel')
            if not name:
                logging.error('GPInferenceThread: Input feature has no name')
                return None

            if feature_type == 'channel':
                channel = feature.get('channel')
                if channel not in channels:
                    return None
                try:
                    input_values[name] = float(channels[channel])
                except (TypeError, ValueError):
                    logging.error('GPInferenceThread: Non-numeric input channel %s', channel)
                    return None
            elif feature_type == 'mean':
                feature_channels = feature.get('channels', [])
                if not feature_channels or any(channel not in channels for channel in feature_channels):
                    return None
                try:
                    input_values[name] = float(np.mean([float(channels[channel]) for channel in feature_channels]))
                except (TypeError, ValueError):
                    logging.error('GPInferenceThread: Non-numeric channel in mean feature %s', name)
                    return None
            else:
                logging.error('GPInferenceThread: Unsupported input feature type %r', feature_type)
                return None
        return list(input_values.values()), input_values

    def parse_inference_request(self, message, topic, partition, offset) -> Optional[Dict[str, Any]]:
        channels = message.get('channels', {})
        calculated_features = self._calculate_input_features(channels)
        if calculated_features is None:
            logging.debug('GPInferenceThread: Skipping %s:%s:%s; required input is unavailable', topic, partition, offset)
            return None
        input_values, named_input_values = calculated_features
        return {
            'features': np.asarray([input_values], dtype=float),
            'input_values': named_input_values,
            'timestamp': message.get('timestamp', time.time()),
            'run_number': message.get('run_number'),
        }

    def perform_inference(self, inference_request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            if self.model is None:
                self.load_model()

            if self.return_std:
                mean, standard_deviation = self.model.predict(inference_request['features'], return_std=True)
            else:
                mean = self.model.predict(inference_request['features'])
                standard_deviation = None

            mean_values = np.asarray(mean, dtype=float).reshape(-1)
            if len(mean_values) != len(self.output_names):
                raise ValueError(
                    f'GPInferenceThread: Model returned {len(mean_values)} outputs; expected {len(self.output_names)}'
                )

            predicted_outputs = dict(zip(self.output_names, mean_values.tolist()))
            result = {
                'predictions': mean_values.astype(np.float32),
                'predicted_outputs': predicted_outputs,
                'input_values': inference_request['input_values'],
                'model_version': self.model_version,
                'status': 'success',
                'timestamp': inference_request['timestamp'],
            }
            if standard_deviation is not None:
                std_values = np.asarray(standard_deviation, dtype=float).reshape(-1)
                if len(std_values) == 1:
                    result['prediction_stddev'] = float(std_values[0])
                elif len(std_values) == len(self.output_names):
                    result['prediction_stddev'] = dict(zip(self.output_names, std_values.tolist()))
                else:
                    raise ValueError('GPInferenceThread: Model returned an unexpected standard-deviation shape')
            return result
        except Exception as error:
            logging.error('GPInferenceThread: Inference failed: %s', error)
            return None

    def process_message(self, message, topic, partition, offset) -> Tuple[bool, List[Tuple]]:
        try:
            if isinstance(message, bytes):
                message = message.decode('utf-8')
            message_data = json.loads(message)
            inference_request = self.parse_inference_request(message_data, topic, partition, offset)
            if inference_request is None:
                return True, []
            inference_result = self.perform_inference(inference_request)
            if inference_result is None:
                return False, []

            output_channels = {
                'agent_id': self.agent_id,
                'model_version': inference_result['model_version'],
                'status': inference_result['status'],
                **inference_result['input_values'],
                **inference_result['predicted_outputs'],
            }
            if 'prediction_stddev' in inference_result:
                output_channels['prediction_stddev'] = inference_result['prediction_stddev']
            output_message = {'timestamp': time.time(), 'channels': output_channels}
            if inference_request['run_number'] is not None:
                output_message['run_number'] = inference_request['run_number']

            kafka_topic = self.producer.sanitize_topic_name(self.output_topic)
            return True, [(kafka_topic, json.dumps(output_message))]
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            logging.error('GPInferenceThread: Cannot process %s:%s:%s: %s', topic, partition, offset, error)
            return False, []

    def _store_inference_result(self, inference_request: Any, inference_result: Any) -> None:
        """GP inference is Kafka-only; calibration consumers retain run-level state."""
        del inference_request, inference_result


class GaussianProcessAgent(AgentBase):
    """Config-driven, inference-only Gaussian-process agent."""

    def __init__(self, config_path: Optional[str] = None, config_key: Optional[str] = None):
        super().__init__('GaussianProcessAgent')
        config_loader = ConfigLoader(config_path or os.environ.get('CONFIG_PATH', '/app/config.yaml'))
        gp_config = config_loader.config.get(config_key or 'gp_agent1', {})
        if not gp_config:
            raise ValueError(f'No Gaussian-process configuration found for {config_key or "gp_agent1"}')
        self.enabled_threads = gp_config.get('enabled_threads', ['inference'])
        self.agent_config = gp_config.copy()
        self.agent_config['agent_id'] = self.agent_id
        self.agent_config['switch_function'] = self.is_switch_on

    def create_data_ingest_component(self):
        return None

    def create_ml_training_component(self):
        return None

    def create_ml_inference_component(self):
        if 'inference' in self.enabled_threads:
            return GaussianProcessMLInferenceThread(self.agent_id, self.agent_config)
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_config', type=str, default='gp_agent1')
    args = parser.parse_args()
    setup_logging()
    GaussianProcessAgent(config_key=args.agent_config).start()


if __name__ == '__main__':
    main()