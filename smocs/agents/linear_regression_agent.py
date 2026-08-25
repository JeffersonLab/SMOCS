import argparse
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from smocs.cores import AgentBase, MLInferenceThreadBase
from smocs.utils import ChannelFilter, ConfigLoader, setup_logging


class LinearRegressionMLInferenceThread(MLInferenceThreadBase):
    """
    Inference-only linear regression thread.

    The regression coefficients are loaded directly from config so end users can
    adjust them without rebuilding a model artifact.
    """

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        model_input_config = config.get('model_input', {})
        model_output_config = config.get('model_output', {})

        self.raw_input_channels = self._resolve_input_channels(model_input_config)
        self.derived_inputs = model_input_config.get('derived_inputs', {})
        self.input_channels = list(self.derived_inputs) or list(self.raw_input_channels)
        self.input_channel = self.input_channels[0] if len(self.input_channels) == 1 else None

        self.output_names = self._resolve_output_names(model_output_config)
        self.output_name = self.output_names[0] if len(self.output_names) == 1 else None

        self.regression_models: Dict[str, Dict[str, Any]] = {}
        self.model_version = 'config'

        super().__init__(agent_id, config)

    def _resolve_input_channels(self, model_input_config: Dict[str, Any]) -> List[str]:
        input_channel = model_input_config.get('channel')
        if input_channel:
            return [input_channel]

        input_channels = model_input_config.get('channels', [])
        return list(input_channels)

    def _resolve_output_names(self, model_output_config: Dict[str, Any]) -> List[str]:
        output_name = model_output_config.get('name')
        if output_name:
            return [output_name]

        output_channels = model_output_config.get('channels', [])
        return list(output_channels)

    def load_model(self):
        """Load regression parameters from config."""
        regression_parameters = self.config.get('regression_parameters', {})

        if not self.input_channels:
            logging.error("LRInferenceThread: At least one model_input channel must be configured")
            self.model = None
            return

        if not self.output_names:
            logging.error("LRInferenceThread: At least one model_output name must be configured")
            self.model = None
            return

        parsed_models: Dict[str, Dict[str, Any]] = {}

        uses_single_output_shorthand = (
            len(self.input_channels) == 1
            and len(self.output_names) == 1
            and (
                'intercept' in regression_parameters
                or 'coefficient' in regression_parameters
                or 'slope' in regression_parameters
            )
        )

        if uses_single_output_shorthand:
            parsed_models[self.output_names[0]] = {
                'intercept': float(regression_parameters.get('intercept', 0.0)),
                'coefficients': {
                    self.input_channels[0]: float(
                        regression_parameters.get('coefficient', regression_parameters.get('slope', 0.0))
                    )
                },
            }
        else:
            for output_name in self.output_names:
                output_parameters = regression_parameters.get(output_name)
                if output_parameters is None:
                    logging.error(f"LRInferenceThread: Missing regression parameters for output {output_name}")
                    self.model = None
                    return

                coefficients = output_parameters.get('coefficients', {})
                parsed_models[output_name] = {
                    'intercept': float(output_parameters.get('intercept', 0.0)),
                    'coefficients': {
                        input_channel: float(coefficients.get(input_channel, 0.0))
                        for input_channel in self.input_channels
                    },
                }

        if not parsed_models:
            logging.error("LRInferenceThread: No regression parameters configured")
            self.model = None
            return

        self.regression_models = parsed_models
        self.model = parsed_models
        logging.info(
            f"LRInferenceThread: Loaded config-based linear regression model for inputs {self.input_channels} and outputs {self.output_names}"
        )

    def parse_inference_request(self, message_data, topic, partition, offset) -> Optional[Dict[str, Any]]:
        """Parse a full sensor message for inference."""
        try:
            channels = message_data.get('channels', {})
            if not channels:
                logging.error("LRInferenceThread: No channels in message data")
                self.model = None
                return

            missing_channels = [channel for channel in self.raw_input_channels if channel not in channels]
            if missing_channels:
                logging.debug(
                    f"LRInferenceThread: Skipping message from {topic}:{partition}:{offset}, missing channels {missing_channels}"
                )
                return None

            raw_input_values = {
                channel: float(channels[channel])
                for channel in self.raw_input_channels
            }
            input_values = self._calculate_derived_inputs(raw_input_values) if self.derived_inputs else raw_input_values

            return {
                'input_values': input_values,
                'timestamp': message_data.get('timestamp', time.time()),
                'channels': channels,
                'run_number': message_data.get('run_number'),
            }
        except Exception as e:
            logging.error(f"LRInferenceThread: Error parsing inference request: {e}")
            return None

    def _calculate_derived_inputs(self, raw_input_values: Dict[str, float]) -> Dict[str, float]:
        """Calculate configured model features from the raw EPICS channels."""
        input_values = {}
        for input_name, input_config in self.derived_inputs.items():
            if input_config.get('type') != 'pressure_over_temperature':
                raise ValueError(f"Unsupported derived input type for {input_name}")

            pressure = raw_input_values[input_config['pressure_channel']]
            temperature_channels = input_config['temperature_channels']
            temperatures = [raw_input_values[channel] for channel in temperature_channels]
            temperature_kelvin = input_config.get('kelvin_offset', 273.15) + sum(temperatures) / len(temperatures)
            if temperature_kelvin <= 0:
                raise ValueError(f"Derived input {input_name} has non-positive temperature")
            if not input_config.get('minimum_pressure', float('-inf')) <= pressure <= input_config.get('maximum_pressure', float('inf')):
                raise ValueError(f"Derived input {input_name} has pressure outside its configured range")

            input_values[input_name] = pressure / temperature_kelvin
        return input_values

    def perform_inference(self, inference_request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate the configured linear regression equations."""
        try:
            if self.model is None:
                self.load_model()
                if self.model is None:
                    return None

            input_values = inference_request['input_values']
            predictions: List[float] = []
            predicted_outputs: Dict[str, float] = {}

            for output_name in self.output_names:
                regression_model = self.regression_models[output_name]
                prediction = regression_model['intercept']

                for input_channel in self.input_channels:
                    coefficient = regression_model['coefficients'].get(input_channel, 0.0)
                    prediction += coefficient * input_values[input_channel]

                prediction = float(prediction)
                predictions.append(prediction)
                predicted_outputs[output_name] = prediction

            primary_output_name = self.output_names[0]
            primary_input_channel = self.input_channels[0]
            primary_input_value = input_values[primary_input_channel]

            return {
                'predictions': np.array(predictions, dtype=np.float32),
                'predicted_outputs': predicted_outputs,
                'input_values': input_values,
                'input_channels': list(self.input_channels),
                'output_names': list(self.output_names),
                'predicted_value': predicted_outputs[primary_output_name],
                'input_value': primary_input_value,
                'input_channel': primary_input_channel,
                'output_name': primary_output_name,
                'model_version': self.model_version,
                'status': 'success',
                'timestamp': inference_request['timestamp'],
            }
        except Exception as e:
            logging.error(f"LRInferenceThread: Error performing inference: {e}")
            return None

    def process_message(self, message, topic, partition, offset) -> Tuple[bool, List[Tuple]]:
        """Process full EPICS messages so the trigger PV can be independent from model inputs."""
        try:
            if isinstance(message, bytes):
                message = message.decode('utf-8')

            message_data = json.loads(message)

            if self.channel_filter:
                filtered_result = self.channel_filter.filter_channels(message_data)
                if filtered_result is None:
                    logging.debug(
                        f"LRInferenceThread: Skipping message from {topic}:{partition}:{offset} due to channel filtering"
                    )
                    return True, []

                channel_names, channel_values = filtered_result
            else:
                filtered_result = ChannelFilter.extract_all_channels(message_data)
                if filtered_result is None:
                    logging.debug(
                        f"LRInferenceThread: Skipping message from {topic}:{partition}:{offset} - no valid channels"
                    )
                    return True, []

                channel_names, channel_values = filtered_result

            filtered_channels = dict(zip(channel_names, channel_values))
            message_data['channels'] = filtered_channels

            logging.debug(f"LRInferenceThread: Extracted {len(channel_values)} channels for inference")

            if not self.switch_fn(filtered_channels):
                logging.debug(f"LRInferenceThread: Switch is OFF, inference is not performed on the following message")
                logging.debug(f"LRInferenceThread: Message {filtered_channels}")
                return False, []

            inference_request = self.parse_inference_request(message_data, topic, partition, offset)
            if inference_request is None:
                return False, []

            inference_result = self.perform_inference(inference_request)
            if inference_result is None:
                return False, []

            self._store_inference_result(inference_request, inference_result)

            output_channels = {
                'agent_id': self.agent_id,
                'model_version': inference_result.get('model_version', self.model_version),
                'status': inference_result.get('status', 'unknown'),
            }

            if len(inference_result['input_channels']) == 1:
                output_channels['input_channel'] = inference_result['input_channel']
                output_channels['input_value'] = inference_result['input_value']

            if len(inference_result['output_names']) == 1:
                output_channels['output_name'] = inference_result['output_name']

            for input_channel, input_value in inference_result['input_values'].items():
                output_channels[f'{input_channel}_input'] = input_value

            output_channels.update(inference_result['predicted_outputs'])

            output_message = {
                'timestamp': time.time(),
                'channels': output_channels,
            }
            if inference_request.get('run_number') is not None:
                output_message['run_number'] = inference_request['run_number']

            kafka_topic = self.producer.sanitize_topic_name(self.output_topic)
            return True, [(kafka_topic, json.dumps(output_message))]
        except json.JSONDecodeError as e:
            logging.error(
                f"LRInferenceThread: JSON decode error for message from {topic}:{partition}:{offset}: {e}"
            )
            return False, []
        except Exception as e:
            logging.error(f"LRInferenceThread: Error processing inference message: {e}")
            return False, []

    def _store_inference_result(self, inference_request: Dict[str, Any], inference_result: Dict[str, Any]):
        """Store the regression outputs using the existing prediction table."""
        try:
            if inference_result.get('status') != 'success':
                return

            source_timestamp = inference_request.get('timestamp')
            if source_timestamp is None:
                logging.warning("LRInferenceThread: No timestamp in inference request, cannot store result")
                return

            if isinstance(source_timestamp, (int, float)):
                timestamp_dt = datetime.fromtimestamp(source_timestamp)
                source_timestamp_str = timestamp_dt.strftime('%Y-%m-%d %H:%M:%S.%f')
            else:
                source_timestamp_str = str(source_timestamp)

            status = self.db_manager.record_prediction(
                prediction=inference_result['predictions'],
                prediction_timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                key_value=source_timestamp_str,
                key='state_source_timestamp',
            )

            if status != 0:
                logging.error(f"LRInferenceThread: Failed to store inference result, status: {status}")
        except Exception as e:
            logging.error(f"LRInferenceThread: Error storing inference result: {e}")


class LinearRegressionAgent(AgentBase):
    """Config-driven linear regression agent."""

    def __init__(self, config_path: str = None, config_key: str = None):
        super().__init__("LinearRegressionAgent")

        if config_path:
            config_loader = ConfigLoader(config_path)
            if config_key is None:
                config_key = 'linear_regression_agent1'
            linear_regression_config = config_loader.config.get(config_key, {})
            self.enabled_threads = linear_regression_config.get('enabled_threads', ['inference'])
        else:
            linear_regression_config = {
                'enabled_threads': ['inference'],
                'model_input': {'channel': ''},
                'model_output': {'name': 'predicted_value'},
                'regression_parameters': {
                    'intercept': 0.0,
                    'coefficient': 0.0,
                },
                'kafka_topics': {
                    'input': 'CEBAF',
                    'output': 'linear-regression-predictions',
                },
            }
            self.enabled_threads = ['inference']

        if 'kafka_topics' not in linear_regression_config:
            logging.warning("LRAgent: No kafka_topics found in config, using defaults")
            linear_regression_config['kafka_topics'] = {
                'input': 'CEBAF',
                'output': 'linear-regression-predictions',
            }

        self.agent_config = linear_regression_config.copy()
        self.agent_config['agent_id'] = self.agent_id
        self.agent_config['switch_function'] = self.is_switch_on

        logging.info(f"LRAgent: LinearRegressionAgent initialized with config: {self.agent_config}")
        logging.info(f"LRAgent: Enabled threads: {self.enabled_threads}")

    def create_data_ingest_component(self):
        """Create data ingestion thread component."""
        return None

    def create_ml_training_component(self):
        """Create ML training thread component."""
        return None

    def create_ml_inference_component(self):
        """Create ML inference thread component."""
        if 'inference' in self.enabled_threads:
            return LinearRegressionMLInferenceThread(self.agent_id, self.agent_config)
        return None


def main():
    """Main entry point for the linear regression agent."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--agent_config',
        help='Key for agent configuration in config file',
        type=str,
        default='linear_regression_agent1',
    )
    args = parser.parse_args()

    setup_logging()

    config_path = os.getenv('CONFIG_PATH', '/app/config.yaml')

    try:
        agent = LinearRegressionAgent(config_path, args.agent_config)
        agent.start()
    except KeyboardInterrupt:
        logging.info('Shutting down linear regression agent...')
    except Exception as e:
        logging.error(f'Error running linear regression agent: {e}')
        raise


if __name__ == '__main__':
    main()