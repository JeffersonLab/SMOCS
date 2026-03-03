import os
import json
import time
import logging
import argparse
from typing import Dict, Any, Optional, List, Tuple

import pickle
import numpy as np
from datetime import datetime

from smocs.cores import (
    AgentBase,
    MLInferenceThreadBase,
)
from smocs.utils import ConfigLoader, ChannelFilter, setup_logging


class RidgeRegressionMLInferenceThread(MLInferenceThreadBase):
    """
    ML inference thread for ridge regression agent.
    Loads a pre‑trained model and makes predictions on incoming data.
    """

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.model = None
        self.scaler = None
        self.feature_channels = None
        self.target_names = []
        self.alpha = None
        self.current_model_version = None
        self.last_model_check = 0
        self.model_check_interval = config.get("model_check_interval", 30)

        # no preprocessing or expression support in minimal inference agent
        self.preprocessing_manager = None


        super().__init__(agent_id, config)

    def load_model(self):
        """Load the latest ridge regression model from local directory."""
        try:
            models_dir = "/app/models"
            latest_file = f"{models_dir}/latest_model.json"

            if not os.path.exists(latest_file):
                logging.warning("RRMLInferenceThread: No latest model file found")
                return

            # Read latest model info
            with open(latest_file, "r") as f:
                latest_info = json.load(f)

            model_version = latest_info["version"]

            # Check if we already have this version loaded
            if self.current_model_version == model_version:
                return

            model_file = f"{models_dir}/{latest_info['model_file']}"

            if not os.path.exists(model_file):
                logging.error(
                    f"RRMLInferenceThread: Model file not found: {model_file}"
                )
                return

            # Load the sklearn model
            with open(model_file, "rb") as f:
                model_data = pickle.load(f)

            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.feature_channels = model_data["feature_channels"]
            self.target_names = model_data.get("target_names", [])
            self.alpha = model_data.get("alpha", 1.0)
            self.current_model_version = model_version

            logging.info(
                f"RRMLInferenceThread: Loaded model v{model_version}: targets={self.target_names}, features={self.feature_channels}, alpha={self.alpha}"
            )

        except Exception as e:
            logging.error(f"RRMLInferenceThread: Error loading model: {e}")

    def check_for_model_updates(self):
        """Check if a new model is available and load if necessary."""
        current_time = time.time()
        if current_time - self.last_model_check > self.model_check_interval:
            self.load_model()
            self.last_model_check = current_time



    def perform_inference(
        self, inference_request: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Perform inference using the loaded ML model.
        """
        try:
            # Always use ML model prediction; check for new model periodically
            self.check_for_model_updates()

            if self.model is None:
                self.load_model()
                if self.model is None:
                    logging.debug(
                        "RRMLInferenceThread: No model available for inference"
                    )
                    return None

            channels = inference_request["channels"]

            # Extract feature values based on feature_channels loaded from model
            if not self.feature_channels:
                logging.error(
                    "RRMLInferenceThread: No feature channels defined in loaded model"
                )
                return None

            feature_values = []
            for channel_name in self.feature_channels:
                if channel_name in channels:
                    feature_values.append(channels[channel_name])
                else:
                    logging.warning(
                        f"RRMLInferenceThread: Missing feature channel {channel_name} in message"
                    )
                    return None

            X = np.array(feature_values, dtype=np.float32).reshape(1, -1)

            # Extract actual target values if present in the inference request
            # This allows for error computation if the actual values are provided
            if not self.target_names:
                logging.warning(
                    "RRMLInferenceThread: No target names defined in loaded model; cannot compute actual values"
                )
                return None
            
            target_values = []
            for target_name in self.target_names:
                if target_name in channels:
                    target_values.append(channels[target_name])
                else:
                    logging.warning(
                        f"RRMLInferenceThread: Missing target channel {target_name} in message"
                    )
                    return None
                
            y_actual = np.array(target_values, dtype=np.float32).reshape(1, -1)
            

            # Normalize features if configured
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X

            # Make prediction
            y_pred = self.model.predict(X_scaled)[0].reshape(1, -1)

            abs_residuals = np.abs(y_pred - y_actual)

            logging.info(f"Prediction shape: {y_pred.shape}, Actual shape: {y_actual.shape}, Residuals shape: {abs_residuals.shape}")

            y_pred = y_pred.squeeze()
            y_actual = y_actual.squeeze()
            abs_residuals = abs_residuals.squeeze()


            result = {
                "prediction_0": y_pred.tolist()[0],
                "prediction_1": y_pred.tolist()[1],
                "actual_0": y_actual.tolist()[0],
                "actual_1": y_actual.tolist()[1],
                "prediction_error_0": abs_residuals.tolist()[0],
                "prediction_error_1": abs_residuals.tolist()[1],
                # "features_0": X.flatten()[0],
                # "features_1": X.flatten()[1],
                "model_version": self.current_model_version,
                "alpha": self.alpha,
                "timestamp": time.time_ns(),
                "status": "success",
            }
            # result = {
            #     "prediction": y_pred.tolist(),
            #     "actual": y_actual.tolist(),
            #     "prediction_error": abs_residuals.tolist(),
            #     "features": X.flatten().tolist(),
            #     "model_version": self.current_model_version,
            #     "alpha": self.alpha,
            #     "timestamp": inference_request["timestamp"],
            #     "status": "success",
            # }

            logging.debug(
                # log vector predictions gracefully
                f"RRMLInferenceThread: Prediction={y_pred.tolist()}, Actual={y_actual.tolist()}, Error={abs_residuals.tolist()}"
            )

            return result

        except Exception as e:
            logging.error(f"RRMLInferenceThread: Error performing inference: {e}")
            return None

    def parse_inference_request(
        self, message_data, topic, partition, offset
    ) -> Optional[Dict[str, Any]]:
        """
        Parse inference request from sensor message.

        Args:
            message_data: Already parsed and filtered message dict
            topic: Kafka topic
            partition: Kafka partition
            offset: Kafka offset

        Returns:
            Parsed sensor data or None
        """
        try:
            channels = message_data.get("channels", {})

            if not channels:
                logging.error(
                    "RRMLInferenceThread: No channels in filtered message data"
                )
                return None

            # We simply forward the filtered channels and timestamp and
            # attach any actual values for the target PVs (if present).
            result = {
                "channels": channels,
                "timestamp": message_data.get("timestamp", time.time_ns()),
            }

            return result

        except Exception as e:
            logging.error(f"RRMLInferenceThread: Error parsing inference request: {e}")
            return None

    def process_message(
        self, message, topic, partition, offset
    ) -> Tuple[bool, List[Tuple]]:
        """
        Process incoming message with optional channel filtering and return inference results.
        """
        try:
            # Parse message
            if isinstance(message, bytes):
                message = message.decode("utf-8")

            message_data = json.loads(message)

            logging.info(
                f"RRMLInferenceThread: Received message from {topic}:{partition}:{offset}"
            )

            # Apply channel filtering or extract all channels
            if self.channel_filter:
                # Use configured channel filtering
                filtered_result = self.channel_filter.filter_channels(message_data)
                if filtered_result is None:
                    logging.info(
                        f"RRMLInferenceThread: Skipping message from {topic}:{partition}:{offset} due to channel filtering"
                    )
                    return True, []

                channel_names, channel_values = filtered_result
                logging.info(
                    f"RRMLInferenceThread: Filtered {len(channel_values)} channels from message"
                )
            else:
                # Extract all numeric channels when no filter configured
                filtered_result = ChannelFilter.extract_all_channels(message_data)
                if filtered_result is None:
                    logging.debug(
                        f"RRMLInferenceThread: Skipping message from {topic}:{partition}:{offset} - no valid channels"
                    )
                    return True, []

                channel_names, channel_values = filtered_result

            # Create clean channel dictionary for agent processing
            filtered_channels = dict(zip(channel_names, channel_values))
            message_data["channels"] = filtered_channels

            logging.info(
                f"RRMLInferenceThread: Extracted {len(channel_values)} channels for inference: {list(filtered_channels.keys())}"
            )

            # Parse inference request with processed data
            inference_request = self.parse_inference_request(
                message_data, topic, partition, offset
            )

            if inference_request is None:
                logging.info("RRMLInferenceThread: Failed to parse inference request")
                return False, []

            # Perform inference
            inference_result = self.perform_inference(inference_request)

            if inference_result is None or (
                isinstance(inference_result, list) and len(inference_result) == 0
            ):
                logging.info("RRMLInferenceThread: No inference result generated")
                return True, []  # Continue but don't produce output

            logging.info(
                f"RRMLInferenceThread: Successfully generated inference result"
            )

            # Store inference result to database
            self._store_inference_result(inference_request, inference_result)

            output_channels = inference_result
            output_channels["agent_id"] = self.agent_id

            output_message = {"timestamp": time.time(), "channels": output_channels}
            output_topic = self.config.get("kafka_topics", {}).get(
                "output", "ridge-regression-predictions"
            )
            kafka_topic = self.producer.sanitize_topic_name(output_topic)
            return True, [(kafka_topic, json.dumps(output_message))]

        except Exception as e:
            logging.error(f"RRMLInferenceThread: Error processing message: {e}")
            return False, []

    def _store_inference_result(
        self, inference_request: Dict[str, Any], inference_result: Dict[str, Any]
    ):
        """
        Store inference result to database using DBManager's record_prediction function.

        Args:
            inference_request: Original inference request with timestamp
            inference_result: Result from perform_inference containing prediction
        """
        try:
            if inference_result.get("status") != "success":
                logging.warning(f"RRMLInferenceThread: Not storing failed inference result")
                return

            # Extract timestamp from inference request
            source_timestamp = inference_request.get("timestamp")
            if source_timestamp is None:
                logging.warning("RRMLInferenceThread: No timestamp in inference request, cannot store result")
                return

            # Convert timestamp to proper format if needed
            if isinstance(source_timestamp, (int, float)):
                timestamp_dt = datetime.fromtimestamp(source_timestamp)
                source_timestamp_str = timestamp_dt.strftime("%Y-%m-%d %H:%M:%S.%f")
            else:
                source_timestamp_str = str(source_timestamp)

            # Create prediction array from inference result
            prediction = inference_result.get("prediction")
            if prediction is None:
                logging.warning("RRMLInferenceThread: No prediction in inference result, cannot store")
                return

            # Convert prediction to numpy array
            if isinstance(prediction, list):
                prediction_array = np.array(prediction, dtype=np.float32)
            elif isinstance(prediction, np.ndarray):
                prediction_array = prediction.astype(np.float32)
            else:
                prediction_array = np.array([prediction], dtype=np.float32)

            # Get current timestamp for prediction timestamp
            prediction_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

            # Use record_prediction to store the inference result
            status = self.db_manager.record_prediction(
                prediction=prediction_array,
                prediction_timestamp=prediction_timestamp,
                key_value=source_timestamp_str,
                key="state_source_timestamp",
            )

            if status == 0:
                logging.info(
                    f"RRMLInferenceThread: Successfully stored prediction for timestamp {source_timestamp_str}"
                )
            else:
                logging.error(
                    f"RRMLInferenceThread: Failed to store prediction, status: {status}"
                )

        except Exception as e:
            logging.error(f"RRMLInferenceThread: Error storing inference result: {e}")


class RidgeRegressionAgent(AgentBase):
    """
    Ridge regression agent for predicting target EPICS PV from other sensor channels.
    """

    def __init__(self, config_path: str = None, config_key: str = None):
        super().__init__("RidgeRegressionAgent")

        # Load configuration
        if config_path:
            config_loader = ConfigLoader(config_path)
            if config_key is None:
                config_key = "ridge_regression_agent"
            rr_config = config_loader.config.get(config_key, {})


        else:
            logging.warning("No config_path provided, exiting RidgeRegressionAgent initialization")
            return

        # Inference-only
        self.enabled_threads = ["inference"]

        # Ensure kafka_topics are included in the config passed to threads
        if "kafka_topics" not in rr_config:
            logging.warning("RRAgent: No kafka_topics found in config, using defaults")
            rr_config["kafka_topics"] = {
                "input": "sensor-data",
                "output": "ridge-regression-predictions",
            }

        # Add agent_id to config for threads to use
        self.agent_config = rr_config.copy()
        self.agent_config["agent_id"] = self.agent_id

        logging.info(
            f"RRAgent: RidgeRegressionAgent initialized in inference-only mode"
        )
        logging.info(f"RRAgent: Enabled threads: {self.enabled_threads}")

    def create_data_ingest_component(self):
        """Data ingestion not used in inference-only mode."""
        return None

    def create_ml_training_component(self):
        """Training not used in inference-only mode."""
        return None

    def create_ml_inference_component(self):
        """Create ML inference thread component."""
        if "inference" in self.enabled_threads:
            return RidgeRegressionMLInferenceThread(self.agent_id, self.agent_config)
        return None


def main():
    """Main entry point for ridge regression inference agent."""
    parser = argparse.ArgumentParser(
        description='Ridge Regression Inference Agent - Run ridge regression inference on EPICS data via Kafka',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic agent with default config
  python -m smocs.agents.ridge_regression_agent

  # Custom agent config key
  python -m smocs.agents.ridge_regression_agent --agent_config_key ridge_agent1
        """
    )

    parser.add_argument(
        "--agent_config_key",
        help="Key for agent configuration dict in the config file",
        type=str,
        default="ridge_regression_agent",
    )

    args = parser.parse_args()

    setup_logging()

    config_path = os.getenv("CONFIG_PATH", "/app/config.yaml")
    config_key = args.agent_config_key

    try:
        agent = RidgeRegressionAgent(config_path, config_key)
        agent.start()

    except KeyboardInterrupt:
        logging.info("Shutting down ridge regression inference agent...")
    except Exception as e:
        logging.error(f"Error running ridge regression inference agent: {e}")
        raise


if __name__ == "__main__":
    main()
