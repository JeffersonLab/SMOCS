import os
import json
import time
import logging
import argparse
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from tensorflow import keras
from tensorflow.keras import layers

from smocs.agents.autoencoder_agent import (
    AutoencoderAgent,
    AutoencoderDataIngestThread,
    AutoencoderMLInferenceThread,
    AutoencoderMLTrainingThread,
)
from smocs.preprocessing import PreprocessingManager
from smocs.utils import ChannelFilter, setup_logging


def _make_context_preprocessing_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build a PreprocessingManager-compatible config for the context channels."""
    model_input = config.get('model_input', {})
    return {
        'window_size': config['window_size'],
        'preprocessing_pipeline': config.get('preprocessing_pipeline', []),
        'model_input': {
            'channels': model_input.get('context_channels', []),
            'bounds': model_input.get('context_bounds', []),
        },
    }


def _parse_context_atol(raw_tol, n_context_channels: int) -> np.ndarray:
    """
    Parse context_tolerance from config.
    Always returns shape (1, n_context_channels) so that np.allclose comparisons
    against (window_size, n_context_channels) arrays broadcast unambiguously along
    the time axis — avoiding the (10,) vs (10,) shape collision when
    window_size == n_context_channels.
    """
    if isinstance(raw_tol, (list, tuple)):
        if len(raw_tol) == 1:
            values = [raw_tol[0]] * n_context_channels
        elif len(raw_tol) == n_context_channels:
            values = list(raw_tol)
        else:
            raise ValueError(
                f"context_tolerance list length ({len(raw_tol)}) must be 1 or "
                f"match number of context channels ({n_context_channels})"
            )
    else:
        values = [float(raw_tol)] * n_context_channels
    return np.array(values, dtype=np.float64).reshape(1, n_context_channels)


# ---------------------------------------------------------------------------
# Data ingest thread
# ---------------------------------------------------------------------------

class ConditionalAutoencoderDataIngestThread(AutoencoderDataIngestThread):
    """
    Extends the plain AE ingest thread so that both input channels and context
    channels are written into the state blob.  Everything else (switch check,
    DB write) is inherited unchanged.
    """

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        # Rebuild channel filter to capture input + context in one pass.
        # The base class built it with input channels only.
        model_input = config.get('model_input', {})
        input_channels = model_input.get('channels', [])
        context_channels = model_input.get('context_channels', [])
        if input_channels and context_channels:
            self.channel_filter = ChannelFilter(input_channels + context_channels)
            logging.info(
                f"CAEDataIngestThread: channel filter rebuilt — "
                f"{len(input_channels)} input + {len(context_channels)} context channels"
            )


# ---------------------------------------------------------------------------
# Training thread
# ---------------------------------------------------------------------------

class ConditionalAutoencoderMLTrainingThread(AutoencoderMLTrainingThread):
    """
    Extends the plain AE training thread with a context-conditioned architecture.

    The state blob stored by the ingest thread contains
        [input_channels | context_channels]
    in that order.  get_training_data() returns a single numpy array with the
    same layout so that training_loop()'s .shape access does not break;
    train_model() / eval_model() split it at the input_split index.
    """

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        model_input = config.get('model_input', {})
        # Set before super().__init__() because load_existing_model() is called
        # from super and our override of _validate_architecture_compatibility reads these.
        self.n_input_channels = len(model_input.get('channels', []))
        self.n_context_channels = len(model_input.get('context_channels', []))
        self.context_dim = None

        super().__init__(agent_id, config)

        # Context preprocessing manager (super created the input one already)
        self.context_preprocessing_manager = PreprocessingManager(
            _make_context_preprocessing_config(config)
        )

    # -- model persistence --------------------------------------------------

    def load_existing_model(self):
        super().load_existing_model()
        if self.model is not None:
            try:
                with open("/app/models/latest_model.json", 'r') as f:
                    metadata = json.load(f)
                self.context_dim = metadata.get('context_dim')
            except Exception:
                pass

    def _validate_architecture_compatibility(self, saved_arch: Dict) -> bool:
        if not super()._validate_architecture_compatibility(saved_arch):
            return False
        saved_ctx = saved_arch.get('context_dim')
        expected_ctx = self.n_context_channels * self.window_size
        if saved_ctx is not None and saved_ctx != expected_ctx:
            logging.warning(
                f"CAEMLTrainingThread: context_dim mismatch: "
                f"saved={saved_ctx}, expected={expected_ctx}"
            )
            return False
        return True

    def save_model(self, model_metrics: Dict[str, Any], eval_results: Dict[str, Any]):
        # Let super() handle the atomic file write, then patch context_dim in.
        super().save_model(model_metrics, eval_results)
        latest_file = "/app/models/latest_model.json"
        try:
            with open(latest_file, 'r') as f:
                metadata = json.load(f)
            metadata['context_dim'] = self.context_dim
            metadata['architecture_config']['context_dim'] = self.context_dim
            with open(latest_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logging.error(f"CAEMLTrainingThread: Failed to patch context_dim into metadata: {e}")

    # -- model architecture -------------------------------------------------

    def build_model(self, input_dim: int, context_dim: int = None):
        """
        Functional Keras model:
          encoder : concat(input, context) → bottleneck
          decoder : concat(bottleneck, context) → reconstructed_input
        The decoder target is the input only; context is never reconstructed.
        """
        if context_dim is None:
            context_dim = self.n_context_channels * self.window_size

        self.context_dim = context_dim

        inp = keras.Input(shape=(input_dim,), name='input_data')
        ctx = keras.Input(shape=(context_dim,), name='context_data')

        x = layers.Concatenate()([inp, ctx])
        for i, dim in enumerate(self.encoder_dims):
            x = layers.Dense(dim, activation='relu', name=f'encoder_{i}')(x)
        bottleneck = x

        x = layers.Concatenate()([bottleneck, ctx])
        for i, dim in enumerate(reversed(self.encoder_dims[:-1])):
            x = layers.Dense(dim, activation='relu', name=f'decoder_{i}')(x)
        out = layers.Dense(input_dim, activation='linear', name='output')(x)

        self.model = keras.Model(inputs=[inp, ctx], outputs=out)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae'],
        )
        self.input_dim = input_dim

        logging.info(
            f"CAEMLTrainingThread: Built CAE — "
            f"input_dim={input_dim}, context_dim={context_dim}, "
            f"encoder_dims={self.encoder_dims}"
        )

    # -- data / training / eval ---------------------------------------------

    def _input_split(self) -> int:
        """Column index separating input from context in a combined window array."""
        return self.window_size * self.n_input_channels

    def get_training_data(self) -> Optional[np.ndarray]:
        """
        Returns shape (n_windows, window_size*(n_input+n_context)).
        Columns [:input_split] are preprocessed input; [input_split:] are
        preprocessed context.  The single ndarray keeps training_loop()'s
        .shape access happy.
        """
        try:
            total_samples = self.db_manager.get_size("agent_inferences")
            logging.info(
                f"CAEMLTrainingThread: Database has {total_samples} samples "
                f"(need {self.min_training_samples})"
            )

            if total_samples < self.min_training_samples:
                return None
            if total_samples <= self.last_training_count:
                logging.debug("CAEMLTrainingThread: No new data since last training")
                return None

            batch_data = self.db_manager.sample_batch(
                batch_size=self.batch_size * self.samples_multiplier,
                segment_length=self.window_size,
                agent_type="diagnostics",
                mode="latest",
            )

            if batch_data is None or len(batch_data['state']) == 0:
                logging.warning("CAEMLTrainingThread: sample_batch returned no data")
                return None

            # states: (batch, window_size, n_input + n_context)
            states = batch_data['state']
            input_states = states[:, :, :self.n_input_channels]
            context_states = states[:, :, self.n_input_channels:]

            input_windows = self.preprocessing_manager.execute_pipeline(input_states)
            context_windows = self.context_preprocessing_manager.execute_pipeline(context_states)

            if input_windows is None or len(input_windows) == 0:
                logging.error("CAEMLTrainingThread: Preprocessing yielded no windows")
                return None

            combined = np.concatenate([input_windows, context_windows], axis=1)
            self.last_training_count = total_samples

            logging.info(
                f"CAEMLTrainingThread: Prepared {len(combined)} windows "
                f"(combined shape {combined.shape})"
            )
            return combined

        except Exception as e:
            logging.error(f"CAEMLTrainingThread: Error getting training data: {e}")
            return None

    def train_model(self, training_data: np.ndarray) -> Dict[str, Any]:
        try:
            split = self._input_split()
            input_data = training_data[:, :split]
            context_data = training_data[:, split:]

            if self.model is None:
                self.build_model(input_data.shape[1], context_data.shape[1])

            logging.info(
                f"CAEMLTrainingThread: Training — "
                f"input {input_data.shape}, context {context_data.shape}"
            )

            history = self.model.fit(
                [input_data, context_data],
                input_data,          # reconstruct input channels only
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=0.2,
                verbose=0,
            )

            setup_logging()

            return {
                'loss': float(history.history['loss'][-1]),
                'val_loss': float(history.history['val_loss'][-1]),
                'epochs_trained': len(history.history['loss']),
                'training_samples': len(input_data),
            }
        except Exception as e:
            logging.error(f"CAEMLTrainingThread: Error training model: {e}")
            return {'error': str(e)}

    def eval_model(self) -> Dict[str, Any]:
        try:
            if self.model is None:
                return {'error': 'No model to evaluate'}

            combined = self.get_training_data()
            if combined is None:
                return {'error': 'No evaluation data available'}

            split = self._input_split()
            n = min(100, len(combined))
            input_subset = combined[:n, :split]
            context_subset = combined[:n, split:]

            reconstructions = self.model.predict(
                [input_subset, context_subset], verbose=0
            )
            mse_errors = np.mean((input_subset - reconstructions) ** 2, axis=1)

            return {
                'mean_reconstruction_error': float(np.mean(mse_errors)),
                'std_reconstruction_error': float(np.std(mse_errors)),
                'max_reconstruction_error': float(np.max(mse_errors)),
                'anomaly_threshold_95': self.get_anomaly_threshold(mse_errors),
                'eval_samples': n,
            }
        except Exception as e:
            logging.error(f"CAEMLTrainingThread: Error evaluating model: {e}")
            return {'error': str(e)}


# ---------------------------------------------------------------------------
# Inference thread
# ---------------------------------------------------------------------------

class ConditionalAutoencoderMLInferenceThread(AutoencoderMLInferenceThread):
    """
    Extends the plain AE inference thread with:
      - context-conditioned model prediction
      - context-drift detection via np.allclose on raw (un-normalized) context windows
    """

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        model_input = config.get('model_input', {})
        # Must be set before super().__init__() because MLInferenceThreadBase.__init__()
        # calls self.load_model() and our override reads these attributes.
        self.n_input_channels = len(model_input.get('channels', []))
        self.n_context_channels = len(model_input.get('context_channels', []))
        self.context_dim = None

        super().__init__(agent_id, config)

        # Rebuild channel filter: input channels first, context channels second.
        # The base class built it with input channels only.
        input_channels = model_input.get('channels', [])
        context_channels = model_input.get('context_channels', [])
        if input_channels and context_channels:
            self.channel_filter = ChannelFilter(input_channels + context_channels)
            logging.info(
                f"CAEMLInferenceThread: channel filter rebuilt — "
                f"{len(input_channels)} input + {len(context_channels)} context channels"
            )

        # Context preprocessing manager (super created the input one already)
        self.context_preprocessing_manager = PreprocessingManager(
            _make_context_preprocessing_config(config)
        )

        # context_tolerance: scalar or per-channel list (raw/un-normalized units)
        raw_tol = model_input.get('context_tolerance', 1e-3)
        self.context_atol = _parse_context_atol(raw_tol, self.n_context_channels)

        # Last context window (window_size, n_context) where reconstruction was normal.
        # Stored in raw (un-normalized) units so the tolerance is interpretable.
        self.last_normal_context_window: Optional[np.ndarray] = None

    # -- storage ------------------------------------------------------------

    def _store_inference_result(self, inference_request, inference_result):
        # The base class skips storage when status != 'success', but our perform_inference
        # returns status = 'normal'/'anomaly'/'context_drift'.  Pass a copy with
        # status='success' so the base class proceeds to write to the DB.
        super()._store_inference_result(
            inference_request, dict(inference_result, status='success')
        )

    # -- model loading ------------------------------------------------------

    def load_model(self):
        old_version = self.current_model_version
        super().load_model()
        # If a new version was loaded, also pull context_dim from its metadata
        if self.current_model_version != old_version and self.model is not None:
            try:
                with open("/app/models/latest_model.json", 'r') as f:
                    metadata = json.load(f)
                self.context_dim = metadata.get('context_dim')
                logging.info(f"CAEMLInferenceThread: context_dim={self.context_dim}")
            except Exception as e:
                logging.warning(f"CAEMLInferenceThread: Could not read context_dim: {e}")

    # -- inference ----------------------------------------------------------

    def perform_inference(self, inference_request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            self.check_for_model_updates()

            if self.model is None:
                self.load_model()
                if self.model is None:
                    logging.debug("CAEMLInferenceThread: No model available")
                    return None

            # sensor_values: (n_input + n_context,) in channel-filter order
            sensor_values = inference_request['sensor_values']
            self.recent_data.append(sensor_values)

            if len(self.recent_data) > self.window_size * 2:
                self.recent_data = self.recent_data[-self.window_size * 2:]

            if len(self.recent_data) < self.window_size:
                logging.debug("CAEMLInferenceThread: Buffer not yet full")
                return None

            # recent_arr: (window_size, n_input + n_context)
            recent_arr = np.array(
                self.recent_data[-self.window_size:], dtype=np.float32
            )
            input_window_raw = recent_arr[:, :self.n_input_channels]   # (T, n_input)
            context_window_raw = recent_arr[:, self.n_input_channels:]  # (T, n_context)

            try:
                # Add batch dimension, preprocess → (1, T*n_channels)
                input_window = self.preprocessing_manager.execute_pipeline(
                    [input_window_raw]
                )
                context_window = self.context_preprocessing_manager.execute_pipeline(
                    [context_window_raw]
                )
            except Exception as e:
                logging.error(f"CAEMLInferenceThread: Preprocessing failed: {e}")
                return None

            # Model reconstructs input channels only; context is conditioning only
            reconstruction_normalized = self.model.predict(
                [input_window, context_window], verbose=0
            )   # (1, T*n_input)

            # Denormalize reconstruction for human-readable output
            try:
                bounds_normalizer = next(
                    (p for p in self.preprocessing_manager.pipeline
                     if p.__class__.get_name() == 'bounds_normalizer'),
                    None,
                )
                reconstruction_original = (
                    bounds_normalizer.denormalize(reconstruction_normalized)
                    if bounds_normalizer else reconstruction_normalized
                )
            except Exception:
                reconstruction_original = reconstruction_normalized

            # Reconstruction error in normalized space
            error_score = float(
                np.mean((input_window - reconstruction_normalized) ** 2)
            )
            is_anomaly = (
                bool(error_score > self.anomaly_threshold)
                if self.anomaly_threshold else False
            )
            is_drift = False

            # Drift check: only runs when an anomaly is tentatively flagged and
            # a reference normal context window is available.
            # Comparison uses raw (un-normalized) values so context_tolerance
            # is expressed in the same physical units as the channel bounds.
            if is_anomaly and self.last_normal_context_window is not None:
                is_drift = not np.allclose(
                    context_window_raw,
                    self.last_normal_context_window,
                    atol=self.context_atol,
                    rtol=0,
                )
                if is_drift:
                    is_anomaly = False   # reclassify: error is due to context shift

            # Update the reference only on clean normal steps
            if not is_anomaly and not is_drift:
                self.last_normal_context_window = context_window_raw.copy()

            # Most-recent timestep for per-channel output fields
            reconstruction_reshaped = reconstruction_original.reshape(
                self.window_size, self.n_input_channels
            )
            most_recent_reconstruction = reconstruction_reshaped[-1]   # (n_input,)
            most_recent_input = input_window_raw[-1]                   # (n_input,)

            if is_anomaly:
                logging.warning(
                    f"CAEMLInferenceThread: ANOMALY — "
                    f"error={error_score:.4f} > threshold={self.anomaly_threshold:.4f}"
                )
            if is_drift:
                logging.warning(
                    f"CAEMLInferenceThread: CONTEXT DRIFT — "
                    f"error={error_score:.4f}, context window changed"
                )

            return {
                'reconstruction_normalized': reconstruction_normalized.flatten(),
                'reconstruction_original': reconstruction_original.flatten(),
                'original_window_normalized': input_window.flatten(),
                'error_score': error_score,
                'is_anomaly': is_anomaly,
                'is_drift': is_drift,
                'anomaly_threshold': self.anomaly_threshold,
                'model_version': self.current_model_version,
                'timestamp': inference_request['timestamp'],
                'status': (
                    'context_drift' if is_drift
                    else ('anomaly' if is_anomaly else 'normal')
                ),
                'most_recent_input': most_recent_input,
                'most_recent_reconstruction': most_recent_reconstruction,
            }

        except Exception as e:
            logging.error(f"CAEMLInferenceThread: Error in perform_inference: {e}")
            return None

    def process_message(self, message, topic, partition, offset) -> Tuple[bool, List[Tuple]]:
        """
        Mirrors AutoencoderMLInferenceThread.process_message but adds is_drift
        and status to the output channels dict.
        """
        try:
            if isinstance(message, bytes):
                message = message.decode('utf-8')

            message_data = json.loads(message)

            if self.channel_filter:
                filtered_result = self.channel_filter.filter_channels(message_data)
                if filtered_result is None:
                    logging.debug(
                        f"CAEMLInferenceThread: Skipping {topic}:{partition}:{offset} "
                        f"— channel filter rejected"
                    )
                    return True, []
                channel_names, channel_values = filtered_result
            else:
                filtered_result = ChannelFilter.extract_all_channels(message_data)
                if filtered_result is None:
                    return True, []
                channel_names, channel_values = filtered_result

            filtered_channels = dict(zip(channel_names, channel_values))
            message_data['channels'] = filtered_channels

            if not self.switch_fn(filtered_channels):
                logging.debug("CAEMLInferenceThread: Switch OFF, skipping inference")
                return False, []

            inference_request = self.parse_inference_request(
                message_data, topic, partition, offset
            )
            if inference_request is None:
                return False, []

            inference_result = self.perform_inference(inference_request)
            if inference_result is None:
                return False, []

            self._store_inference_result(inference_request, inference_result)

            # Only input channels appear in the _input / _reconstructed pairs;
            # context channels are conditioning only and are not reported per-channel.
            model_input_channels = self.config.get('model_input', {}).get('channels', [])
            output_channels = {
                'agent_id': self.agent_id,
                'error_score': inference_result.get('error_score', 0.0),
                'is_anomaly': inference_result.get('is_anomaly', False),
                'is_drift': inference_result.get('is_drift', False),
                'anomaly_threshold': inference_result.get('anomaly_threshold', 0.0),
                'model_version': inference_result.get('model_version', 0),
                'status': inference_result.get('status', 'unknown'),
            }

            most_recent_input = inference_result.get('most_recent_input')
            most_recent_reconstruction = inference_result.get('most_recent_reconstruction')
            if most_recent_input is not None and most_recent_reconstruction is not None:
                for i, ch in enumerate(model_input_channels):
                    if i < len(most_recent_input) and i < len(most_recent_reconstruction):
                        output_channels[f'{ch}_input'] = float(most_recent_input[i])
                        output_channels[f'{ch}_reconstructed'] = float(most_recent_reconstruction[i])

            output_message = {'timestamp': time.time(), 'channels': output_channels}
            kafka_topic = self.producer.sanitize_topic_name(self.output_topic)
            return True, [(kafka_topic, json.dumps(output_message))]

        except json.JSONDecodeError as e:
            logging.error(f"CAEMLInferenceThread: JSON decode error: {e}")
            return False, []
        except Exception as e:
            logging.error(f"CAEMLInferenceThread: Error processing message: {e}")
            return False, []


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ConditionalAutoencoderAgent(AutoencoderAgent):
    """
    Drop-in replacement for AutoencoderAgent that uses context-conditioned threads.

    Additional config keys under model_input:
      context_channels  : [list of PV/channel names used as context]
      context_bounds    : [[min, max], ...]  — one pair per context channel
      context_tolerance : 0.05  or  [0.05, 0.1, ...]  — drift detection tolerance
                          in raw (un-normalized) channel units
    """

    def create_data_ingest_component(self):
        if 'ingest' in self.enabled_threads:
            return ConditionalAutoencoderDataIngestThread(self.agent_id, self.agent_config)
        return None

    def create_ml_training_component(self):
        if 'training' in self.enabled_threads:
            return ConditionalAutoencoderMLTrainingThread(self.agent_id, self.agent_config)
        return None

    def create_ml_inference_component(self):
        if 'inference' in self.enabled_threads:
            return ConditionalAutoencoderMLInferenceThread(self.agent_id, self.agent_config)
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent_config_key",
        type=str,
        default='conditional_autoencoder_agent1',
        help="Key for this agent's section in config.yaml",
    )
    args = parser.parse_args()

    setup_logging()
    config_path = os.getenv('CONFIG_PATH', '/app/config.yaml')

    try:
        agent = ConditionalAutoencoderAgent(config_path, args.agent_config_key)
        agent.start()
    except KeyboardInterrupt:
        logging.info("Shutting down conditional autoencoder agent...")
    except Exception as e:
        logging.error(f"Error running conditional autoencoder agent: {e}")
        raise


if __name__ == "__main__":
    main()
