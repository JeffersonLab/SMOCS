"""
wave_generator_env.py

A continuous, single-sample wave generator Gymnasium environment
designed for testing conditional autoencoder anomaly detection.
"""

import numpy as np
import gymnasium as gym
from copy import copy
from gymnasium import spaces
from enum import IntEnum
from dataclasses import dataclass
from typing import Any, Dict, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class WaveType(IntEnum):
    SINE     = 0
    SQUARE   = 1
    SAWTOOTH = 2
    TRIANGLE = 3


class AnomalyType(IntEnum):
    NONE             = 0
    AMPLITUDE_SPIKE  = 1
    AMPLITUDE_DROP   = 2
    FREQUENCY_SHIFT  = 3
    DRIFT_SHIFT      = 4
    NOISE_BURST      = 5
    WAVE_TYPE_CHANGE = 6
    PHASE_JUMP       = 7


class Command(IntEnum):
    NOOP              = 0
    CONFIGURE         = 1
    INJECT            = 2
    RESET_WITH_PARAMS = 3
    RESET_DEFAULT     = 4


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WaveParams:
    """Parameters fully describing a wave signal."""
    wave_type:   WaveType = WaveType.SINE
    amplitude:   float    = 5.0
    frequency:   int      = 3
    drift:       float    = 0.0
    noise_level: float    = 0.5


@dataclass
class AutoAnomalyConfig:
    """Configuration for automatic probabilistic anomaly injection."""
    enabled:        bool  = True
    p_per_step:     float = 0.002   # ~1 anomaly per 500 steps
    cooldown_steps: int   = 200     # minimum quiet steps between anomalies
    duration_min:   int   = 10
    duration_max:   int   = 300
    magnitude_min:  float = 1.0
    magnitude_max:  float = 10.0


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class WaveGeneratorEnv(gym.Env):
    """
    Continuous single-sample wave generator for anomaly detection testing.

    Emits one wave sample per step. Wave shape, amplitude, frequency, drift,
    and noise are all configurable. Anomalies can be injected via the action
    space or automatically via a configurable per-step probability.

    Observation space (Dict):
        signal    Box(1,)  — current wave sample
        condition Box(4,)  — declared [wave_type, amplitude, frequency, drift]

    Action space (Box float32, shape (9,)):
        [0] command            0–4    see Command enum
        [1] wave_type          0–3    see WaveType enum
        [2] amplitude          1–20   physical units
        [3] frequency          1–20   cycles per reference_period
        [4] drift            -20–20   vertical offset
        [5] noise_level        0–10   Gaussian noise std dev
        [6] anomaly_type       0–7    see AnomalyType enum
        [7] anomaly_magnitude  1–10   severity multiplier
        [8] anomaly_duration   1–500  steps

        Note: The environment uses np.clip() on all parameters when they are
        actually used (CONFIGURE, INJECT commands). The zeros action [0,...,0]
        is semantically a NOOP — parameters at indices 2-8 are ignored when
        command=0 (NOOP).

    Commands:
        0  NOOP              — continue unchanged
        1  CONFIGURE         — apply indices [1–5], update declared condition
        2  INJECT            — inject anomaly from indices [6–8], bypasses cooldown
        3  RESET_WITH_PARAMS — hard reset then apply indices [1–5]
        4  RESET_DEFAULT     — hard reset to config defaults
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        reference_period:        int   = 100,
        default_wave_type:       int   = int(WaveType.SINE),
        default_amplitude:       float = 5.0,
        default_frequency:       int   = 3,
        default_drift:           float = 0.0,
        default_noise_level:     float = 0.5,
        auto_anomaly_enabled:    bool  = True,
        auto_anomaly_p_per_step: float = 0.002,
        auto_anomaly_cooldown:   int   = 200,
        auto_anomaly_dur_min:    int   = 10,
        auto_anomaly_dur_max:    int   = 300,
        auto_anomaly_mag_min:    float = 1.0,
        auto_anomaly_mag_max:    float = 10.0,
    ):
        super().__init__()

        self.reference_period = reference_period

        self.defaults = WaveParams(
            wave_type=WaveType(default_wave_type),
            amplitude=default_amplitude,
            frequency=default_frequency,
            drift=default_drift,
            noise_level=default_noise_level,
        )

        self.auto_anomaly_cfg = AutoAnomalyConfig(
            enabled=auto_anomaly_enabled,
            p_per_step=auto_anomaly_p_per_step,
            cooldown_steps=auto_anomaly_cooldown,
            duration_min=auto_anomaly_dur_min,
            duration_max=auto_anomaly_dur_max,
            magnitude_min=auto_anomaly_mag_min,
            magnitude_max=auto_anomaly_mag_max,
        )

        self.observation_space = spaces.Dict({
            'signal': spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            'condition': spaces.Box(
                low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
            ),
        })

        self.action_space = spaces.Box(
            low=np.array( [0, 0,  1.0,  1, -20.0,  0.0, 0,  1.0,   1], dtype=np.float32),
            high=np.array([4, 3, 20.0, 20,  20.0, 10.0, 7, 10.0, 500], dtype=np.float32),
            dtype=np.float32,
        )

        self._reset_state()

    # ── Public Gymnasium API ──────────────────────────────────────────────────

    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        self._reset_state()
        sample = self._generate_sample()
        return self._build_observation(sample), self._build_info(sample)

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        self._process_action(action)
        self._maybe_auto_anomaly()

        sample = self._generate_sample()

        self._tick_anomaly()
        self._tick_cooldown()
        self._timestep += 1

        return self._build_observation(sample), 0.0, False, False, self._build_info(sample)

    # ── State Initialization ──────────────────────────────────────────────────

    def _reset_state(self):
        """Reset all mutable state to defaults."""
        self._current  = copy(self.defaults)
        self._declared = copy(self.defaults)

        self._phase:    float = 0.0
        self._timestep: int   = 0

        self._anomaly_type:      AnomalyType = AnomalyType.NONE
        self._anomaly_magnitude: float       = 0.0
        self._anomaly_duration:  int         = 0
        self._anomaly_remaining: int         = 0
        self._anomaly_source:    str         = 'none'
        self._anomaly_alt_wave:  WaveType    = WaveType.SINE

        self._cooldown_remaining: int    = 0
        self._last_command:       Command = Command.NOOP

    # ── Action Processing ─────────────────────────────────────────────────────

    def _process_action(self, action: np.ndarray):
        """Decode command and dispatch to the appropriate handler."""
        command = Command(int(np.clip(round(float(action[0])), 0, 4)))
        self._last_command = command

        if command == Command.CONFIGURE:
            self._apply_wave_config(action)

        elif command == Command.INJECT:
            self._inject_anomaly(action, source='action')

        elif command == Command.RESET_WITH_PARAMS:
            self._reset_state()
            self._apply_wave_config(action)

        elif command == Command.RESET_DEFAULT:
            self._reset_state()

        # NOOP falls through — nothing to do

    def _apply_wave_config(self, action: np.ndarray):
        """Update current and declared wave parameters from action indices [1–5]."""
        wave_type   = WaveType(int(np.clip(round(float(action[1])), 0, 3)))
        amplitude   = float(np.clip(action[2],  1.0, 20.0))
        frequency   = int(np.clip(round(float(action[3])), 1, 20))
        drift       = float(np.clip(action[4], -20.0, 20.0))
        noise_level = float(np.clip(action[5],  0.0, 10.0))

        # Both current and declared track wave shape — noise is hidden from declared
        for params in (self._current, self._declared):
            params.wave_type  = wave_type
            params.amplitude  = amplitude
            params.frequency  = frequency
            params.drift      = drift

        self._current.noise_level = noise_level

    def _inject_anomaly(self, action: np.ndarray, source: str):
        """Set anomaly state from action indices [6–8]. Always bypasses cooldown."""
        anomaly_type = AnomalyType(int(np.clip(round(float(action[6])), 0, 7)))
        if anomaly_type == AnomalyType.NONE:
            return

        magnitude = float(np.clip(action[7],  1.0,  10.0))
        duration  = int(np.clip(round(float(action[8])), 1, 500))

        self._start_anomaly(anomaly_type, magnitude, duration, source)

    def _start_anomaly(
        self,
        anomaly_type: AnomalyType,
        magnitude:    float,
        duration:     int,
        source:       str,
    ):
        """
        Activate an anomaly.

        PHASE_JUMP applies its effect immediately and then simply marks the
        period as anomalous — no per-step signal modification is needed.
        All other types modify generation parameters each step for `duration` steps.
        """
        self._anomaly_type      = anomaly_type
        self._anomaly_magnitude = magnitude
        self._anomaly_duration  = duration
        self._anomaly_remaining = duration
        self._anomaly_source    = source

        if anomaly_type == AnomalyType.WAVE_TYPE_CHANGE:
            # Cycle to the next wave type for the anomaly duration
            self._anomaly_alt_wave = WaveType((int(self._current.wave_type) + 1) % len(WaveType))

        elif anomaly_type == AnomalyType.PHASE_JUMP:
            # One-time discontinuity; signal continues normally from the new phase
            self._phase = (self._phase + np.pi * magnitude) % (2.0 * np.pi)

    # ── Auto-Anomaly ──────────────────────────────────────────────────────────

    def _maybe_auto_anomaly(self):
        """Roll per-step probability and inject a random anomaly if it fires."""
        cfg = self.auto_anomaly_cfg

        if (not cfg.enabled
                or self._anomaly_type != AnomalyType.NONE
                or self._cooldown_remaining > 0
                or np.random.random() >= cfg.p_per_step):
            return

        anomaly_type = AnomalyType(np.random.randint(1, len(AnomalyType)))
        duration     = self._exp_sample(cfg.duration_min,  cfg.duration_max)
        magnitude    = self._exp_sample(cfg.magnitude_min, cfg.magnitude_max)

        self._start_anomaly(anomaly_type, float(magnitude), int(duration), source='auto')

    @staticmethod
    def _exp_sample(lo: float, hi: float) -> float:
        """Exponential sample clipped to [lo, hi], biased heavily toward lo."""
        scale = (hi - lo) * 0.2
        return float(np.clip(np.random.exponential(scale) + lo, lo, hi))

    # ── Wave Generation ───────────────────────────────────────────────────────

    def _generate_sample(self) -> float:
        """Compute one sample at the current phase then advance the phase."""
        amplitude   = self._current.amplitude
        frequency   = self._current.frequency
        drift       = self._current.drift
        noise_level = self._current.noise_level
        wave_type   = self._current.wave_type

        # Overlay anomaly modifications (PHASE_JUMP is handled at injection time)
        if self._anomaly_type not in (AnomalyType.NONE, AnomalyType.PHASE_JUMP):
            amplitude, frequency, drift, noise_level, wave_type = \
                self._apply_anomaly_overlay(amplitude, frequency, drift, noise_level, wave_type)

        sample  = amplitude * self._wave_fn(wave_type, self._phase) + drift
        sample += np.random.normal(0.0, noise_level) if noise_level > 0 else 0.0

        self._phase = (
            self._phase + (2.0 * np.pi * frequency) / self.reference_period
        ) % (2.0 * np.pi)

        return float(sample)

    def _apply_anomaly_overlay(
        self,
        amplitude:   float,
        frequency:   int,
        drift:       float,
        noise_level: float,
        wave_type:   WaveType,
    ) -> Tuple[float, int, float, float, WaveType]:
        """Return generation parameters modified by the currently active anomaly."""
        m = self._anomaly_magnitude

        if self._anomaly_type == AnomalyType.AMPLITUDE_SPIKE:
            amplitude = amplitude * m

        elif self._anomaly_type == AnomalyType.AMPLITUDE_DROP:
            amplitude = amplitude / m

        elif self._anomaly_type == AnomalyType.FREQUENCY_SHIFT:
            frequency = int(np.clip(frequency + int(m), 1, 40))

        elif self._anomaly_type == AnomalyType.DRIFT_SHIFT:
            drift = drift + (m * amplitude)

        elif self._anomaly_type == AnomalyType.NOISE_BURST:
            noise_level = noise_level + m

        elif self._anomaly_type == AnomalyType.WAVE_TYPE_CHANGE:
            wave_type = self._anomaly_alt_wave

        return amplitude, frequency, drift, noise_level, wave_type

    @staticmethod
    def _wave_fn(wave_type: WaveType, phase: float) -> float:
        """Evaluate the wave function, normalized to [-1, 1]."""
        if wave_type == WaveType.SINE:
            return float(np.sin(phase))
        elif wave_type == WaveType.SQUARE:
            return float(np.sign(np.sin(phase)))
        elif wave_type == WaveType.SAWTOOTH:
            return float((phase / np.pi) - 1.0)
        else:  # TRIANGLE
            return float((2.0 / np.pi) * np.arcsin(np.sin(phase)))

    # ── Anomaly / Cooldown Ticks ──────────────────────────────────────────────

    def _tick_anomaly(self):
        """Decrement anomaly counter; clear state and start cooldown when done."""
        if self._anomaly_type == AnomalyType.NONE:
            return

        self._anomaly_remaining -= 1

        if self._anomaly_remaining <= 0:
            self._anomaly_type      = AnomalyType.NONE
            self._anomaly_magnitude = 0.0
            self._anomaly_duration  = 0
            self._anomaly_remaining = 0
            self._anomaly_source    = 'none'
            self._cooldown_remaining = self.auto_anomaly_cfg.cooldown_steps

    def _tick_cooldown(self):
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

    # ── Observation & Info ────────────────────────────────────────────────────

    def _build_observation(self, sample: float) -> Dict[str, np.ndarray]:
        return {
            'signal': np.array([sample], dtype=np.float32),
            'condition': np.array([
                float(self._declared.wave_type),
                self._declared.amplitude,
                float(self._declared.frequency),
                self._declared.drift,
            ], dtype=np.float32),
        }

    def _build_info(self, sample: float) -> Dict[str, Any]:
        return {
            'timestep':                  self._timestep,

            # Declared condition — what the CAE is told
            'declared_wave_type':        int(self._declared.wave_type),
            'declared_wave_type_name':   self._declared.wave_type.name,
            'declared_amplitude':        self._declared.amplitude,
            'declared_frequency':        self._declared.frequency,
            'declared_drift':            self._declared.drift,

            # Actual wave state — ground truth
            'actual_wave_type':          int(self._current.wave_type),
            'actual_wave_type_name':     self._current.wave_type.name,
            'actual_amplitude':          self._current.amplitude,
            'actual_frequency':          self._current.frequency,
            'actual_drift':              self._current.drift,
            'actual_noise_level':        self._current.noise_level,
            'current_phase':             self._phase,

            # Anomaly ground truth
            'is_anomaly':                self._anomaly_type != AnomalyType.NONE,
            'anomaly_type':              int(self._anomaly_type),
            'anomaly_type_name':         self._anomaly_type.name,
            'anomaly_steps_remaining':   self._anomaly_remaining,
            'anomaly_duration':          self._anomaly_duration,
            'anomaly_magnitude':         self._anomaly_magnitude,
            'anomaly_source':            self._anomaly_source,

            # Cooldown
            'in_cooldown':               self._cooldown_remaining > 0,
            'cooldown_steps_remaining':  self._cooldown_remaining,

            # Convenience — raw sample also in obs
            'current_sample':            sample,

            # Last command processed
            'last_command':              int(self._last_command),
            'last_command_name':         self._last_command.name,
        }
