import time
import numpy as np
import gymnasium as gym


class EnvUITF(gym.Env):
    """Continuous control environment for the UITF, backed by live EPICS PVs.

    PV connections are deferred to the first reset()/step() call, so the env can be
    built just to read its spaces (e.g. by RLControlAgent) without touching EPICS.
    """

    def __init__(
            self,
            obs_pv_names: list[str],
            obs_min: list[float],
            obs_max: list[float],
            obs_rtol: list[float],
            obs_atol: list[float],
            action_pv_names: list[str],
            action_min: list[float],
            action_max: list[float],
            action_rtol: list[float],
            action_atol: list[float],
            action_design_settings: list[float],
            action_min_perturbation: list[float],
            action_max_perturbation: list[float],
            bpm_pv_names: list[str],
            bpm_pv_refs: list[float],
            bpm_terminate_atol: list[float],
            transient_time: float = 0.1,
            put_timeout: float = 5.0,
            connection_timeout: float = 5.0,
            mode: str = 'delta',
            action_abs_min: list[float] | None = None,
            action_abs_max: list[float] | None = None,
            dry_run: bool = False
    ) -> None:
        """
        Initialize the UITF environment.

        Parameters:
        - obs_pv_names: list of observation PV names. e.g., 'IPMK101.XPOS' & 'IPMK101.YPOS' are BPMs. We need to confirm that these are ABSOLUTE positions.
        - obs_min: list of minimum values for each observation PV
        - obs_max: list of maximum values for each observation PV
        - obs_rtol: relative tolerance for each observation PV
        - obs_atol: absolute tolerance for each observation PV
        - action_pv_names: list of action PV names
        - action_min: list of minimum values for each action PV
        - action_max: list of maximum values for each action PV
        - action_rtol: relative tolerance for each action PV
        - action_atol: absolute tolerance for each action PV
        - action_design_settings: "Optimal" design settings for action PVs
        - action_min_perturbation: minimum allowed deviation from design settings for action PVs
        - action_max_perturbation: maximum allowed deviation from design settings for action PVs
        - bpm_pv_names: list of BPM PV names
        - bpm_pv_refs: reference values for the BPM PVs to match. Note that the BPM values read are absolute positions, NOT relative
        - bpm_terminate_atol: absolute tolerance for the BPM termination condition
        - transient_time: time to wait (s) after setting an action PV before reading back
        - put_timeout: max time (s) to wait for the IOC to complete processing of all action puts before raising
        - connection_timeout: max time (s) to wait for each PV to connect in connect() before raising
        - mode: 'delta' or 'abs' mode for actions
        - action_abs_min: absolute minimum setpoint for each action PV. Required in 'delta' mode; in 'abs' mode optional and must equal action_min
        - action_abs_max: absolute maximum setpoint for each action PV. Required in 'delta' mode; in 'abs' mode optional and must equal action_max
        - dry_run: if True, nothing is ever written to the PVs: reset() keeps the current setpoints and step() ignores the agent's action.
        """
        super().__init__()

        # PV names from config; the PV objects are created in connect()
        self.obs_pv_names = obs_pv_names
        self.observation_space = gym.spaces.Box(low=np.array(obs_min, dtype='f4'), high=np.array(obs_max, dtype='f4'), dtype='f4')
        self.obs_rtol = np.array(obs_rtol, dtype='f4')
        self.obs_atol = np.array(obs_atol, dtype='f4')
        assert np.all(0 <= self.obs_atol), f'obs_atol must be non-negative but got {self.obs_atol}'

        self.action_pv_names = action_pv_names
        self.action_space = gym.spaces.Box(low=np.array(action_min, dtype='f4'), high=np.array(action_max, dtype='f4'), dtype='f4')
        self.action_rtol = np.array(action_rtol, dtype='f4')
        self.action_atol = np.array(action_atol, dtype='f4')
        assert np.all(0 <= self.action_atol), f'action_atol must be non-negative but got {self.action_atol}'
        self.action_design_settings = np.array(action_design_settings, dtype='f4')
        self.action_min_perturbation = np.array(action_min_perturbation, dtype='f4')
        self.action_max_perturbation = np.array(action_max_perturbation, dtype='f4')

        self.bpm_pv_names = bpm_pv_names
        self.bpm_pv_refs = np.array(bpm_pv_refs, dtype='f4')
        self.bpm_terminate_atol = np.array(bpm_terminate_atol, dtype='f4')
        assert np.all(0 <= self.bpm_terminate_atol), f'bpm_terminate_atol must be non-negative but got {self.bpm_terminate_atol}'

        self.transient_time = transient_time
        self.put_timeout = put_timeout
        self.connection_timeout = connection_timeout
        self.mode = mode
        self.dry_run = dry_run
        
        assert self.mode in {'delta', 'abs'}, f'mode must be either "delta" or "abs" but got {self.mode} !!!'

        if self.mode == 'delta':
            # action_space bounds the delta, so the absolute setpoint needs its own bounds
            if action_abs_min is None or action_abs_max is None:
                raise ValueError("action_abs_min and action_abs_max are required when using delta mode")
            self.action_abs_min = np.array(action_abs_min, dtype='f4')
            self.action_abs_max = np.array(action_abs_max, dtype='f4')
            assert np.all(self.action_abs_min <= self.action_abs_max), f'action_abs_min must be <= action_abs_max but got {self.action_abs_min} vs {self.action_abs_max}'

            # The agent outputs a change, so it must observe the current setpoints to know what it is changing
            obs_pv_names_set = set(obs_pv_names)
            missing = [name for name in self.action_pv_names if name not in obs_pv_names_set]
            if missing:
                raise ValueError(f"In delta mode, all action PVs must also be observation PVs so the agent sees the current setpoints. Missing: {missing}")
            abs_min, abs_max = self.action_abs_min, self.action_abs_max
        else:
            # action_space already bounds the absolute setpoint; action_abs_min/max are redundant but must match if given
            assert action_abs_min is None or np.array_equal(np.array(action_abs_min, dtype='f4'), self.action_space.low), \
                f'In abs mode, action_abs_min must equal action_min but got {action_abs_min} vs {self.action_space.low}'
            assert action_abs_max is None or np.array_equal(np.array(action_abs_max, dtype='f4'), self.action_space.high), \
                f'In abs mode, action_abs_max must equal action_max but got {action_abs_max} vs {self.action_space.high}'
            abs_min, abs_max = self.action_space.low, self.action_space.high

        # reset() applies design settings +/- perturbation without clipping, so they must lie within the absolute bounds
        reset_min = self.action_design_settings + self.action_min_perturbation
        reset_max = self.action_design_settings + self.action_max_perturbation
        assert np.all(abs_min <= reset_min) and np.all(reset_max <= abs_max), \
            f'Reset setpoints [{reset_min}, {reset_max}] must lie within [{abs_min}, {abs_max}]'

        self.bpm_idxs_in_obs = []
        for name in self.bpm_pv_names:
            try:
                idx = self.obs_pv_names.index(name)
            except ValueError:
                raise ValueError(f"All BPM PVs must be part of the observation PVs. Missing: {name}")
            self.bpm_idxs_in_obs.append(idx)

        self.step_count = 0
        self.is_connected = False
        # The following are added only for the close method not to fail if called before connect() is called
        self.obs_pvs = []
        self.action_pvs = []

    
    def connect(self):
        """Create the EPICS PV objects (called lazily on first reset/step)."""
        import epics                    # not imported at the top to enable rl agent to instantiate the env object without requiring pyepics to be installed
        self.obs_pvs = [epics.PV(name) for name in self.obs_pv_names]
        self.action_pvs = [epics.PV(name) for name in self.action_pv_names]
        # Fail fast on misspelled names or unreachable IOCs instead of reading None later
        unconnected = [pv.pvname for pv in self.obs_pvs + self.action_pvs if not pv.wait_for_connection(timeout=self.connection_timeout)]
        if unconnected:
            raise ConnectionError(f'Could not connect to PVs within {self.connection_timeout}s: {unconnected}')
        self.is_connected = True


    def _read_pvs(self, pvs) -> np.ndarray:
        """Read the given PVs, raising if any reading is invalid so it never reaches the agent as NaN."""
        values = np.array([pv.get() for pv in pvs], dtype='f4')        # a failed read (None) becomes NaN here
        if not np.isfinite(values).all():
            bad = [pv.pvname for pv, value in zip(pvs, values) if not np.isfinite(value)]
            raise RuntimeError(f'[step {self.step_count}] Invalid readings (None/NaN/inf) from PVs: {bad}')
        return values


    def _apply_action(self) -> None:
        # NOTE: self.action ALWAYS contains the absolute setpoints to be applied, even in 'delta' mode where the agent outputs changes relative to the last setpoints
        # 1) Parallel network dispatch: each PV's put_complete flips to True once its IOC finishes processing
        for pv, action_item in zip(self.action_pvs, self.action.tolist()):
            pv.put(action_item, use_complete=True)
        # 2) Block until all puts complete. pyepics updates put_complete from its own callback thread, so no polling is needed
        deadline = time.monotonic() + self.put_timeout
        while not all(pv.put_complete for pv in self.action_pvs):
            if time.monotonic() > deadline:
                pending = [pv.pvname for pv in self.action_pvs if not pv.put_complete]
                raise TimeoutError(f'[step {self.step_count}] Action puts did not complete within {self.put_timeout}s: {pending}')
            time.sleep(0.01)
        # 3) Let the machine settle, then make sure the action PVs actually hold the requested values
        time.sleep(self.transient_time)
        readback = self._read_pvs(self.action_pvs)
        if not np.allclose(self.action, readback, rtol=self.action_rtol, atol=self.action_atol):
            raise ValueError(f'[step {self.step_count}] Action PVs did not accept the requested values!!! self.action: {self.action}, Readback: {readback}')


    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        super().reset(seed=seed)
        if not self.is_connected:
            self.connect()
        self.step_count = 0
        if self.dry_run:
            self.action = self._read_pvs(self.action_pvs)
        else:
            # Reset action to design settings with random uniform perturbation
            self.action = self.action_design_settings + self.np_random.uniform(low=self.action_min_perturbation, high=self.action_max_perturbation)
            self._apply_action()
        self.obs = self._read_pvs(self.obs_pvs)         # Assumptions: 1) We are reading ABSOLUTE BPM values. 2) Corrector PVs can be both read and set
        return self.obs, {}


    def step(self, action: np.ndarray):
        """Take an action and observe the result."""
        action = np.asarray(action, dtype='f4')     # match the f4 setpoints so the exact clipping checks below aren't fooled by float64 inputs
        if not self.is_connected:
            self.connect()
        self.step_count += 1        # incremented first so error messages raised during this step report its number

        # Make sure obs PVs haven't changed
        current_obs = self._read_pvs(self.obs_pvs)
        if not np.allclose(self.obs, current_obs, rtol=self.obs_rtol, atol=self.obs_atol):  # Check if the current observation matches the previous one
            raise ValueError(f'[step {self.step_count}] Observation PVs have changed BEFORE applying action!!! self.obs: {self.obs}, Current obs: {current_obs}')

        # Make sure absolute actions haven't changed
        current_action = self._read_pvs(self.action_pvs)
        if not np.allclose(self.action, current_action, rtol=self.action_rtol, atol=self.action_atol):
            raise ValueError(f'[step {self.step_count}] Action PVs have changed BEFORE applying action!!! self.action: {self.action}, Current action: {current_action}')

        info = {}
        # NOTE: action CAN be delta, but self.action is ALWAYS absolute
        if self.dry_run:
            self.action = current_action
            if self.mode == 'abs':
                info['applied_action'] = self.action
            else:
                info['applied_action'] = np.zeros_like(self.action, dtype='f4')
        else:
            clipped_action = np.clip(action, self.action_space.low, self.action_space.high)     # This is enough clipping in 'abs' mode, delta mode will be clipped later against absolute limits
            if self.mode == 'abs':
                self.action = clipped_action
                if not np.array_equal(clipped_action, action):
                    # Change in absolute action due to clipping, hence the env is overriding the requested absolute action ==> record it in info
                    info['applied_action'] = clipped_action
            else:
                old_abs_action = self.action
                self.action = np.clip(clipped_action + old_abs_action, self.action_abs_min, self.action_abs_max)
                # if any clipping had any effect, record the applied action in info
                if not np.array_equal(self.action, action + old_abs_action):
                    # Change in OVERALL absolute action due to clipping, hence the env is overriding the requested delta action ==> record the effective delta action in info
                    info['applied_action'] = self.action - old_abs_action
            self._apply_action()

        self.obs = self._read_pvs(self.obs_pvs)
        reward = -1 * np.sum((self.obs[self.bpm_idxs_in_obs] - self.bpm_pv_refs)**2).item()

        # set terminated to True if any BPM is outside the min/max specified in self.observation_space +/- bpm_terminate_atol (+ for min, - for max)
        terminated = (self.obs[self.bpm_idxs_in_obs] < self.observation_space.low[self.bpm_idxs_in_obs] + self.bpm_terminate_atol).any().item() or \
                     (self.observation_space.high[self.bpm_idxs_in_obs] - self.bpm_terminate_atol < self.obs[self.bpm_idxs_in_obs]).any().item()
        truncated = False

        return self.obs, reward, terminated, truncated, info


    def close(self):
        """Disconnect from all PVs."""
        for pv in self.obs_pvs + self.action_pvs:
            pv.disconnect()
        self.is_connected = False
