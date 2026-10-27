import time
import logging
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
            mode: str = 'delta',
            
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
        - transient_time: time to wait after setting an action PV before reading back
        - mode: 'delta' or 'abs' mode for actions
        """
        super().__init__()

        # PV names from config; the PV objects are created in _connect()
        self.obs_pv_names = obs_pv_names
        self.observation_space = gym.spaces.Box(low=obs_min, high=obs_max, dtype='f4')
        self.obs_rtol = np.array(obs_rtol, dtype='f4')
        self.obs_atol = np.array(obs_atol, dtype='f4')
        assert np.all(0 <= self.obs_atol), f'obs_atol must be non-negative but got {self.obs_atol}'

        self.action_pv_names = action_pv_names
        self.action_space = gym.spaces.Box(low=action_min, high=action_max, dtype='f4')
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
        self.mode = mode
        
        assert self.mode in {'delta', 'abs'}, f'mode must be either "delta" or "abs" but got {self.mode} !!!'
        
        if self.mode == 'delta':
            self.action_idxs_in_obs = []        # only needed in 'delta' mode
            for name in self.action_pv_names:
                try:
                    idx = self.obs_pv_names.index(name)
                except ValueError:
                    raise ValueError(f"All action PVs must be part of the observation PVs when using delta mode. Missing: {name}")
                self.action_idxs_in_obs.append(idx)
        
        self.bpm_idxs_in_obs = []
        for name in self.bpm_pv_names:
            try:
                idx = self.obs_pv_names.index(name)
            except ValueError:
                raise ValueError(f"All BPM PVs must be part of the observation PVs. Missing: {name}")
            self.bpm_idxs_in_obs.append(idx)

        self.step_count = 0
        self.is_connected = False

    
    def connect(self):
        """Create the EPICS PV objects (called lazily on first reset/step)."""
        import epics                    # not imported at the top to enable rl agent to instantiate the env object without requiring pyepics to be installed
        self.obs_pvs = [epics.PV(name) for name in self.obs_pv_names]
        self.action_pvs = [epics.PV(name) for name in self.action_pv_names]
        self.is_connected = True


    def _get_obs(self):
        return np.array([pv.get() for pv in self.obs_pvs], dtype='f4')          # Assumptions: 1) We are reading ABSOLUTE BPM values. 2) Corrector PVs can be both read and set


    def _apply_action(self) -> None:
        # 0) Clip self.action to the allowed range using self.action_space
        self.action = np.clip(self.action, self.action_space.low, self.action_space.high)
        # 1) Parallel network dispatch
        for pv, action_item in zip(self.action_pvs, self.action.tolist()):
            pv.put(action_item, use_complete=True)
        # 2) Synchronous block until IOC updates record processing
        while not all(pv.put_complete for pv in self.action_pvs):
            epics.ca.poll()      # epics is already imported in self.connect()
            time.sleep(0.01)
        time.sleep(self.transient_time)


    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        super().reset(seed=seed)
        if not self.is_connected:
            self.connect()
        # Reset action to design settings with random uniform perturbation
        self.action = self.action_design_settings + np.random.uniform(low=self.action_min_perturbation, high=self.action_max_perturbation)
        self._apply_action()
        self.step_count = 0
        self.obs = self._get_obs()
        return self.obs, {}


    def step(self, action: np.ndarray):
        """Take an action and observe the result."""
        if not self.is_connected:
            self.connect()

        # Make sure obs PVs haven't changed
        current_obs = self._get_obs()
        if not np.allclose(self.obs, current_obs, rtol=self.obs_rtol, atol=self.obs_atol):  # Check if the current observation matches the previous one
            logging.error(f'Observation PVs have changed BEFORE applying action!!! self.obs: {self.obs}, Current obs: {current_obs}')
        
        # Make sure absolute actions haven't changed
        current_action = np.array([pv.get() for pv in self.action_pvs], dtype='f4')
        if not np.allclose(self.action, current_action, rtol=self.action_rtol, atol=self.action_atol):
            logging.error(f'Action PVs have changed BEFORE applying action!!! self.action: {self.action}, Current action: {current_action}')
        
        self.action = action if (self.mode == 'abs') else action + self.obs[self.action_idxs_in_obs]        # NOTE: action can be delta, but self.action is always absolute
        self._apply_action()

        self.obs = self._get_obs()
        reward = -1 * np.sum((self.obs[self.bpm_idxs_in_obs] - self.bpm_pv_refs)**2).item()
        self.step_count += 1

        # set terminated to True if any BPM is outside the min/max specified in self.observation_space +/- bpm_terminate_atol (+ for min, - for max)
        terminated = (self.obs[self.bpm_idxs_in_obs] < self.observation_space.low[self.bpm_idxs_in_obs] + self.bpm_terminate_atol).any().item() or \
                     (self.observation_space.high[self.bpm_idxs_in_obs] - self.bpm_terminate_atol < self.obs[self.bpm_idxs_in_obs]).any().item()
        truncated = False

        return self.obs, reward, terminated, truncated, {}


    def disconnect(self):
        """Disconnect from all PVs."""
        for pv in self.obs_pvs + self.action_pvs:
            pv.disconnect()
        self.is_connected = False
