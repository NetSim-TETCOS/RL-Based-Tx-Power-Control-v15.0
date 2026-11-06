"""
env_general.py  -  Gymnasium environment for 5G RL Tx-Power Control
Communicates with NetSim via the netsim_socket_client pyd (NetSim v15 API).

Protocol (C drives the clock):
  Init:
    1. connect() to NetSim - NetSim blocks until Python connects.
    2. Receive handshake      -> num_gNBs (int), num_UEs (int).
    3. Receive random SINRs   -> sinrs[num_UEs], tputs=0, reward=0.
    4. Send dummy reply       -> SEG_INT 0  (completes C's paired exchange).

  Per RL step (C pushes first):
    5. Receive state  -> sinrs[num_UEs] (double[]), tputs[num_UEs] (double[]), reward (double).
    6. Compute action -> new_powers[num_gNBs].
    7. Send action    -> new_powers (double[]).

Observation : [sinrs (num_UEs), tputs (num_UEs), gNB_powers (num_gNBs)]
Action      : Discrete(5^num_gNBs) combinatorial index into {-3,-1,0,+1,+3} dBm per gNB
Reward      : raw sum throughput (Mbps)
"""

import sys
import os
import time
import numpy as np
import torch
import random
import gymnasium as gym
from gymnasium import spaces

# -- netsim_socket_client.pyd lives in the same folder as this script -----------
_netsim_root = os.environ.get("NETSIM_INSTALL", r"C:\Program Files\NetSim\Standard_v15_0")
for _dll_dir in [os.path.join(_netsim_root, "bin", "bin_x64"),
                 os.path.join(_netsim_root, "bin")]:
    if os.path.isdir(_dll_dir):
        os.add_dll_directory(_dll_dir)

_client_dir = os.path.dirname(os.path.abspath(__file__))
if _client_dir not in sys.path:
    sys.path.insert(0, _client_dir)

from netsim_socket_client import (
    connect, Message, add_variable,
    send_and_receive,
    setup_logging,
    SEG_INT, SEG_DOUBLE_ARRAY,
)

# ------------------------------------------------------------------------------

class generalized(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        host="127.0.0.1",
        port=5555,
        num_gNBs=3,
        num_UEs=6,
        seed_val=31,
        num_ep=1000,
        num_step=500,
        log_to_console=False,
        log_to_file=False,
        log_path="netsim_rl.log",
    ):
        super().__init__()

        setup_logging(log_path=log_path, to_console=log_to_console, to_file=log_to_file)

        self.host     = host
        self.port     = port
        self.num_gNBs = num_gNBs
        self.num_UEs  = num_UEs

        self.adjustments  = [-3, -1, 0, 1, 3]
        self.gNB_powers   = [40.0] * self.num_gNBs
        self.P_MIN, self.P_MAX = 27.0, 46.0

        self.num_ep   = num_ep
        self.num_step = num_step

        self._build_spaces()

        self.state = np.zeros(self.num_UEs * 2 + self.num_gNBs, dtype=np.float64)
        self.it     = 0
        self.epi    = 0
        self.total_rewards: list[float]        = []
        self.total_rewards_per_ep: list[float] = []

        self.sock     = None
        self.seed_val = seed_val
        self._seed(seed_val)
        self.render_mode = None

    # -- helpers ----------------------------------------------------------------

    def _build_spaces(self):
        self.num_actions = len(self.adjustments) ** self.num_gNBs
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=-500.0, high=500.0,
            shape=(self.num_UEs * 2 + self.num_gNBs,),
            dtype=np.float64,
        )

    def _seed(self, seed=None):
        self.seed_val = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    def _action_to_deltas(self, action) -> list:
        action_index = int(action)
        deltas = []
        for _ in range(self.num_gNBs):
            deltas.append(self.adjustments[action_index % len(self.adjustments)])
            action_index //= len(self.adjustments)
        return deltas

    def _build_state(self, sinrs, tputs):
        return np.concatenate([
            np.array(sinrs, dtype=np.float64),
            np.array(tputs, dtype=np.float64),
            np.array(self.gNB_powers, dtype=np.float64),
        ])

    def _compute_reward(self, tputs):
        return float(np.sum(tputs))

    # -- episode lifecycle ------------------------------------------------------

    def _connect_and_handshake(self):
        print(f"[env] Waiting for NetSim at {self.host}:{self.port} ...")
        while True:
            try:
                self.sock = connect(self.host, self.port)
                print("[env] Connected.")
                break
            except Exception:
                time.sleep(0.5)

        handshake = send_and_receive(self.sock, None, expect_reply=True)
        handshake.reset_cursor()
        gnbs = handshake.get_value()
        ues  = handshake.get_value()

        if gnbs is not None and ues is not None:
            self.num_gNBs = int(gnbs)
            self.num_UEs  = int(ues)
            print(f"[env] Handshake: num_gNBs={self.num_gNBs}, num_UEs={self.num_UEs}")
            self._build_spaces()
            self.gNB_powers = [40.0] * self.num_gNBs
        else:
            print("[env] Warning: handshake values missing, using defaults.")

        init_msg = send_and_receive(self.sock, None, expect_reply=True)
        sinrs = np.zeros(self.num_UEs, dtype=np.float64)
        if init_msg is not None:
            init_msg.reset_cursor()
            sinrs_raw = init_msg.get_value()
            if sinrs_raw is not None:
                sinrs = np.array(list(sinrs_raw), dtype=np.float64)
                print(f"[env] Initial SINRs: {sinrs}")

        self.state = self._build_state(sinrs, np.zeros(self.num_UEs))

        dummy = Message()
        add_variable(dummy, SEG_INT, 0)
        send_and_receive(self.sock, dummy, expect_reply=False)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._seed(seed if seed is not None else self.seed_val)

        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

        if self.it != 0:
            self.epi += 1
            avg_r = sum(self.total_rewards) / self.it if self.it else 0.0
            self.total_rewards_per_ep.append(avg_r)
            print(f"[env] Episode {self.epi} ended | avg reward = {avg_r:.4f}")

        self.gNB_powers = [40.0] * self.num_gNBs
        self.total_rewards.clear()
        self.it = 0

        return self.state.copy(), {}

    # -- core step --------------------------------------------------------------

    def step(self, action):
        if self.it == 0:
            self._connect_and_handshake()

        self.it += 1

        deltas = self._action_to_deltas(action)
        self.gNB_powers = [
            max(self.P_MIN, min(self.P_MAX, p + dp))
            for p, dp in zip(self.gNB_powers, deltas)
        ]

        ok, sinrs, tputs, _ = self._netsim_exchange()

        if not ok:
            rew = self.total_rewards[-1] if self.total_rewards else 0.0
            return self.state.copy(), rew, True, False, {}

        reward = self._compute_reward(tputs)
        self.state = self._build_state(sinrs, tputs)
        self.total_rewards.append(reward)

        return self.state.copy(), reward, False, False, {}

    # -- NetSim protocol --------------------------------------------------------

    def _netsim_exchange(self):
        dummy = np.zeros(self.num_UEs)
        try:
            state_msg = send_and_receive(self.sock, None, expect_reply=True)
            if state_msg is None:
                return False, dummy, dummy, 0.0

            state_msg.reset_cursor()
            sinrs_raw = state_msg.get_value()
            tputs_raw = state_msg.get_value()
            reward    = state_msg.get_value()

            if sinrs_raw is None or tputs_raw is None or reward is None:
                return False, dummy, dummy, 0.0

            sinrs  = list(sinrs_raw)
            tputs  = list(tputs_raw)
            reward = float(reward)

            action_msg = Message()
            add_variable(action_msg, SEG_DOUBLE_ARRAY,
                         self.gNB_powers, len(self.gNB_powers))
            send_and_receive(self.sock, action_msg, expect_reply                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                