"""
agents_netsim.py  -  Train a PPO / A2C agent on the NetSim 5G
                     Tx-Power Control environment (v15 pyd API).

Usage:
    python agents_netsim.py

NetSim must be started (and waiting for a connection) before this script runs.
All outputs (plot, model, vecnorm, log) are saved to a timestamped subfolder
under results/ so runs are never overwritten.
"""

import os
import datetime
import random
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from env_general import generalized

start_time = time.time()

# -- reproducibility ------------------------------------------------------------
SEED = 31
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# ==============================================================================
#  USER SETTINGS — edit these before each run
# ==============================================================================
NUM_GNBs = 3       # default; confirmed/overridden by NetSim handshake
NUM_UEs  = 6       # default; confirmed/overridden by NetSim handshake
PORT     = 5555    # NetSim socket port (fixed)

ALGO     = "PPO"   # "PPO" or "A2C"
EPISODES = 1500
STEPS    = 500

# network architecture
policy_kwargs = dict(
    net_arch=[256, 256],
    activation_fn=torch.nn.Tanh,
)
# ==============================================================================

# -- run output directory -------------------------------------------------------
_base            = os.path.dirname(os.path.abspath(__file__))
_activation_name = policy_kwargs["activation_fn"].__name__
_act_short       = {"Identity": "Id", "Tanh": "Tanh", "ReLU": "ReLU"}.get(_activation_name, _activation_name[:4])
_timestamp       = datetime.datetime.now().strftime("%Y%m%d_%H%M")
_run_name        = f"{_timestamp}_{ALGO}_{_act_short}_{NUM_GNBs}x{NUM_UEs}"
run_dir          = os.path.join(_base, "results", _run_name)
os.makedirs(run_dir, exist_ok=True)
print(f"[run]   Results folder: {run_dir}")

# -- EMA helper -----------------------------------------------------------------
_EMA_ALPHA = 0.05


def _compute_ema(values, alpha=_EMA_ALPHA):
    ema, prev = [], None
    for v in values:
        s = v if prev is None else alpha * v + (1.0 - alpha) * prev
        ema.append(s)
        prev = s
    return ema


# -- LR / clip schedules --------------------------------------------------------
# Warmup for first ~50 episodes, then linear decay.
# This prevents instant convergence and gives the network room to explore early.
_WARMUP_FRAC = 50 / EPISODES   # fraction of training spent warming up (~50 ep)

def warmup_decay(peak: float, final: float = 5e-6):
    """Ramps from 0 → peak over first 50 episodes, then decays to final."""
    def schedule(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining          # 0.0 at start → 1.0 at end
        if progress < _WARMUP_FRAC:
            return peak * (progress / _WARMUP_FRAC)  # linear ramp up
        decay = (progress - _WARMUP_FRAC) / (1.0 - _WARMUP_FRAC)
        return final + (1.0 - decay) * (peak - final)  # linear decay to final
    return schedule


# -- live reward plot callback --------------------------------------------------
_ALGO_COLORS = {"PPO": "steelblue", "A2C": "seagreen"}


class LiveRewardPlot(BaseCallback):
    """Updates a matplotlib window after every completed episode."""

    def __init__(self, algo: str, num_gNBs: int, num_UEs: int, episodes: int, verbose=0):
        super().__init__(verbose)
        self._prev_ep_count = 0
        self._best  = float("-inf")
        self._algo  = algo
        self._num_gNBs = num_gNBs
        self._num_UEs  = num_UEs
        self._episodes = episodes
        color = _ALGO_COLORS.get(algo, "steelblue")

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 5), dpi=100)
        self.raw_line, = self.ax.plot([], [], color=color, lw=1.0, alpha=0.35,
                                      label="per-episode")
        self.ema_line, = self.ax.plot([], [], color=color, lw=2.2,
                                      label=f"EMA (a={_EMA_ALPHA})")
        self.best_line, = self.ax.plot([], [], color="tomato", lw=1.2,
                                       ls="--", alpha=0.75, label="best so far")
        self.ax.set_xlabel("Episode", fontsize=10)
        self.ax.set_ylabel("Avg Sum Throughput (Mbps)", fontsize=10)
        self.ax.legend(fontsize=9, loc="lower right")
        self.ax.grid(True, linestyle="--", alpha=0.4)
        self.ax.spines[["top", "right"]].set_visible(False)
        self._refresh_title(0)
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _refresh_title(self, ep_count):
        best_str = f"{self._best:.1f} Mbps" if self._best > float("-inf") else "--"
        self.ax.set_title(
            f"{self._algo}: Live Reward  |  "
            f"{self._num_gNBs} gNBs · {self._num_UEs} UEs  |  "
            f"ep {ep_count}/{self._episodes}  |  best: {best_str}",
            fontsize=10, fontweight="bold",
        )

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [False])
        if not dones[0]:
            return True

        rewards = self.training_env.get_attr("total_rewards_per_ep")[0]
        ep_count = len(rewards)
        if ep_count <= self._prev_ep_count:
            return True

        self._prev_ep_count = ep_count
        self._best = max(self._best, max(rewards))
        xs  = list(range(ep_count))
        ema = _compute_ema(rewards)

        self.raw_line.set_data(xs, rewards)
        self.ema_line.set_data(xs, ema)
        self.best_line.set_data([0, ep_count - 1], [self._best, self._best])
        self.ax.relim()
        self.ax.autoscale_view()
        self._refresh_title(ep_count)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        return True

    def _on_training_end(self):
        plt.ioff()
        plt.close(self.fig)


# -- build env ------------------------------------------------------------------
_env_fn = lambda: generalized(
    host="127.0.0.1",
    port=PORT,
    num_gNBs=NUM_GNBs,
    num_UEs=NUM_UEs,
    seed_val=SEED,
    num_ep=EPISODES,
    num_step=STEPS,
    log_to_console=False,
    log_to_file=True,
    log_path=os.path.join(run_dir, "netsim_rl.log"),
)

env = DummyVecEnv([_env_fn])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

# -- build agent ----------------------------------------------------------------
if ALGO == "PPO":
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        n_steps=500,
        learning_rate=warmup_decay(peak=2e-4, final=5e-6),  # ramp up 50 ep, then decay
        clip_range=warmup_decay(peak=0.20,  final=0.08),    # same warmup on clip range
        ent_coef=0.01,    # keeps exploration alive without slowing convergence
        batch_size=256,   # stable gradient estimates
        n_epochs=10,
        verbose=0,
    )
elif ALGO == "A2C":
    model = A2C(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        n_steps=20,
        learning_rate=0.002,
        ent_coef=0.01,
        verbose=0,
    )
else:
    raise ValueError(f"Unknown ALGO '{ALGO}'. Choose PPO or A2C.")

# -- train ----------------------------------------------------------------------
plot_cb = LiveRewardPlot(ALGO, NUM_GNBs, NUM_UEs, EPISODES)
print(f"[train] {ALGO} ({_activation_name}) · {EPISODES} ep x {STEPS} steps/ep ...")
try:
    model.learn(total_timesteps=EPISODES * STEPS, callback=plot_cb)
except KeyboardInterrupt:
    print("[train] Interrupted by user.")
    plt.ioff()

# -- read actual topology confirmed by handshake --------------------------------
actual_gNBs    = env.get_attr("num_gNBs")[0]
actual_UEs     = env.get_attr("num_UEs")[0]
rewards_per_ep = env.get_attr("total_rewards_per_ep")[0]
n_eps          = len(rewards_per_ep)

# -- final static plot ----------------------------------------------------------
_color = _ALGO_COLORS.get(ALGO, "steelblue")

fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
ax.plot(rewards_per_ep, color=_color, linewidth=2)
ax.set_title(
    f"{ALGO}: Avg. Sum Throughput per Episode\n"
    f"{actual_gNBs} gNBs · {actual_UEs} UEs · {n_eps} episodes x {STEPS} steps",
    fontsize=12, fontweight="bold", pad=12,
)
ax.set_xlabel("Episode", fontsize=11)
ax.set_ylabel("Avg. Sum Throughput (Mbps)", fontsize=11)
ax.tick_params(labelsize=10)
ax.grid(True, linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()

plot_path = os.path.join(run_dir, "reward_plot.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"[train] Plot    -> {plot_path}")
plt.show()

# -- save model + vecnorm -------------------------------------------------------
model.save(os.path.join(run_dir, "model.zip"))
env.save(os.path.join(run_dir, "vecnorm.pkl"))
print(f"[train] Model   -> {os.path.join(run_dir, 'model.zip')}")
print(f"[train] VecNorm -> {os.path.join(run_dir, 'vecnorm.pkl')}")
print(f"\n[train] Total time: {(time.time() - start_time) / 60:.1f} min")
print(f"[run]   All outputs in: {run_dir}")
