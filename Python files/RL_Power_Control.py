"""
RL_Power_Control_v15.py
───────────────────────
Tabular Q-Learning for 5G Tx-Power Control – NetSim v15.

Episode model (matches v14.3):
  • 1 NetSim simulation run  =  1 episode  =  500 RL steps
  • Python reconnects fresh for each episode (retry loop, same as v14.3)
  • For N episodes, run the NetSim simulation N times

Protocol (NetSim is TCP server, port 5555):
  ① Handshake  NetSim → Python : SEG_INT num_gNBs, SEG_INT num_UEs
  ② Per RL step (NetSim sends first):
       NetSim → Python : SEG_DOUBLE_ARRAY sinrs[num_UEs],
                         SEG_DOUBLE_ARRAY tputs[num_UEs],
                         SEG_DOUBLE       reward
       Python → NetSim : SEG_DOUBLE_ARRAY new_powers[num_gNBs]

Run with:
    py -3.13 RL_Power_Control_v15.py
(pyd requires Python 3.13)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── netsim_socket_client.pyd (NetSim v15 Python Interface) ───────────────────
# Add NetSim bin_x64 to DLL search path so pyd's dependencies resolve.
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
    send_and_receive, setup_logging,
    SEG_DOUBLE_ARRAY,
)

# ── output directories ────────────────────────────────────────────────────────
parent_dir = os.getcwd()
path_plots = os.path.join(parent_dir, "plots")
path_logs  = os.path.join(parent_dir, "logs")
os.makedirs(path_plots, exist_ok=True)
os.makedirs(path_logs,  exist_ok=True)

start_time = time.time()

# ── optional pyd logging ──────────────────────────────────────────────────────
setup_logging(
    log_path=os.path.join(path_logs, "netsim_rl.log"),
    to_console=False,
    to_file=True,
)

# ── user inputs ───────────────────────────────────────────────────────────────
num_gNBs     = int(input("Enter number of gNBs     : "))
num_UEs      = int(input("Enter number of UEs      : "))
num_episodes = int(input("Enter number of episodes : "))

steps_per_episode = 500

# ── Q-learning hyperparameters (identical to v14.3) ──────────────────────────
adjustments     = [-3, -1, 0, 1, 3]
percentiles     = [-7, 3]
num_UE_buckets  = len(percentiles) + 1        # 3 buckets per UE
num_gNB_buckets = len(adjustments)            # 5 choices per gNB

num_states  = num_UE_buckets  ** num_UEs
num_actions = num_gNB_buckets ** num_gNBs

q_table = np.zeros((num_states, num_actions))
P_MIN, P_MAX = 27.0, 46.0

exploration   = num_episodes / 2
epsilon       = 1.0
EPSILON_DECAY = 0.25 ** (1.0 / exploration)
MIN_EPSILON   = 0.25
gamma = 0.9
alpha = 0.3

# ── CSV logging ───────────────────────────────────────────────────────────────
gnb_log_path = os.path.join(path_logs, "gnb_powers_log.csv")
all_log_path = os.path.join(path_logs, "all_values_log.csv")

gnb_log_f = open(gnb_log_path, 'w')
all_log_f = open(all_log_path, 'w')
gnb_log_f.write(','.join([f'gNB_Power_{i+1}' for i in range(num_gNBs)]) + '\n')
all_log_f.write(
    ','.join([f'UE{i+1}_Throughput' for i in range(num_UEs)]) +
    ',Sum_Throughput,' +
    ','.join([f'UE{i+1}_SINR' for i in range(num_UEs)]) +
    ',Episode,Iteration\n'
)

def log_gnb_powers(powers):
    gnb_log_f.write(','.join(f'{p:.4f}' for p in powers) + '\n')

def log_all_values(tputs, reward, sinrs, ep, it):
    row = ([f'{v:.4f}' for v in tputs] + [f'{reward:.4f}'] +
           [f'{v:.4f}' for v in sinrs] + [str(ep), str(it)])
    all_log_f.write(','.join(row) + '\n')

# ── helpers ───────────────────────────────────────────────────────────────────
def action_to_deltas(action_index):
    deltas = []
    for _ in range(num_gNBs):
        deltas.append(adjustments[action_index % num_gNB_buckets])
        action_index //= num_gNB_buckets
    return deltas

def get_state(sinr_values):
    state = 0
    for sinr in sinr_values:
        if   sinr < percentiles[0]: bucket = 0
        elif sinr < percentiles[1]: bucket = 1
        else:                        bucket = 2
        state = state * num_UE_buckets + bucket
    return state

# ── main loop: 1 episode = 1 NetSim simulation run ───────────────────────────
total_rewards_per_episode = []

print(f"\n[QL] Starting {num_episodes} episodes  (steps per episode = simulation_time / frame_duration)")
print("[QL] Each episode requires ONE NetSim simulation run.\n")

try:
    for episode in range(num_episodes):

        # ── connect (retry loop identical to v14.3) ───────────────────────────
        print(f"[QL] Episode {episode+1}/{num_episodes}: waiting for NetSim on 127.0.0.1:5555 ...")
        sock = None
        while True:
            try:
                sock = connect("127.0.0.1", 5555)
                print(f"[QL] Connected for episode {episode+1}.")
                break
            except Exception:
                time.sleep(0.5)
                continue

        # ── handshake ─────────────────────────────────────────────────────────
        try:
            hs_msg = send_and_receive(sock, None, expect_reply=True)
            hs_msg.reset_cursor()
            hs_gnbs = hs_msg.get_value()
            hs_ues  = hs_msg.get_value()
            if hs_gnbs is not None and hs_ues is not None:
                num_gNBs = int(hs_gnbs)
                num_UEs  = int(hs_ues)
                # Rebuild spaces if topology changed
                new_states  = num_UE_buckets  ** num_UEs
                new_actions = num_gNB_buckets ** num_gNBs
                if q_table.shape != (new_states, new_actions):
                    q_table     = np.zeros((new_states, new_actions))
                    num_states  = new_states
                    num_actions = new_actions
                print(f"[QL]   num_gNBs={num_gNBs}  num_UEs={num_UEs}")
        except Exception as e:
            print(f"[QL] Handshake failed: {e}")
            try: sock.close()
            except: pass
            continue

        # ── episode state (reset each episode, same as v14.3) ─────────────────
        gNB_powers   = [40.0] * num_gNBs
        total_reward = 0.0
        prev_state   = None
        prev_action  = None
        t            = 0   # actual step counter — driven by NetSim call count

        # ── inner loop: runs until NetSim closes the connection ───────────────
        # Steps = number of RL frames NetSim fires, which equals
        # simulation_time / frame_duration  (e.g. 15 s / 10 ms = 1500 steps).
        # Python never cuts the loop short — the simulation time decides it.
        while True:

            # Step A: receive state from NetSim
            try:
                state_msg = send_and_receive(sock, None, expect_reply=True)
            except (ConnectionResetError, ConnectionAbortedError, OSError) as e:
                print(f"\n[QL] Simulation ended at step {t}: {e}")
                break

            if state_msg is None:
                print(f"\n[QL] NetSim closed connection at step {t} (simulation complete).")
                break

            state_msg.reset_cursor()
            sinr_raw = state_msg.get_value()
            tput_raw = state_msg.get_value()
            reward   = state_msg.get_value()

            if sinr_raw is None:
                # Malformed — send back current powers and skip
                try:
                    act_msg = Message()
                    add_variable(act_msg, SEG_DOUBLE_ARRAY, gNB_powers, len(gNB_powers))
                    send_and_receive(sock, act_msg, expect_reply=False)
                except OSError:
                    break
                continue

            sinrs  = list(sinr_raw)
            tputs  = list(tput_raw) if tput_raw is not None else [0.0] * num_UEs
            reward = float(reward)  if reward   is not None else 0.0
            state  = get_state(sinrs)

            # Step B: Q-update (reward belongs to previous action)
            if prev_state is not None:
                q_table[prev_state, prev_action] += alpha * (
                    reward
                    + gamma * np.max(q_table[state])
                    - q_table[prev_state, prev_action]
                )
                total_reward += reward

            # Step C: ε-greedy action
            if np.random.rand() < epsilon:
                action_index = np.random.randint(num_actions)
            else:
                action_index = int(np.argmax(q_table[state]))

            deltas = action_to_deltas(action_index)
            gNB_powers = [
                max(P_MIN, min(P_MAX, p + dp))
                for p, dp in zip(gNB_powers, deltas)
            ]

            # Step D: send new powers
            act_msg = Message()
            add_variable(act_msg, SEG_DOUBLE_ARRAY, gNB_powers, len(gNB_powers))
            try:
                send_and_receive(sock, act_msg, expect_reply=False)
            except (ConnectionResetError, ConnectionAbortedError, OSError) as e:
                print(f"\n[QL] Send failed at step {t}: {e}")
                break

            t += 1   # count only fully completed steps

            # Step E: logging (last episode only)
            if episode == num_episodes - 1:
                log_gnb_powers(gNB_powers)
                log_all_values(tputs, reward, sinrs, episode, t)

            prev_state  = state
            prev_action = action_index

            print(f"Episode {episode:4d}  Step {t:4d}  reward={reward:8.4f}  ε={epsilon:.4f}")

        # ── end of episode ────────────────────────────────────────────────────
        actual_steps = t if t > 0 else 1   # guard against zero-step episodes
        avg_r = total_reward / actual_steps
        total_rewards_per_episode.append(avg_r)

        if epsilon > MIN_EPSILON:
            epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        print("=" * 79)
        print(f"Episode {episode+1}/{num_episodes} completed | "
              f"Steps: {actual_steps} | Avg Reward: {avg_r:.4f} | ε: {epsilon:.4f}")
        print("=" * 79)

        try:
            sock.close()
        except Exception:
            pass

        if episode < num_episodes - 1:
            print(f"\n[QL] Start the next NetSim simulation run for episode {episode+2}.")

except KeyboardInterrupt:
    print("\n[QL] Interrupted by user.")

finally:
    try: gnb_log_f.close()
    except: pass
    try: all_log_f.close()
    except: pass

# ── reward plot ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
ax.plot(total_rewards_per_episode, color='steelblue', linewidth=2)
ax.set_title(
    f"Q-Learning: Sum Throughput per Episode\n"
    f"{num_gNBs} gNBs · {num_UEs} UEs · {num_episodes} episodes",
    fontsize=12, fontweight='bold', pad=12
)
ax.set_xlabel("Episode", fontsize=11)
ax.set_ylabel("Avg. Sum Throughput (Mbps)", fontsize=11)
ax.tick_params(labelsize=10)
ax.grid(True, linestyle='--', alpha=0.4)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(path_plots, "Sum_Throughputs.png"), dpi=150, bbox_inches='tight')
plt.show()
print("[QL] Reward plot → plots/Sum_Throughputs.png")

# ── gNB powers plot (last episode) ───────────────────────────────────────────
gnb_df = pd.read_csv(gnb_log_path)
if len(gnb_df) > 1:
    gnb_df.drop(gnb_df.tail(1).index, inplace=True)

# Adaptive height: cap each subplot at 3 in, total at 24 in (fits any gNB count)
h_each = max(2.0, min(3.5, 24.0 / max(num_gNBs, 1)))
fig, axes = plt.subplots(
    num_gNBs, 1,
    figsize=(11, h_each * num_gNBs),
    sharex=True,
    constrained_layout=True,
    dpi=150,
)
if num_gNBs == 1:
    axes = [axes]

# Use matplotlib's built-in color cycle (tab10) — works for any number of gNBs
for i, ax in enumerate(axes):
    col = f'gNB_Power_{i+1}'
    if col not in gnb_df.columns:
        continue
    ax.plot(gnb_df[col], label=f'gNB {i+1}', color=f'C{i % 10}', linewidth=1.0)
    ax.set_ylabel("Power\n(dBm)", fontsize=8)
    ax.set_ylim(P_MIN - 2, P_MAX + 2)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.6)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)
    if i == 0:
        ax.set_title("gNB Powers vs. Time – Optimal Policy (Last Episode)")

axes[-1].set_xlabel("Iterations")
fig.savefig(os.path.join(path_plots, "gNB_Powers.png"), dpi=150, bbox_inches='tight')
plt.show()
print("[QL] gNB powers plot → plots/gNB_Powers.png")

# ── save Q-table + optimal policy ────────────────────────────────────────────
pd.DataFrame(q_table).to_csv("Q_Table.csv")
policy = [float(np.max(q_table[i])) for i in range(num_states)]
pd.DataFrame(policy, columns=["max_Q"]).to_csv("optimal_policy.csv")
print("[QL] Q_Table.csv and optimal_policy.csv saved.")

print(f"\n[QL] Done – {(time.time() - start_time) / 60:.1f} minutes total")
