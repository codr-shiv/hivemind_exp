# 🤖 HiveMind: Single-Agent Reinforcement Learning Environment

A high-fidelity, physically simulated Reinforcement Learning environment built on **Gymnasium** and **PyBullet** for differential drive robots. The system features **8-adjacent cell manipulation (orthogonal + diagonal)**, **360° LiDAR perception**, **dual-scale egocentric perception**, and **on-policy PPO training** with complete metric convergence (+10.98 mean episode return, ~40-step completion time).

---

## 🌟 Key Highlights & Achieved Performance

- **100% Task Convergence**: Evaluated over 500,000 timesteps, the PPO agent achieves an average episode return of **+10.98** (out of a theoretical maximum of $+11.00$).
- **High-Speed Execution**: Solves the complete 2-phase task (Obstacle Navigation $\rightarrow$ Resource Pickup $\rightarrow$ Depot Delivery) in an average of **~41 steps** (~250 FPS during training).
- **8-Adjacent Cell Manipulation**: Supports resource pickup and delivery from all **8 neighboring cells** (Top, Bottom, Left, Right, and all 4 Diagonals) at range $\approx 0.32\text{m}$.
- **Dual-Scale Observation Space**: Combines a local $15 \times 15 \times 5$ egocentric grid (CNN for local obstacle dodging) with an egocentric target compass vector `target_vec` (MLP for global compass guidance across the $4\text{m} \times 4\text{m}$ world).
- **Dynamic Physics & Substep Interpolation**: Differential drive kinematics rendered over **30 physics substeps per action**, eliminating visual teleportation and providing continuous wheel movement.

---

## 📂 Project Structure

```text
HiveMind/
├── hivemind_env/
│   ├── __init__.py           # Environment registration ('HiveMind-SingleAgent')
│   ├── env.py                # Core Gymnasium environment class (PyBullet simulation engine)
│   ├── models.py             # Custom PyTorch feature extractor (CustomCombinedExtractor)
│   └── assets/
│       └── diff_drive_bot.urdf # URDF definition of the robot, arm, gripper, and wheels
├── models/
│   ├── ppo_hivemind_latest.zip # Checkpoint model updated every 5,000 steps
│   └── ppo_hivemind_final.zip  # Final converged model (500k timesteps)
├── tensorboard_logs/         # TensorBoard diagnostic event logs (PPO_1, PPO_2, PPO_4)
├── scripts/
│   ├── test_arm_pickup.py    # Unit test for arm swivel angles and finger dynamics
│   └── test_robot_mechanics_end_to_end.py # End-to-end simulation verification
├── train_new.py              # Master PPO training pipeline with VecMonitor & LR decay
├── test_model.py             # 3D PyBullet GUI evaluation & demo script
└── README.md                 # Complete project documentation
```

---

## 📐 Environment & Physics Design

### 1. Map & Arena Geometry
- **Grid Resolution**: $20 \times 20$ grid cells.
- **Cell Dimensions**: $0.2\text{m} \times 0.2\text{m}$ per cell ($4.0\text{m} \times 4.0\text{m}$ total arena size from $-2.0\text{m}$ to $+2.0\text{m}$).
- **Randomized Obstacles**: 6 to 12 procedural obstacles spawned per episode.
- **Solvability Guarantee**: Every randomized map is pre-validated using an **8-connected Breadth-First Search (BFS)** path finder to guarantee that an unobstructed path exists from Robot $\rightarrow$ Resource $\rightarrow$ Depot before spawning.

### 2. Robotic Arm & Manipulator Kinematics
- **Arm Reach Threshold**: $1.6 \times \text{cell\_size} = 0.32\text{m}$, allowing interaction with diagonal cells ($\sqrt{0.2^2 + 0.2^2} \approx 0.2828\text{m}$).
- **Dynamic Swivel Angle**: The arm dynamically swivels to target center:
  $$\theta_{\text{arm\_yaw}} = \text{atan2}(\Delta y, \Delta x) - \theta_{\text{robot\_yaw}}$$
- **LiDAR Height Adjustment**: When carrying a resource, the 180-beam LiDAR ray height automatically elevates from $z=0.05\text{m} \rightarrow z=0.15\text{m}$, preventing the carried resource block ($z=0.10\text{m}$) from self-occluding the LiDAR scan.

---

## 🧠 Perception & Neural Network Architecture

### 1. Dict Observation Space
```python
spaces.Dict({
    "grid": spaces.Box(low=0.0, high=1.0, shape=(15, 15, 5), dtype=np.float32),
    "is_carrying": spaces.Discrete(2),
    "target_vec": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
})
```

- **`grid` ($15 \times 15 \times 5$)**: Egocentric spatial matrix around the robot ($\pm 1.4\text{m}$).
  - *Channel 0*: Obstacles & Walls (populated strictly via 180-beam 360° LiDAR raycasting).
  - *Channel 1*: Resource cell location.
  - *Channel 2*: Depot cell location.
  - *Channel 3*: Arena boundary / void cells.
  - *Channel 4*: Robot center (`[7,7]`) and front heading indicator (`[6,7]`).
- **`is_carrying` (Scalar)**: Discrete flag ($0 = \text{Phase 1: Search Resource}$, $1 = \text{Phase 2: Deliver to Depot}$).
- **`target_vec` (2D Vector)**: Normalized egocentric vector $[\Delta x_{\text{ego}}/4.0, \Delta y_{\text{ego}}/4.0]$ pointing directly to current active goal, guaranteeing 100% target visibility anywhere in the world.

### 2. Neural Feature Extractor (`CustomCombinedExtractor`)
- **Spatial CNN Branch**: $15 \times 15 \times 5 \rightarrow \text{Conv2D}(32) \rightarrow \text{MaxPool} \rightarrow \text{Conv2D}(64) \rightarrow \text{MaxPool} \rightarrow \text{Flatten} \rightarrow 576\text{ dims}$.
- **Concatenation Layer**: Combines CNN spatial features ($576\text{d}$) + `is_carrying` flag ($2\text{d}$) + `target_vec` compass ($2\text{d}$) $\rightarrow 580\text{ dims}$.
- **Feature Projection**: Linear projection to $256\text{ dimensions}$ feeding into PPO Actor ($\pi$) and Value ($V$) heads (`[128, 64]`).

---

## 🎯 Action Space & Reward System

### Action Space (`Discrete(7)`)
| Action ID | Action Name | Description |
| :---: | :--- | :--- |
| `0` | **Move Forward** | Moves robot 1 grid cell forward ($0.2\text{m}$) over 30 substeps |
| `1` | **Move Backward** | Moves robot 1 grid cell backward ($0.2\text{m}$) over 30 substeps |
| `2` | **Turn Left 90°** | Rotates robot $+90^\circ$ counter-clockwise over 30 substeps |
| `3` | **Turn Right 90°** | Rotates robot $-90^\circ$ clockwise over 30 substeps |
| `4` | **Pick Up** | Swivels arm to adjacent resource, closes fingers, elevates LiDAR, pulls resource |
| `5` | **Drop Off** | Swivels arm to adjacent depot, places resource, opens fingers, resets LiDAR |
| `6` | **Stay** | Holds current position for 30 substeps |

### Reward Structure
- **Step Penalty**: $-0.05$ per action.
- **Potential-Based Reward Shaping (PBRS)**: $1.0 \times (\text{dist}_{\text{prev}} - \text{dist}_{\text{curr}})$, providing dense progress guidance toward active target.
- **Pickup Milestone Reward**: $+2.0$ upon picking up resource.
- **Task Completion Dropoff Reward**: $+10.0$ upon placing resource on depot (terminal).
- **Collision Penalty**: $-2.0$ on touching obstacles, walls, depot (before pickup), or resource (before pickup) (terminal).

---

## ⚡ PPO Training Setup & Diagnostics

The agent is trained using **Proximal Policy Optimization (PPO)** from Stable-Baselines3 across 4 parallel environments:

```python
model = PPO(
    policy="MultiInputPolicy",
    env=vec_env,
    learning_rate=dynamic_linear_lr_schedule(initial_lr=3e-4, min_lr=3e-5),
    n_steps=2048,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=policy_kwargs,
    tensorboard_log="./tensorboard_logs"
)
```

### Final Converged Metrics (Step 500,000)
- **`rollout/ep_rew_mean`**: **+10.98** (Max theoretical reward $\approx +11.00$).
- **`rollout/ep_len_mean`**: **41.2 steps** (Average mission duration).
- **`train/approx_kl`**: **0.018** (Stable policy update step size).
- **`train/explained_variance`**: **0.55+** (Strong Critic return estimation).

---

## 🚀 How to Run

### 1. Prerequisites & Environment Setup
Ensure you have the `hivemind` Conda environment activated:
```bash
conda activate hivemind
```

### 2. Train the PPO Model
To start training from scratch with live progress logging and periodic checkpoint saving:
```bash
python train_new.py
```

### 3. Monitor in TensorBoard
To launch TensorBoard and visualize learning curves in real time:
```bash
tensorboard --logdir=./tensorboard_logs/
```
Open `http://localhost:6006` in your web browser.

### 4. Run the 3D PyBullet GUI Visual Demo
To visualize the trained policy performing 8-way adjacent pickup and delivery in 3D:
```bash
python test_model.py
```

---

## 🛠️ Verification Scripts

To test specific sub-components independently:
- **Arm Swivel & Kinematics Test**:
  ```bash
  python scripts/test_arm_pickup.py
  ```
- **End-to-End Environment Mechanics Test**:
  ```bash
  python scripts/test_robot_mechanics_end_to_end.py
  ```
