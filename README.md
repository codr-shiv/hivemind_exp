# 🤖 HiveMind: Single-Agent Reinforcement Learning Environment

A high-fidelity, physically simulated Reinforcement Learning environment built on **Gymnasium** and **PyBullet** for differential drive robots. The system features **Strict Partial Observability ($15 \times 15$ local spatial vision)**, **8-adjacent cell manipulation (orthogonal + diagonal)**, **360° LiDAR perception**, **anti-oscillation action memory**, and **GPU-accelerated PPO training** with parallelized worker multiprocessing.

---

## 🌟 Key Architectural & Performance Highlights

- **Strict Partial Observability**: Removed global navigation vectors (`target_vec`). The robot relies strictly on its local $15 \times 15 \times 5$ spatial vision matrix ($\pm 1.4\text{m}$ radius), task phase flag (`is_carrying`), and 1-step action memory (`last_action`).
- **Anti-Oscillation Memory & Smooth Locomotion**: Incorporates `last_action` into the observation state space and applies action-reversal penalties (`-0.08`) and forward momentum bonuses (`+0.02`), completely eliminating back-and-forth toggling and jitter.
- **Dual-Phase Exploration & Visual Target Homing**:
  - *Phase 1 (Search)*: Unvisited grid cell coverage bonus (`+0.1`) incentivizes systematic spatial exploration under partial observability.
  - *Phase 2 (Visual Attraction)*: When a target (Resource or Depot) enters the local $15 \times 15$ visual matrix ($\le 1.4\text{m}$), potential-based reward shaping `(dist_prev - dist_curr) * 2.0` pulls the robot directly to the target.
- **8-Adjacent Cell Manipulation**: Supports resource pickup and delivery from all **8 neighboring cells** (Top, Bottom, Left, Right, and all 4 Diagonals) at range $\approx 0.32\text{m}$.
- **Hardware Acceleration (GPU + CPU Multiprocessing)**: Built with `SubprocVecEnv` to scale environment physics calculations across multi-core CPU processes (595+ FPS throughput) while accelerating policy backpropagation on NVIDIA GPUs (`device="cuda"`).

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
│   └── ppo_hivemind_final.zip  # Final trained model checkpoint
├── tensorboard_logs/         # TensorBoard diagnostic event logs (PPO_1, PPO_2, PPO_8, etc.)
├── scripts/
│   ├── test_perception.py    # Unit test for LiDAR perception & grid channel integrity
│   ├── test_arm_pickup.py    # Unit test for arm swivel angles and finger dynamics
│   ├── test_robot_mechanics_end_to_end.py # Rigorous A* and PyBullet global transform test
│   └── test_env.py           # Quick environment step & exploration reward verification
├── train_new.py              # Master GPU & multi-core PPO training script
├── test_model.py             # 3D PyBullet GUI evaluation & demo script
└── README.md                 # Complete project documentation
```

---

## 🧠 Perception & Neural Network Architecture

### 1. Dict Observation Space (Pure Partial Observability)
```python
spaces.Dict({
    "grid": spaces.Box(low=0.0, high=1.0, shape=(15, 15, 5), dtype=np.float32),
    "is_carrying": spaces.Discrete(2),
    "last_action": spaces.Discrete(7)
})
```

- **`grid` ($15 \times 15 \times 5$)**: Egocentric spatial matrix around the robot ($\pm 1.4\text{m}$).
  - *Channel 0*: Obstacles & Walls (populated strictly via 180-beam 360° LiDAR raycasting).
  - *Channel 1*: Resource cell location (visible when within $15 \times 15$ window).
  - *Channel 2*: Depot cell location (visible when within $15 \times 15$ window).
  - *Channel 3*: Arena boundary / void cells.
  - *Channel 4*: Robot center (`[7,7]`) and front heading indicator (`[6,7]`).
- **`is_carrying` (Scalar)**: Discrete flag ($0 = \text{Phase 1: Search Resource}$, $1 = \text{Phase 2: Deliver to Depot}$).
- **`last_action` (Scalar)**: Discrete 1-step action memory flag ($0..6$), one-hot encoded into a 7-dimensional vector to eliminate action oscillation.

### 2. Neural Feature Extractor (`CustomCombinedExtractor`)
- **Spatial CNN Branch**: $15 \times 15 \times 5 \rightarrow \text{Conv2D}(32) \rightarrow \text{MaxPool} \rightarrow \text{Conv2D}(64) \rightarrow \text{MaxPool} \rightarrow \text{Flatten} \rightarrow 576\text{ dims}$.
- **Feature Concatenation Layer**: Combines CNN spatial features ($576\text{d}$) + `is_carrying` one-hot ($2\text{d}$) + `last_action` one-hot ($7\text{d}$) $\rightarrow \mathbf{585\text{ dims}}$.
- **Feature Projection**: Linear layer (`Linear(585, 256)` + `ReLU`) feeding into PPO Actor ($\pi$) and Value ($V$) heads (`[128, 64]`).

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
- **Step Time Penalty**: $-0.02$ per action.
- **Cell Coverage Exploration Bonus**: $+0.1$ for visiting previously unvisited grid cells (resets upon pickup).
- **Local Visual Attraction Reward**: $(dist_{\text{prev}} - dist_{\text{curr}}) \times 2.0$ when target is within $1.4\text{m}$ ($15 \times 15$ window).
- **Anti-Oscillation Reversal Penalty**: $-0.08$ on immediate inverse action toggling (`Forward` $\leftrightarrow$ `Backward` or `Left` $\leftrightarrow$ `Right`).
- **Forward Momentum Bonus**: $+0.02$ on consecutive forward moves (`Action 0` after `Action 0`).
- **Pickup Milestone Reward**: $+5.0$ upon picking up resource.
- **Task Completion Dropoff Reward**: $+20.0$ upon placing resource on depot (terminal).
- **Collision Penalty**: $-2.0$ on touching obstacles, boundary walls, depot (before pickup), or resource (before pickup) (terminal).

---

## ⚡ PPO Training Pipeline & Hardware Acceleration

Training is powered by Stable-Baselines3 PPO with multi-process environment vectorization and GPU acceleration:

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
    ent_coef=0.015,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=policy_kwargs,
    tensorboard_log="./tensorboard_logs",
    device=device  # Auto-selects CUDA when GPU is present
)
```

---

## 🚀 Exact Run Instructions

### 1. Environment Setup
Activate the `hivemind` Conda environment:
```bash
conda activate hivemind
```

### 2. Launch Training

**Standard Training (Default: Auto GPU/CPU, 8 CPU workers)**:
```bash
python train_new.py
```

**Workstation High-Performance Training (NVIDIA GPU + 16 CPU Parallel Workers)**:
```bash
python train_new.py --num-envs 16 --device cuda
```

**Custom Timesteps & Reproducible Seed**:
```bash
python train_new.py --timesteps 500000 --num-envs 8 --seed 42
```

### 3. Monitor Real-Time Training Progress (TensorBoard)
Launch TensorBoard to view reward curves, policy entropy, and step throughput:
```bash
tensorboard --logdir=./tensorboard_logs/
```
Open `http://localhost:6006` in your web browser.

### 4. Run 3D PyBullet Visual Demo / Evaluation
Evaluate the trained policy model in 3D PyBullet GUI:
```bash
python test_model.py
```

For headless/direct evaluation:
```bash
python test_model.py --direct --episodes 5
```

---

## 🛠️ Verification Suite

To run all sub-component verification scripts:

```bash
# Test 1: LiDAR perception and 5-channel grid mapping
python scripts/test_perception.py

# Test 2: Arm swivel kinematics, finger clamping, and Lidar height elevation
python scripts/test_arm_pickup.py

# Test 3: Rigorous end-to-end global PyBullet transforms and A* path execution
python scripts/test_robot_mechanics_end_to_end.py

# Test 4: Environment step rewards and exploration bonus
python scripts/test_env.py
```
