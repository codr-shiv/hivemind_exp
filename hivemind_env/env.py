import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as pb
import pybullet_data
import random
from collections import deque
import math
import time

def _bfs_path_to_adjacent_exists(grid, start, target_cell):
    """Checks if a valid path exists from start to any 8-adjacent cell of target_cell, treating target_cell as impassable."""
    rows, cols = grid.shape
    temp_grid = grid.copy()
    temp_grid[target_cell] = 1  # Target cell itself is impassable
    
    adj_targets = set()
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        ar, ac = target_cell[0] + dr, target_cell[1] + dc
        if 0 <= ar < rows and 0 <= ac < cols and temp_grid[ar, ac] == 0:
            adj_targets.add((ar, ac))
            
    if not adj_targets or temp_grid[start] == 1:
        return False
        
    if start in adj_targets:
        return True
        
    queue = deque([start])
    visited = set([start])
    while queue:
        r, c = queue.popleft()
        if (r, c) in adj_targets:
            return True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and temp_grid[nr, nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc))
    return False

class HiveMindSingleAgentEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "DIRECT"], "render_fps": 10}

    def __init__(self, render_mode=None, **kwargs):
        super().__init__()
        self.render_mode = render_mode
        self.dt = 1.0 / 240.0
        self.is_carrying = False
        self.grid_size = 20
        self.cell_size = 0.2
        
        # Actions: 0: Forward, 1: Backward, 2: Turn Left, 3: Turn Right, 4: Pick Up, 5: Drop Off, 6: Stay
        self.action_space = spaces.Discrete(7)
        
        # Observation: Local grid around agent 15x15x5 
        # Channels: 0: obstacles, 1: resources, 2: depot, 3: boundaries, 4: agent's heading
        self.obs_size = 15
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=1, shape=(self.obs_size, self.obs_size, 5), dtype=np.float32),
            "is_carrying": spaces.Discrete(2)
        })

        if self.render_mode == "human":
            self.client_id = pb.connect(pb.GUI)
        else:
            self.client_id = pb.connect(pb.DIRECT)
            
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        
        self.robot_id = None
        self.resource_id = None
        self.depot_id = None
        self.obstacle_ids = []
        self.depot_pos_grid = (0, 0)
        self.max_steps = 500
        self.current_step = 0

    def _grid_to_world(self, r, c):
        x = (c - self.grid_size/2.0) * self.cell_size + (self.cell_size/2.0)
        y = (self.grid_size/2.0 - r) * self.cell_size - (self.cell_size/2.0)
        return x, y
        
    def _world_to_grid(self, x, y):
        c = int((x / self.cell_size) + self.grid_size/2.0)
        r = int(self.grid_size/2.0 - (y / self.cell_size))
        return r, c

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.is_carrying = False
        self.current_step = 0
        
        pb.resetSimulation(physicsClientId=self.client_id)
        pb.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        pb.loadURDF("plane.urdf", physicsClientId=self.client_id)

        r_pos, res_pos, dep_pos, obstacles = self._generate_valid_map()
        self.depot_pos_grid = dep_pos
        self.visited_cells = set([r_pos])

        rx, ry = self._grid_to_world(r_pos[0], r_pos[1])
        resx, resy = self._grid_to_world(res_pos[0], res_pos[1])
        dx, dy = self._grid_to_world(dep_pos[0], dep_pos[1])

        # Robot URDF loading with backward-compatible fallback
        import os
        urdf_path = os.path.join(os.path.dirname(__file__), "assets", "diff_drive_bot.urdf")
        if os.path.exists(urdf_path):
            self.robot_id = pb.loadURDF(urdf_path, basePosition=[rx, ry, 0.005], physicsClientId=self.client_id)
            self.has_urdf_wheels = True
            self.left_wheel_indices = []
            self.right_wheel_indices = []
            self.arm_yaw_joint_idx = None
            self.left_finger_joint_idx = None
            self.right_finger_joint_idx = None
            self.lidar_joint_idx = None
            self.current_arm_yaw = 0.0
            self.current_finger_pos = 0.015  # open claw
            self.current_lidar_height = 0.0  # lower Lidar height (z=0.08 above base)

            for i in range(pb.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
                info = pb.getJointInfo(self.robot_id, i, physicsClientId=self.client_id)
                jname = info[1].decode("utf-8")
                if "left_wheel" in jname:
                    self.left_wheel_indices.append(i)
                elif "right_wheel" in jname:
                    self.right_wheel_indices.append(i)
                elif jname == "arm_yaw_joint":
                    self.arm_yaw_joint_idx = i
                elif jname == "left_finger_joint":
                    self.left_finger_joint_idx = i
                elif jname == "right_finger_joint":
                    self.right_finger_joint_idx = i
                elif jname == "lidar_joint":
                    self.lidar_joint_idx = i

            self._set_arm_and_lidar_joints(self.current_arm_yaw, self.current_finger_pos, self.current_lidar_height)
        else:
            self.has_urdf_wheels = False
            self.left_wheel_indices = []
            self.right_wheel_indices = []
            self.arm_yaw_joint_idx = None
            self.left_finger_joint_idx = None
            self.right_finger_joint_idx = None
            self.lidar_joint_idx = None
            self.current_arm_yaw = 0.0
            self.current_finger_pos = 0.015
            self.current_lidar_height = 0.0
            robot_col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[self.cell_size*0.4, self.cell_size*0.4, 0.05], physicsClientId=self.client_id)
            robot_vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[self.cell_size*0.4, self.cell_size*0.4, 0.05], rgbaColor=[0, 0, 1, 1], physicsClientId=self.client_id)
            self.robot_id = pb.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=robot_col, baseVisualShapeIndex=robot_vis, basePosition=[rx, ry, 0.05], physicsClientId=self.client_id)

        # Resource
        res_col = pb.createCollisionShape(pb.GEOM_CYLINDER, radius=self.cell_size*0.3, height=0.1, physicsClientId=self.client_id)
        res_vis = pb.createVisualShape(pb.GEOM_CYLINDER, radius=self.cell_size*0.3, length=0.1, rgbaColor=[0, 1, 0, 1], physicsClientId=self.client_id)
        self.resource_id = pb.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=res_col, baseVisualShapeIndex=res_vis, basePosition=[resx, resy, 0.05], physicsClientId=self.client_id)

        # Depot (Collision & Visual)
        depot_col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[self.cell_size*0.4, self.cell_size*0.4, 0.05], physicsClientId=self.client_id)
        depot_vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[self.cell_size*0.5, self.cell_size*0.5, 0.01], rgbaColor=[1, 0, 0, 0.5], physicsClientId=self.client_id)
        self.depot_id = pb.createMultiBody(baseMass=0, baseCollisionShapeIndex=depot_col, baseVisualShapeIndex=depot_vis, basePosition=[dx, dy, 0.05], physicsClientId=self.client_id)

        # Obstacles
        self.obstacle_ids = []
        for (obs_r, obs_c, size_r, size_c) in obstacles:
            cx, cy = self._grid_to_world(obs_r + size_r/2.0 - 0.5, obs_c + size_c/2.0 - 0.5)
            hx, hy = (size_c * self.cell_size) / 2.0, (size_r * self.cell_size) / 2.0
            
            obs_col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[hx, hy, 0.1], physicsClientId=self.client_id)
            obs_vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[hx, hy, 0.1], rgbaColor=[0.5, 0.5, 0.5, 1], physicsClientId=self.client_id)
            obs_id = pb.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=obs_col, baseVisualShapeIndex=obs_vis, basePosition=[cx, cy, 0.1], physicsClientId=self.client_id)
            self.obstacle_ids.append(obs_id)

        # Static Boundary Walls around the 20x20 grid (4.0m x 4.0m)
        self.wall_ids = []
        wall_half_extents = [(2.2, 0.1, 0.2), (2.2, 0.1, 0.2), (0.1, 2.0, 0.2), (0.1, 2.0, 0.2)]
        wall_positions = [(0, 2.1, 0.1), (0, -2.1, 0.1), (-2.1, 0, 0.1), (2.1, 0, 0.1)]
        for he, pos in zip(wall_half_extents, wall_positions):
            w_col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=he, physicsClientId=self.client_id)
            w_vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=he, rgbaColor=[0.2, 0.2, 0.2, 1], physicsClientId=self.client_id)
            w_id = pb.createMultiBody(baseMass=0, baseCollisionShapeIndex=w_col, baseVisualShapeIndex=w_vis, basePosition=pos, physicsClientId=self.client_id)
            self.wall_ids.append(w_id)
            
        # Draw grid lines for visualization
        if self.render_mode == "human":
            pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0, physicsClientId=self.client_id)
            for i in range(self.grid_size + 1):
                x = (i - self.grid_size/2.0) * self.cell_size
                pb.addUserDebugLine([x, -self.grid_size/2.0 * self.cell_size, 0.01], 
                                    [x, self.grid_size/2.0 * self.cell_size, 0.01], 
                                    [0,0,0], physicsClientId=self.client_id)
                y = (i - self.grid_size/2.0) * self.cell_size
                pb.addUserDebugLine([-self.grid_size/2.0 * self.cell_size, y, 0.01], 
                                    [self.grid_size/2.0 * self.cell_size, y, 0.01], 
                                    [0,0,0], physicsClientId=self.client_id)

        for _ in range(10):
            pb.stepSimulation(physicsClientId=self.client_id)

        return self._get_obs(), self._get_info()

    def _set_arm_and_lidar_joints(self, arm_yaw, finger_pos, lidar_height):
        """Resets joint positions for the robotic arm yaw, claw fingers, and Lidar mount."""
        if getattr(self, "robot_id", None) is None:
            return
        if getattr(self, "arm_yaw_joint_idx", None) is not None:
            pb.resetJointState(self.robot_id, self.arm_yaw_joint_idx, arm_yaw, physicsClientId=self.client_id)
        if getattr(self, "left_finger_joint_idx", None) is not None:
            pb.resetJointState(self.robot_id, self.left_finger_joint_idx, finger_pos, physicsClientId=self.client_id)
        if getattr(self, "right_finger_joint_idx", None) is not None:
            pb.resetJointState(self.robot_id, self.right_finger_joint_idx, -finger_pos, physicsClientId=self.client_id)
        if getattr(self, "lidar_joint_idx", None) is not None:
            pb.resetJointState(self.robot_id, self.lidar_joint_idx, lidar_height, physicsClientId=self.client_id)

    def _get_cardinal_direction_angle(self, target_world_pos, robot_world_pos, robot_yaw):
        """Calculates relative angle to target location relative to robot heading (supports orthogonal & diagonal directions)."""
        dx = target_world_pos[0] - robot_world_pos[0]
        dy = target_world_pos[1] - robot_world_pos[1]
        target_angle = math.atan2(dy, dx)
        rel_angle = target_angle - robot_yaw
        return math.atan2(math.sin(rel_angle), math.cos(rel_angle))

    def _generate_valid_map(self):
        while True:
            grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
            
            # Robot, Resource, and Depot placed randomly in valid non-boundary grid cells
            r_pos = (int(self.np_random.integers(1, 19)), int(self.np_random.integers(1, 19)))
            res_pos = (int(self.np_random.integers(1, 19)), int(self.np_random.integers(1, 19)))
            dep_pos = (int(self.np_random.integers(1, 19)), int(self.np_random.integers(1, 19)))
            
            while res_pos == r_pos:
                res_pos = (int(self.np_random.integers(1, 19)), int(self.np_random.integers(1, 19)))
            while dep_pos == r_pos or dep_pos == res_pos:
                dep_pos = (int(self.np_random.integers(1, 19)), int(self.np_random.integers(1, 19)))
                
            num_obstacles_target = int(self.np_random.integers(6, 13))
            obstacles = []
            
            # Candidate 1x1 obstacle cells: strictly rows 2..17, cols 2..17
            # This guarantees obstacles are never spawned adjacent to boundary walls (rows 0,1,18,19 or cols 0,1,18,19)
            candidate_cells = [(r, c) for r in range(2, 18) for c in range(2, 18)]
            self.np_random.shuffle(candidate_cells)
            
            for obs_r, obs_c in candidate_cells:
                if len(obstacles) >= num_obstacles_target:
                    break
                    
                # Do not overlap with robot, resource, or depot
                if (obs_r, obs_c) in [r_pos, res_pos, dep_pos]:
                    continue
                    
                # Ensure obstacles are not spawned consecutively (no existing obstacle in 8-neighborhood)
                consecutive = False
                for existing_r, existing_c, _, _ in obstacles:
                    if abs(existing_r - obs_r) <= 1 and abs(existing_c - obs_c) <= 1:
                        consecutive = True
                        break
                        
                if not consecutive:
                    grid[obs_r, obs_c] = 1
                    obstacles.append((obs_r, obs_c, 1, 1))
                    
            if _bfs_path_to_adjacent_exists(grid, r_pos, res_pos) and _bfs_path_to_adjacent_exists(grid, res_pos, dep_pos):
                return r_pos, res_pos, dep_pos, obstacles

    def _step_substep(self, left_wheel_delta=0.0, right_wheel_delta=0.0):
        if getattr(self, "has_urdf_wheels", False):
            for idx in getattr(self, "left_wheel_indices", []):
                pos = pb.getJointState(self.robot_id, idx, physicsClientId=self.client_id)[0]
                pb.resetJointState(self.robot_id, idx, pos + left_wheel_delta, physicsClientId=self.client_id)
            for idx in getattr(self, "right_wheel_indices", []):
                pos = pb.getJointState(self.robot_id, idx, physicsClientId=self.client_id)[0]
                pb.resetJointState(self.robot_id, idx, pos + right_wheel_delta, physicsClientId=self.client_id)

        self._set_arm_and_lidar_joints(
            getattr(self, "current_arm_yaw", 0.0),
            getattr(self, "current_finger_pos", 0.015),
            getattr(self, "current_lidar_height", 0.0)
        )

        pb.stepSimulation(physicsClientId=self.client_id)

        if self.render_mode == "human":
            time.sleep(0.015)  # 60 FPS smooth GUI visual delay per physics sub-step

    def step(self, action):
        self.current_step += 1
        reward = -0.02
        terminated = False
        truncated = self.current_step >= self.max_steps

        prev_robot_pos, robot_orn = pb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        rx, ry, rz = prev_robot_pos
        yaw = pb.getEulerFromQuaternion(robot_orn)[2]
        prev_carrying = self.is_carrying

        res_pos_world, _ = pb.getBasePositionAndOrientation(self.resource_id, physicsClientId=self.client_id)
        dep_pos_world, _ = pb.getBasePositionAndOrientation(self.depot_id, physicsClientId=self.client_id)
        target_pos_world = dep_pos_world[:2] if self.is_carrying else res_pos_world[:2]

        dist_prev = np.linalg.norm(np.array(prev_robot_pos[:2]) - np.array(target_pos_world))

        num_substeps = 30
        wheel_radius = 0.033
        wheel_step = (self.cell_size / (2 * math.pi * wheel_radius)) / num_substeps

        if action == 0:  # Move Forward 1 cell
            target_x = rx + self.cell_size * math.cos(yaw)
            target_y = ry + self.cell_size * math.sin(yaw)
            for i in range(1, num_substeps + 1):
                alpha = i / float(num_substeps)
                cur_x = rx + alpha * (target_x - rx)
                cur_y = ry + alpha * (target_y - ry)
                pb.resetBasePositionAndOrientation(self.robot_id, [cur_x, cur_y, rz], robot_orn, physicsClientId=self.client_id)
                self._step_substep(left_wheel_delta=wheel_step, right_wheel_delta=wheel_step)

                if self.is_carrying:
                    arm_world_angle = yaw + self.current_arm_yaw
                    res_x = cur_x + 0.15 * math.cos(arm_world_angle)
                    res_y = cur_y + 0.15 * math.sin(arm_world_angle)
                    pb.resetBasePositionAndOrientation(self.resource_id, [res_x, res_y, 0.05], robot_orn, physicsClientId=self.client_id)

        elif action == 1:  # Move Backward 1 cell
            target_x = rx - self.cell_size * math.cos(yaw)
            target_y = ry - self.cell_size * math.sin(yaw)
            for i in range(1, num_substeps + 1):
                alpha = i / float(num_substeps)
                cur_x = rx + alpha * (target_x - rx)
                cur_y = ry + alpha * (target_y - ry)
                pb.resetBasePositionAndOrientation(self.robot_id, [cur_x, cur_y, rz], robot_orn, physicsClientId=self.client_id)
                self._step_substep(left_wheel_delta=-wheel_step, right_wheel_delta=-wheel_step)

                if self.is_carrying:
                    arm_world_angle = yaw + self.current_arm_yaw
                    res_x = cur_x + 0.15 * math.cos(arm_world_angle)
                    res_y = cur_y + 0.15 * math.sin(arm_world_angle)
                    pb.resetBasePositionAndOrientation(self.resource_id, [res_x, res_y, 0.05], robot_orn, physicsClientId=self.client_id)

        elif action == 2:  # Turn Left 90 degrees
            target_yaw = yaw + (math.pi / 2.0)
            for i in range(1, num_substeps + 1):
                alpha = i / float(num_substeps)
                cur_yaw = yaw + alpha * (target_yaw - yaw)
                iorn = pb.getQuaternionFromEuler([0, 0, cur_yaw])
                pb.resetBasePositionAndOrientation(self.robot_id, [rx, ry, rz], iorn, physicsClientId=self.client_id)
                self._step_substep(left_wheel_delta=-wheel_step, right_wheel_delta=wheel_step)

                if self.is_carrying:
                    arm_world_angle = cur_yaw + self.current_arm_yaw
                    res_x = rx + 0.15 * math.cos(arm_world_angle)
                    res_y = ry + 0.15 * math.sin(arm_world_angle)
                    pb.resetBasePositionAndOrientation(self.resource_id, [res_x, res_y, 0.05], iorn, physicsClientId=self.client_id)

        elif action == 3:  # Turn Right 90 degrees
            target_yaw = yaw - (math.pi / 2.0)
            for i in range(1, num_substeps + 1):
                alpha = i / float(num_substeps)
                cur_yaw = yaw + alpha * (target_yaw - yaw)
                iorn = pb.getQuaternionFromEuler([0, 0, cur_yaw])
                pb.resetBasePositionAndOrientation(self.robot_id, [rx, ry, rz], iorn, physicsClientId=self.client_id)
                self._step_substep(left_wheel_delta=wheel_step, right_wheel_delta=-wheel_step)

                if self.is_carrying:
                    arm_world_angle = cur_yaw + self.current_arm_yaw
                    res_x = rx + 0.15 * math.cos(arm_world_angle)
                    res_y = ry + 0.15 * math.sin(arm_world_angle)
                    pb.resetBasePositionAndOrientation(self.resource_id, [res_x, res_y, 0.05], iorn, physicsClientId=self.client_id)

        elif action == 4 and not self.is_carrying:  # Animated Pickup from adjacent cell (orthogonal & diagonal)
            robot_pos, _ = pb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
            res_pos, _ = pb.getBasePositionAndOrientation(self.resource_id, physicsClientId=self.client_id)
            dist = np.linalg.norm(np.array(robot_pos[:2]) - np.array(res_pos[:2]))
            if 0.05 <= dist <= self.cell_size * 1.6:  # 8-adjacent cell pickup range (~0.32m)
                target_arm_yaw = self._get_cardinal_direction_angle(res_pos, robot_pos, yaw)
                start_finger = self.current_finger_pos
                start_lidar = self.current_lidar_height

                half_substeps = num_substeps // 2

                # Phase 1: Swivel arm to target direction, close fingers, raise lidar, pull object to arm tip
                for i in range(1, half_substeps + 1):
                    alpha = i / float(half_substeps)
                    self.current_arm_yaw = alpha * target_arm_yaw
                    self.current_finger_pos = start_finger + alpha * (-0.01 - start_finger)
                    self.current_lidar_height = start_lidar + alpha * (0.07 - start_lidar)

                    arm_world_angle = yaw + self.current_arm_yaw
                    target_res_x = rx + 0.15 * math.cos(arm_world_angle)
                    target_res_y = ry + 0.15 * math.sin(arm_world_angle)
                    cur_res_x = res_pos[0] + alpha * (target_res_x - res_pos[0])
                    cur_res_y = res_pos[1] + alpha * (target_res_y - res_pos[1])
                    pb.resetBasePositionAndOrientation(self.resource_id, [cur_res_x, cur_res_y, 0.05], robot_orn, physicsClientId=self.client_id)

                    self._step_substep(left_wheel_delta=0.0, right_wheel_delta=0.0)

                # Phase 2: Swivel arm BACK to forward position (0.0) while carrying the resource
                start_res_x = rx + 0.15 * math.cos(yaw + target_arm_yaw)
                start_res_y = ry + 0.15 * math.sin(yaw + target_arm_yaw)
                final_res_x = rx + 0.15 * math.cos(yaw)
                final_res_y = ry + 0.15 * math.sin(yaw)

                for i in range(1, half_substeps + 1):
                    alpha = i / float(half_substeps)
                    self.current_arm_yaw = target_arm_yaw + alpha * (0.0 - target_arm_yaw)
                    self.current_finger_pos = -0.01
                    self.current_lidar_height = 0.07

                    cur_res_x = start_res_x + alpha * (final_res_x - start_res_x)
                    cur_res_y = start_res_y + alpha * (final_res_y - start_res_y)
                    pb.resetBasePositionAndOrientation(self.resource_id, [cur_res_x, cur_res_y, 0.05], robot_orn, physicsClientId=self.client_id)

                    self._step_substep(left_wheel_delta=0.0, right_wheel_delta=0.0)

                self.is_carrying = True
                self.current_arm_yaw = 0.0  # Reset arm link back to forward position
                self.current_finger_pos = -0.01
                self.current_lidar_height = 0.07
                curr_p, _ = pb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
                self.visited_cells = set([self._world_to_grid(curr_p[0], curr_p[1])])
                reward += 5.0
            else:
                for _ in range(num_substeps):
                    self._step_substep(left_wheel_delta=0.0, right_wheel_delta=0.0)

        elif action == 5 and self.is_carrying:  # Animated Drop Off to depot from adjacent cell (orthogonal & diagonal)
            robot_pos, _ = pb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
            dist = np.linalg.norm(np.array(robot_pos[:2]) - np.array(dep_pos_world[:2]))
            if 0.05 <= dist <= self.cell_size * 1.6:  # 8-adjacent cell dropoff range (~0.32m)
                target_arm_yaw = self._get_cardinal_direction_angle(dep_pos_world, robot_pos, yaw)
                start_finger = self.current_finger_pos
                start_lidar = self.current_lidar_height

                half_substeps = num_substeps // 2

                start_res_x = rx + 0.15 * math.cos(yaw)
                start_res_y = ry + 0.15 * math.sin(yaw)

                # Phase 1: Swivel arm toward depot cell, place resource on depot, open fingers, lower lidar
                for i in range(1, half_substeps + 1):
                    alpha = i / float(half_substeps)
                    self.current_arm_yaw = alpha * target_arm_yaw
                    self.current_finger_pos = start_finger + alpha * (0.015 - start_finger)
                    self.current_lidar_height = start_lidar + alpha * (0.0 - start_lidar)

                    cur_res_x = start_res_x + alpha * (dep_pos_world[0] - start_res_x)
                    cur_res_y = start_res_y + alpha * (dep_pos_world[1] - start_res_y)
                    pb.resetBasePositionAndOrientation(self.resource_id, [cur_res_x, cur_res_y, 0.05], robot_orn, physicsClientId=self.client_id)

                    self._step_substep(left_wheel_delta=0.0, right_wheel_delta=0.0)

                # Phase 2: Swivel arm BACK to forward position (0.0) while resource stays on depot
                for i in range(1, half_substeps + 1):
                    alpha = i / float(half_substeps)
                    self.current_arm_yaw = target_arm_yaw + alpha * (0.0 - target_arm_yaw)
                    self.current_finger_pos = 0.015
                    self.current_lidar_height = 0.0

                    pb.resetBasePositionAndOrientation(self.resource_id, [dep_pos_world[0], dep_pos_world[1], 0.05], robot_orn, physicsClientId=self.client_id)

                    self._step_substep(left_wheel_delta=0.0, right_wheel_delta=0.0)

                self.is_carrying = False
                self.current_arm_yaw = 0.0  # Reset arm link back to forward position
                self.current_finger_pos = 0.015
                self.current_lidar_height = 0.0
                pb.resetBasePositionAndOrientation(self.resource_id, [dep_pos_world[0], dep_pos_world[1], 0.05], robot_orn, physicsClientId=self.client_id)
                reward += 20.0
                terminated = True
            else:
                for _ in range(num_substeps):
                    self._step_substep(left_wheel_delta=0.0, right_wheel_delta=0.0)
                    if self.is_carrying:
                        arm_world_angle = yaw + self.current_arm_yaw
                        res_x = rx + 0.15 * math.cos(arm_world_angle)
                        res_y = ry + 0.15 * math.sin(arm_world_angle)
                        pb.resetBasePositionAndOrientation(self.resource_id, [res_x, res_y, 0.05], robot_orn, physicsClientId=self.client_id)

        else:  # Action 6 (Stay) or other unhandled state
            for _ in range(num_substeps):
                self._step_substep(left_wheel_delta=0.0, right_wheel_delta=0.0)
                if self.is_carrying:
                    arm_world_angle = yaw + self.current_arm_yaw
                    res_x = rx + 0.15 * math.cos(arm_world_angle)
                    res_y = ry + 0.15 * math.sin(arm_world_angle)
                    pb.resetBasePositionAndOrientation(self.resource_id, [res_x, res_y, 0.05], robot_orn, physicsClientId=self.client_id)

        # Collision detection (Obstacles, Wall boundaries, Depot, and Resource when not carrying)
        contacts = pb.getContactPoints(bodyA=self.robot_id, physicsClientId=self.client_id)
        for contact in contacts:
            bodyB = contact[2]
            if bodyB in self.obstacle_ids or bodyB in self.wall_ids or bodyB == self.depot_id or (not self.is_carrying and bodyB == self.resource_id):
                reward -= 2.0
                terminated = True
                break

        # Exploration Bonus & Local Visual Attraction Reward
        if not terminated:
            curr_robot_pos, _ = pb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
            curr_grid_pos = self._world_to_grid(curr_robot_pos[0], curr_robot_pos[1])
            if curr_grid_pos not in self.visited_cells:
                self.visited_cells.add(curr_grid_pos)
                reward += 0.1  # Cell coverage exploration bonus

            # Local Visual Attraction: Target is visible in 15x15 local grid matrix (~1.4m radius)
            if prev_carrying == self.is_carrying:
                dist_curr = np.linalg.norm(np.array(curr_robot_pos[:2]) - np.array(target_pos_world))
                if dist_curr <= 1.4:
                    pbrs_reward = (dist_prev - dist_curr) * 2.0
                    reward += pbrs_reward

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _get_lidar_scan(self, num_rays=180, max_range=2.0):
        """Simulate 360-degree 2D LiDAR using PyBullet rayTestBatch."""
        robot_pos, robot_orn = pb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        rx, ry, rz = robot_pos
        yaw = pb.getEulerFromQuaternion(robot_orn)[2]

        ray_froms = []
        ray_tos = []
        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

        # Raised Lidar ray height z=0.15 when carrying (passes over resource top z=0.10)
        if getattr(self, "has_urdf_wheels", False):
            lidar_z = rz + 0.15 if self.is_carrying else rz + 0.08
        else:
            lidar_z = rz + 0.15 if self.is_carrying else rz + 0.05

        for angle in angles:
            abs_angle = yaw + angle
            ray_froms.append([rx, ry, lidar_z])
            ray_tos.append([
                rx + max_range * math.cos(abs_angle),
                ry + max_range * math.sin(abs_angle),
                lidar_z
            ])

        results = pb.rayTestBatch(ray_froms, ray_tos, physicsClientId=self.client_id)
        return results, ray_froms, ray_tos


    def _get_obs(self):
        grid = np.zeros((self.obs_size, self.obs_size, 5), dtype=np.float32)
        
        robot_pos, robot_orn = pb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        rx, ry, _ = robot_pos
        yaw = pb.getEulerFromQuaternion(robot_orn)[2]

        half_obs = self.obs_size // 2

        # Get LiDAR hits for obstacles / walls strictly (excluding resource and depot)
        lidar_results, _, _ = self._get_lidar_scan(num_rays=180, max_range=2.0)
        for hit in lidar_results:
            hit_uid, _, hit_frac, hit_pos, _ = hit
            if (hit_uid in self.obstacle_ids or hit_uid in self.wall_ids) and hit_frac < 1.0:
                hx, hy = hit_pos[0], hit_pos[1]
                # Convert world hit position to egocentric coordinates (forward=dx, left=dy)
                rel_x = hx - rx
                rel_y = hy - ry
                dx = rel_x * math.cos(yaw) + rel_y * math.sin(yaw)
                dy = -rel_x * math.sin(yaw) + rel_y * math.cos(yaw)

                local_r = int(round(half_obs - (dx / self.cell_size)))
                local_c = int(round(half_obs - (dy / self.cell_size)))

                if 0 <= local_r < self.obs_size and 0 <= local_c < self.obs_size:
                    grid[local_r, local_c, 0] = 1.0  # Obstacle channel (strictly obstacles/walls)

        # Populate Egocentric Grid for Resource, Depot, Boundaries, and Self-Agent
        res_pos, _ = pb.getBasePositionAndOrientation(self.resource_id, physicsClientId=self.client_id)
        dep_pos, _ = pb.getBasePositionAndOrientation(self.depot_id, physicsClientId=self.client_id)

        for r_idx in range(self.obs_size):
            for c_idx in range(self.obs_size):
                dx = (half_obs - r_idx) * self.cell_size
                dy = (half_obs - c_idx) * self.cell_size

                # Transform local ego offset to world coordinates
                wx = rx + dx * math.cos(yaw) - dy * math.sin(yaw)
                wy = ry + dx * math.sin(yaw) + dy * math.cos(yaw)

                # Channel 3: Boundary / Void (arena bounds are 4m x 4m -> [-2.0, 2.0])
                if abs(wx) >= 2.0 or abs(wy) >= 2.0:
                    grid[r_idx, c_idx, 3] = 1.0

                # Channel 1: Resource
                if not self.is_carrying:
                    if math.hypot(wx - res_pos[0], wy - res_pos[1]) <= self.cell_size * 0.5:
                        grid[r_idx, c_idx, 1] = 1.0

                # Channel 2: Depot
                if math.hypot(wx - dep_pos[0], wy - dep_pos[1]) <= self.cell_size * 0.7:
                    grid[r_idx, c_idx, 2] = 1.0

        # Channel 4: Self-Agent center & heading indicator
        grid[half_obs, half_obs, 4] = 1.0  # Center cell (robot location)
        if half_obs - 1 >= 0:
            grid[half_obs - 1, half_obs, 4] = 0.5

        return {
            "grid": grid,
            "is_carrying": int(self.is_carrying)
        }

    def _get_info(self):
        if self.robot_id is not None:
            robot_pos, _ = pb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        else:
            robot_pos = (0.0, 0.0, 0.0)
        lidar_results, _, _ = self._get_lidar_scan(num_rays=180, max_range=2.0)
        hit_distances = [hit[2] * 2.0 for hit in lidar_results]
        return {
            "robot_pos": robot_pos,
            "lidar_distances": hit_distances
        }

    def render(self):
        pass

    def close(self):
        if hasattr(self, 'client_id'):
            pb.disconnect(physicsClientId=self.client_id)
