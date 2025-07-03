# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AssemblyKitEnv: direct‐RL environment for the Franka assembly‐kit benchmark."""

from __future__ import annotations

import json
from pathlib import Path
import random
import math
import torch
from collections.abc import Sequence

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObjectCfg, RigidObject, AssetBase
from isaaclab.sensors import Camera
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import convert_to_torch
from isaaclab.utils.math import quat_from_euler_xyz, euler_xyz_from_quat

from .assembly_kit_env_cfg import AssemblyKitEnvCfg


class AssemblyKitEnv(DirectRLEnv):
    """Direct-RL environment for the Assembly-Kit task in Isaac Lab.

    This class handles scene setup, TCP computation (both local and world),
    observation & reward computation, and environment resets for the Franka
    assembly-kit benchmark.

    Attributes:
        cfg (AssemblyKitEnvCfg): Configuration for this environment.
        robot_local_grasp_pos (Tensor): Precomputed local TCP positions.
        robot_local_grasp_rot (Tensor): Precomputed local TCP orientations.
        hand_link_idx (int): Index of the wrist link in the Articulation.
        left_finger_link_idx (int): Index of the left-finger link.
        right_finger_link_idx (int): Index of the right-finger link.
    """

    cfg: AssemblyKitEnvCfg

    # === Add helper functions for computing local TCP ===
    @staticmethod
    def get_env_local_pose(
        env_origin: torch.Tensor, prim_path: str, device: torch.device
    ) -> torch.Tensor:
        """Get a prim’s pose in environment-local coordinates.

        Args:
            env_origin (Tensor[3]): The world-space origin of this env.
            prim_path (str): USD path to the prim (e.g. "/World/.../panda_link7").
            device (torch.device): Device on which to place the result.

        Returns:
            Tensor[7]: [x, y, z, qw, qx, qy, qz] of the prim in env-local frame.
        """

        stage = get_current_stage()
        xformable = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
        world_tf = xformable.ComputeLocalToWorldTransform(0)
        world_pos = world_tf.ExtractTranslation()
        world_quat = world_tf.ExtractRotationQuat()

        px = world_pos[0] - env_origin[0]
        py = world_pos[1] - env_origin[1]
        pz = world_pos[2] - env_origin[2]
        qw = world_quat.real
        qx, qy, qz = world_quat.imaginary

        return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

    @staticmethod
    def compute_local_tcp(
        env_origin: torch.Tensor,
        wrist_prim_path: str,
        lfinger_prim_path: str,
        rfinger_prim_path: str,
        device: torch.device,
        y_offset: float = 0.04,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the gripper’s local tool-center-point (TCP) in the wrist frame.

        Finds the midpoint between left and right finger tips and applies a forward offset.

        Args:
            env_origin:        Tensor giving the environment origin.
            wrist_prim_path:   USD path of the wrist link.
            lfinger_prim_path: USD path of the left finger tip.
            rfinger_prim_path: USD path of the right finger tip.
            device:            Torch device for computation.
            y_offset:          Offset along wrist Y to push TCP forward.

        Returns:
            local_grasp_pos: TCP position in the wrist link frame.
            local_grasp_rot: TCP orientation (quaternion) in the wrist link frame.
        """

        # 1) get initial wrist, left‐finger, right‐finger poses in env-local coords
        wrist_pose = AssemblyKitEnv.get_env_local_pose(
            env_origin, wrist_prim_path, device
        )
        lfinger_pose = AssemblyKitEnv.get_env_local_pose(
            env_origin, lfinger_prim_path, device
        )
        rfinger_pose = AssemblyKitEnv.get_env_local_pose(
            env_origin, rfinger_prim_path, device
        )

        # 2) build midpoint‐finger pose (env-local)
        finger_mid = torch.zeros(7, device=device)
        finger_mid[0:3] = (
            lfinger_pose[0:3] + rfinger_pose[0:3]
        ) / 2.0  # average position
        finger_mid[3:7] = lfinger_pose[3:7]  # pick left‐finger's quaternion

        # 3) invert wrist pose
        wrist_pos_env = wrist_pose[0:3]
        wrist_quat_env = wrist_pose[3:7]
        wrist_inv_rot, wrist_inv_pos = tf_inverse(wrist_quat_env, wrist_pos_env)

        # 4) combine wrist‐inverse with finger_mid to get TCP in wrist‐frame
        local_grasp_rot, local_grasp_pos = tf_combine(
            wrist_inv_rot,
            wrist_inv_pos,
            finger_mid[3:7],
            finger_mid[0:3],
        )

        # 5) apply small offset along wrist‐Y
        local_grasp_pos += torch.tensor([0.0, y_offset, 0.0], device=device)

        return local_grasp_pos, local_grasp_rot

    def __init__(
        self, cfg: AssemblyKitEnvCfg, render_mode: str | None = None, **kwargs
    ):
        """Initializes the assembly-kit environment and computes per-env local TCPs.

        Sets up asset directories, loads episodes, spawns the scene, and precomputes
        each Franka gripper’s TCP in its wrist frame.

        Args:
            cfg:         AssemblyKitEnvCfg with environment parameters.
            render_mode: Optional render mode (e.g. "rgb" or "state").
            **kwargs:    Additional arguments for the base DirectRLEnv.
        """

        self.asset_dir = Path("/home/johann/Downloads/assembly_kit_noremesh")
        self.kit_dir = self.asset_dir.joinpath("kits")
        self.model_dir = self.asset_dir.joinpath("models")

        # parsing json episode file
        with open(self.asset_dir.joinpath("episodes.json")) as json_data:
            self._episode_json = json.load(json_data)
        self._episodes = self._episode_json["episodes"]

        self.symmetry = self._episode_json["config"]["symmetry"]

        self.color = self._episode_json["config"]["color"]
        self.object_scale = self._episode_json["config"]["object_scale"]

        super().__init__(cfg, render_mode, **kwargs)

        self.symmetry = torch.tensor(
            self.symmetry, dtype=torch.float32, device=self.device
        )

        random.seed(self.cfg.seed)

        self.joint_ids, _ = self.robot.find_joints("panda_joint.*|panda_finger_joint.*")

        # Compute and store local TCP once
        device = self.device
        origins = self.scene.env_origins

        local_pos_list = []
        local_rot_list = []
        for i in range(self.num_envs):
            env_origin = origins[i]
            wrist_path = f"/World/envs/env_{i}/Robot/panda_link7"
            lfinger_path = f"/World/envs/env_{i}/Robot/panda_leftfinger"
            rfinger_path = f"/World/envs/env_{i}/Robot/panda_rightfinger"

            pos, rot = AssemblyKitEnv.compute_local_tcp(
                env_origin,
                wrist_path,
                lfinger_path,
                rfinger_path,
                device,
            )
            local_pos_list.append(pos)
            local_rot_list.append(rot)

        self.robot_local_grasp_pos = torch.stack(local_pos_list, dim=0)
        self.robot_local_grasp_rot = torch.stack(local_rot_list, dim=0)

        # Find relevant link indices for runtime TCP computation
        self.hand_link_idx = self.robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self.robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self.robot.find_bodies("panda_rightfinger")[0][0]

    # === New helper to compute world‐TCP each step ===
    def _compute_tcp_world(
        self, env_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the world-frame TCP pose for the selected envs.

        Reads the current wrist link pose, combines it with the stored local TCP,
        and converts positions to environment-local coordinates.

        Args:
            env_ids: Tensor of environment indices (all if None).

        Returns:
            tcp_pos_env: Tensor of shape (N, 3) of TCP positions in env-local coords.
            tcp_rot:     Tensor of shape (N, 4) of TCP orientations (quaternion).
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # read current wrist world-pose
        hand_pos_world = self.robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot_world = self.robot.data.body_quat_w[env_ids, self.hand_link_idx]

        # combine with stored local grasps to get world TCP
        global_tcp_rot, global_tcp_pos_world = tf_combine(
            hand_rot_world,
            hand_pos_world,
            self.robot_local_grasp_rot[env_ids],
            self.robot_local_grasp_pos[env_ids],
        )

        # convert world TCP pos to env-local TCP pos
        env_origins = self.scene.env_origins[env_ids]
        tcp_pos_env = global_tcp_pos_world - env_origins

        return tcp_pos_env, global_tcp_rot

    def _load_table_scene(self):
        """Spawns the ground plane, table, robot, and camera into the simulation.

        Creates a dome light and sets up per-env cloning and debug visualization.
        """

        # Creating the default scene
        spawn_ground_plane(
            prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0, 0, 0)
        )

        # spawn_ground_plane(
        #     prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0, 0, -1.05)
        # )

        # self.table = AssetBase(self.cfg.table_cfg)

        self.robot = Articulation(self.cfg.robot_cfg)

        camera = Camera(cfg=self.cfg.sensors[0])
        camera.set_debug_vis(True)
        self.scene.sensors["camera"] = camera

        # clone and replicate
        self.scene.clone_environments(copy_from_source=True)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _setup_scene(self):
        """Imports kits and models, reads episode JSON, and initializes sampling.

        Spawns the kit and object models, builds lookup tables, and filters inter-env collisions.
        """

        self._load_table_scene()

        # Collecting all the kit USD paths
        kit_usd_paths = sorted(
            [str(p / f"{p.stem}.usda") for p in self.kit_dir.iterdir() if p.is_dir()],
            key=lambda p: int(Path(p).stem.split("_")[-1]),
        )

        # Importing the kits
        self.kit_cfg: RigidObjectCfg = self._get_kit_cfg(kit_usd_paths)
        self.kit: RigidObject = RigidObject(self.kit_cfg)
        self.scene.rigid_objects["Kit"] = self.kit

        # Collecting all the model paths from the kit JSON files
        kit_json_paths = sorted(
            [str(p) for p in self.kit_dir.iterdir() if p.match("*.json")],
            key=lambda p: int(Path(p).stem.split("_")[-1]),
        )

        # Parsing the kit JSON files to get the object IDs and the goal poses
        kit_model_ids = []
        kit_object_positions = []
        kit_object_rots = []
        kit_target_starting_pos = []
        for kit_json_path in kit_json_paths:
            with open(kit_json_path) as json_data:
                kit_json = json.load(json_data)
            kit_model_ids.append([obj["object_id"] for obj in kit_json["objects"]])
            poses = [o["pos"] for o in kit_json["objects"]]
            rots = [o["rot"] for o in kit_json["objects"]]
            kit_object_positions.append(poses)
            kit_object_rots.append(rots)
            kit_target_starting_pos.append(kit_json["start_pos_proposal"])

        self.kit_target_starting_pos = torch.tensor(
            kit_target_starting_pos, dtype=torch.float32, device=self.device
        )
        self.model_target_pos = torch.tensor(
            kit_object_positions, dtype=torch.float32, device=self.device
        )
        self.model_target_rot = torch.tensor(
            kit_object_rots, dtype=torch.float32, device=self.device
        )
        # Stash which kit each env got
        self.kit_ids_per_env = torch.arange(self.num_envs, device=self.device) % len(
            kit_usd_paths
        )

        # Grouping the model paths for MultiUsdFileCfg
        kit_models_paths = [
            [
                str(
                    self.model_dir.joinpath(
                        f"model_{model_id:02d}", f"model_{model_id:02d}.usda"
                    )
                )
                for model_id in group
            ]
            for group in zip(*kit_model_ids)
        ]

        # Importing the models
        self.models: list[RigidObject] = []
        for idx, kit_model_paths in enumerate(kit_models_paths):
            model_cfg = self._get_model_cfg(kit_model_paths, idx)
            self.models.append(RigidObject(model_cfg))
            self.scene.rigid_objects[f"Model_{idx}"] = self.models[-1]

        # Computing the Asset lookup table for the environment instances
        self.env_assets_info = self._get_assets_info_per_env(
            kit_usd_paths, kit_model_ids
        )

        self._init_model_sampling()

        # Filtering collisions for optimization of collisions between environment instances
        self.scene.filter_collisions(
            [
                "/World/ground",
            ]
        )

    def _get_kit_goals(self, kit_id: str) -> tuple[dict, dict]:
        """Parses a kit JSON to extract goal positions and rotations.

        Args:
            kit_id: Identifier of the kit (e.g. "kit_01").

        Returns:
            model_goal_pos: Dict mapping object_id to goal position tensor.
            model_goal_rot: Dict mapping object_id to goal z-rotation.
        """

        # parsing json episode file
        with open(self.kit_dir.joinpath(f"{kit_id}.json")) as json_data:
            kit_json = json.load(json_data)
            # the final 3D goal position of the model
            model_goal_pos = {
                o["object_id"]: convert_to_torch(o["pos"]) for o in kit_json["objects"]
            }
            # the final goal z-axis rotation of the objects
            model_goal_rot = {o["object_id"]: o["rot"] for o in kit_json["objects"]}
            return model_goal_pos, model_goal_rot

    def _get_kit_cfg(self, kit_usd_paths: list[str]) -> RigidObjectCfg:
        """Builds the RigidObjectCfg for the kit assembly platform.

        Args:
            kit_usd_paths: List of USD paths for kit variants.

        Returns:
            Config object for spawning the kit in all envs.
        """
        kit_cfg: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Kit",
            spawn=sim_utils.MultiUsdFileCfg(
                usd_path=kit_usd_paths,
                random_choice=False,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=False,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                    sleep_threshold=0.005,
                    stabilization_threshold=0.0025,
                    max_depenetration_velocity=1000.0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.27807487, 0.20855615, 0.16934046),
                    emissive_color=(0.0, 0.0, 0.0),
                    roughness=0.5,
                    metallic=0.0,
                    opacity=1.0,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.55, 0.0, 0.007), rot=(0.0, 0.0, 0.0, 1.0)
            ),
        )

        return kit_cfg

    def _get_model_cfg(self, model_paths: list[str], model_idx: int) -> RigidObjectCfg:
        """Generates the RigidObjectCfg for a single model with randomized color.

        Args:
            model_paths: List of USD paths for the model variants.
            model_idx:   Index of this model in the kit.

        Returns:
            Config object for spawning the model in the scene.
        """
        r, g, b, a = self.color[random.randint(0, len(self.color) - 1)]

        return RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/Model_{model_idx}",
            spawn=sim_utils.MultiUsdFileCfg(
                usd_path=model_paths,
                random_choice=False,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=False,
                    disable_gravity=False,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                    sleep_threshold=0.005,
                    stabilization_threshold=0.0025,
                    max_depenetration_velocity=1000.0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(r, g, b),
                    emissive_color=(0.0, 0.0, 0.0),
                    roughness=0.5,
                    metallic=0.0,
                    opacity=a,
                ),
                scale=(0.98, 0.98, 1),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.55, 0, 0.1), rot=(0.0, 0.0, 0.0, 1.0)
            ),
        )

    def _get_assets_info_per_env(
        self,
        kit_usd_paths: list[str],
        kit_model_ids: list[list[int]],
    ) -> list[dict]:
        """Constructs per-env metadata about which kit and models to use.

        Args:
            kit_usd_paths:  List of all kit USD paths.
            kit_model_ids:  Nested list mapping kit to its object IDs.

        Returns:
            List of dicts per env with 'kit_id' and 'model_ids'.
        """
        num_envs = self.num_envs
        num_kits = len(kit_usd_paths)
        return [
            {
                "kit_id": env_id % num_kits,
                "model_ids": kit_model_ids[env_id % num_kits],
            }
            for env_id in range(num_envs)
        ]

    def _init_model_sampling(self) -> None:
        """Preallocates tensors for target and other model sampling across envs."""
        # pulled straight from the list-of-dicts you already built
        model_ids_list = [info["model_ids"] for info in self.env_assets_info]
        # assume same-length lists → a dense matrix
        self.model_ids_matrix_per_env = torch.tensor(
            model_ids_list,
            dtype=torch.long,
            device=self.device,
        )

        self.num_shape_types = int(self.model_ids_matrix_per_env.max().item()) + 1

        num_envs, K = self.model_ids_matrix_per_env.shape
        # to write into each reset
        self._current_target = torch.empty(
            num_envs, 1, dtype=torch.long, device=self.device
        )
        # one fewer column
        self._current_others = torch.empty(
            num_envs, K - 1, dtype=torch.long, device=self.device
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # self.robot.set_joint_effort_target(self.actions, joint_ids=self.joint_ids)
        self.robot.set_joint_position_target(self.actions, joint_ids=self.joint_ids)

    def _get_observations(self) -> dict:
        """Collects state or pixel observations for the policy.

        Reads TCP pose, target model pose, computes shape one-hot, and returns
        either concatenated state vectors or camera images.

        Returns:
            Dict with key 'policy' mapping to either a state tensor or image tensor.
        """
        tcp_pos_env, tcp_rot = self._compute_tcp_world()

        target_model_pose = self.get_target_model_pose()

        shape_one_hot = (
            torch.nn.functional.one_hot(
                self.current_target_model_id, num_classes=self.num_shape_types
            )
            .to(self.device)
            .float()
        )

        state_obs = torch.cat(
            (
                tcp_pos_env,
                tcp_rot,
                target_model_pose,
                target_model_pose[:, :3] - tcp_pos_env,
                self.target_goal_pos,
                self.target_goal_rot.unsqueeze(1),
                self.target_goal_pos - target_model_pose[:, :3],
                shape_one_hot,
            ),
            dim=-1,
        )

        rgb = self.scene.sensors["camera"].data.output["rgb"]
        pixels = (rgb.to(torch.float32) / 255.0).clone()  # normalize to [0,1]

        # return according to obs_mode
        if self.cfg.obs_mode == "state":
            return {"policy": state_obs}
        else:
            return {"policy": pixels}

    def get_target_model_pose(self) -> torch.Tensor:
        """Returns the world pose of the current target object in env-local coords.

        Returns:
            Tensor of shape (num_envs, 7) with position and quaternion orientation.
        """

        # Get the index of the target object for each environment
        target_models: list[RigidObject] = [
            self.models[idx.item()] for idx in self._current_target[:, 0]
        ]

        target_model_pose = torch.zeros(
            (self.num_envs, 7), dtype=torch.float32, device=self.device
        )

        for idx, model in enumerate(target_models):
            target_model_pose[idx] = model.data.root_state_w[idx, :7]

        # convert to env-local
        target_model_pose[:, :3] -= self.scene.env_origins

        return target_model_pose

    def _get_rewards(self) -> torch.Tensor:
        total_reward = self.compute_rewards()
        return total_reward

    def compute_rewards(self):
        success_mask = self.is_success()
        ones = torch.ones_like(success_mask, dtype=torch.int, device=self.device)
        return torch.where(success_mask, ones, -ones)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = self.is_success()

        return terminated, time_out

    def is_success(
        self, pos_eps=2e-2, rot_eps=math.radians(4), height_eps=3e-3
    ) -> torch.Tensor:
        """Checks if target objects are correctly placed and oriented.

        Args:
            pos_eps:    Positional tolerance in XY plane.
            rot_eps:    Rotational tolerance around Z axis.
            height_eps: Height threshold for placement in slot.

        Returns:
            Boolean tensor per env indicating success.
        """

        target_model_pose = self.get_target_model_pose()

        # Compute positional difference (XY-plane)
        pos_diff = self.target_goal_pos[:, :2] - target_model_pose[:, :2]
        pos_diff_norm = torch.norm(pos_diff, dim=1)
        pos_correct = pos_diff_norm < pos_eps

        # Compute rotational difference considering symmetry
        target_model_quat = target_model_pose[:, 3:7]
        target_model_euler = euler_xyz_from_quat(target_model_quat)
        target_model_z_rot = target_model_euler[2]

        goal_z_rot = self.target_goal_rot
        rot_diff = (
            torch.abs(target_model_z_rot - goal_z_rot)
            % self.symmetry[self.current_target_model_id]
        )

        # Adjust symmetry difference
        symmetry_val = self.symmetry[self.current_target_model_id]
        half_symmetry = symmetry_val / 2
        rot_diff = torch.where(
            rot_diff > half_symmetry, symmetry_val - rot_diff, rot_diff
        )
        rot_correct = rot_diff < rot_eps

        # Check height to determine if the object is correctly placed in the slot
        height_correct = target_model_pose[:, 2] < height_eps

        success = pos_correct & rot_correct & height_correct

        return success

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Resets robot, kit, and object states for the given env indices.

        Args:
            env_ids: List or tensor of environment indices to reset.
        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        # Resetting the robot
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # resetting kit
        kit_default_state = self.kit.data.default_root_state.clone()[env_ids]
        kit_default_state[:, :3] += self.scene.env_origins[env_ids]
        self.kit.write_root_pose_to_sim(kit_default_state[:, :7], env_ids)
        self.kit.write_root_velocity_to_sim(kit_default_state[:, 7:], env_ids)

        # sample objects for the environments
        self.sample_models_for_envs(
            torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        )

    def sample_models_for_envs(self, env_ids: torch.Tensor) -> None:
        """Randomly selects a target model and places others at goal or side positions.

        Args:
            env_ids: Tensor of environment indices for sampling.
        """
        device = self.device
        models = self.models

        num_envs = env_ids.shape[0]
        num_models = self.model_target_pos.shape[1]

        # determine kit positions
        kit_ids = self.kit_ids_per_env[env_ids]
        kit_def = self.kit.data.default_root_state.clone()[env_ids]
        kit_pos_w = kit_def[:, :3] + self.scene.env_origins[env_ids]

        # pulling JSON target poses for the other model
        model_rel_pos = self.model_target_pos[kit_ids]
        model_rel_rot = self.model_target_rot[kit_ids]

        # pick target model and compute other model indices
        choice = torch.randint(0, num_models, (num_envs,), device=device)
        self._current_target[env_ids] = choice.unsqueeze(1)
        self.current_target_model_id = (
            torch.tensor(
                [info["model_ids"] for info in self.env_assets_info], device=self.device
            )
            .gather(1, self._current_target)
            .squeeze(1)
        )
        all_cols = (
            torch.arange(num_models, device=device)
            .unsqueeze(0)
            .expand(num_envs, num_models)
        )
        others_mask = all_cols != choice.unsqueeze(1)
        others_idx = all_cols[others_mask].view(num_envs, num_models - 1)

        # compute and target starting poses
        sampled_idx = torch.randint(
            low=0,
            high=self.kit_target_starting_pos.shape[1],
            size=(num_envs,),
            device=device,
        )
        offsets = self.kit_target_starting_pos[kit_ids, sampled_idx]
        target_pos = kit_pos_w + offsets
        rot_t = torch.rand(num_envs, device=device) * 2 * math.pi
        target_quat = quat_from_euler_xyz(
            torch.zeros(num_envs, device=device),
            torch.zeros(num_envs, device=device),
            rot_t,
        )

        # Storing the target poses and orientations for observation and reward
        self.target_pos = offsets
        self.target_quat = target_quat
        self.target_goal_pos = model_rel_pos[
            torch.arange(model_rel_pos.size(0), device=choice.device), choice
        ]
        self.target_goal_rot = model_rel_rot[
            torch.arange(model_rel_pos.size(0), device=choice.device), choice
        ]

        # compute world-space poses + rots for others models
        idx_b = torch.arange(num_envs, device=device).unsqueeze(1)
        others_pos = model_rel_pos[idx_b, others_idx]
        others_rot = model_rel_rot[idx_b, others_idx]

        mask_other_to_side = torch.zeros_like(
            others_rot, dtype=torch.bool, device=device
        )
        do_pick = torch.rand(num_envs, device=device) < 0.5
        cols = torch.randint(0, num_models - 1, (num_envs,), device=device)
        rows = torch.arange(num_envs, device=device)
        mask_other_to_side[rows[do_pick], cols[do_pick]] = True
        others_pos[mask_other_to_side] = torch.tensor(
            [[-self.cfg.table_offset, 0.2, 0.1]],
            device=device,
            dtype=others_pos.dtype,
        )
        others_pos_w = (
            kit_pos_w.unsqueeze(1).expand(num_envs, num_models - 1, 3) + others_pos
        )
        others_rot_w = others_rot
        # flatten for per-model grouping
        envs_o_flat = env_ids.unsqueeze(1).expand(num_envs, num_models - 1).reshape(-1)
        idx_o_flat = others_idx.reshape(-1)
        pos_o_flat = others_pos_w.reshape(-1, 3)
        rot_o_flat = others_rot_w.reshape(-1)

        for model_idx in range(num_models):
            default = models[model_idx].data.default_root_state[env_ids].clone()

            mask_t = choice == model_idx
            if mask_t.any():
                envs_t = env_ids[mask_t]
                poses_t = target_pos[mask_t]
                quats_t = target_quat[mask_t]

                default[envs_t, :3] = poses_t
                default[envs_t, 3:7] = quats_t

            mask_o = idx_o_flat == model_idx
            if mask_o.any():
                envs_o = envs_o_flat[mask_o]
                poses_o = pos_o_flat[mask_o]
                rots_o = rot_o_flat[mask_o]

                quats_o = quat_from_euler_xyz(
                    torch.zeros_like(mask_o.sum(), device=device),
                    torch.zeros_like(mask_o.sum(), device=device),
                    rots_o,
                )

                default[envs_o, :3] = poses_o
                default[envs_o, 3:7] = quats_o

            models[model_idx].write_root_pose_to_sim(default[:, :7], env_ids)
