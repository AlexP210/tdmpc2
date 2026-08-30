"""TEMPORARY (2026-08-26): DINO goal-image reward for PushCube, for one overnight TD-MPC2 run.

Delete this file and revert `custom_maniskill.py` and the two configs to drop the experiment.

What it does: replaces PushCube's dense reward with closeness in DINOv3 feature space to a
per-episode goal *image* -- the episode's own scene with the cube sitting on the centre of its
target and the arm posed a few cm behind the cube, so the wrist camera is looking down at the
finished task.

`dino_reward_shaping` picks how that distance becomes a reward. The default is
`negative_distance` (`r_t = -d_t`), which keeps paying while the goal is held -- the right shape
here because `IgnoreTerminations` runs every episode to its 50-step limit and the task is scored
on the final frame. `delta` (`r_t = d_{t-1} - d_t`) is the potential-based alternative, whose
return telescopes to `d_0 - d_T` and pays nothing for staying at the goal. They need very
different `dino_reward_scale` values; see the config comment.

Three things here are less obvious than they look, and each is load-bearing:

1. The goal is rendered by a **second env instance** (the "mirror"), not by teleporting the
   training env and restoring it. Save/teleport/restore looks equivalent but is not: restoring a
   state dict resets physx solver state that the state dict does not capture, and the training
   rollout then drifts (measured: ~4% of pixels differ within 30 steps, growing). The mirror is
   only ever written to, the training env only ever read, so the training trajectory is bit-for-bit
   what it would have been without this file.

2. The arm pose comes from an iterative IK that **walks the target in** and stops at the last
   reachable waypoint. `Kinematics.compute_ik` is a single Levenberg-Marquardt step and diverges
   over a long jump (measured: 28 cm off for a 30 cm jump), and PushCube's goal centre can sit
   outside the Panda's reach at cube height -- the goal region is at `cube_xy + [0.2, 0]` and the
   cube starts as far out as x=0.1, putting the target 0.85 m from a base whose reach at z=0.08 is
   about 0.79 m. Neither the step size nor the early stop is optional.

3. The goal is set on the reward wrapper *during* the inner reset, so `DINORewardWrapper` seeds
   `d_0` against this episode's goal rather than the last one's. `PushCubeDinoReward.reset` does
   that by overriding reset rather than by calling `set_goal` after the fact.
"""

from __future__ import annotations

import os
import pathlib
import types

import numpy as np
import torch

from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion
from mani_skill.utils.structs import Pose

from custom_maniskill_tasks import DINORewardWrapper

# EE position for the goal pose, as an offset from the cube's goal centre in world axes.
# -6 cm in x puts the gripper behind the cube (PushCube pushes towards +x, the robot base is at
# x=-0.615), +6 cm in z lifts the wrist so the whole target is in frame; both were picked by
# rendering the alternatives -- see tools/preview_dino_goal.py.
CUBE_TO_EE = (-0.06, 0.0, 0.06)

IK_STEP = 0.02          # metres of target travel per waypoint
IK_ITERS_PER_STEP = 3
IK_TOLERANCE = 2e-3     # a waypoint missed by more than this counts as out of reach


def _fk_at_base(kinematics, qpos: torch.Tensor) -> Pose:
    """TCP pose in the robot-base frame, from joint angles alone -- no sim state involved."""
    matrix = kinematics.pk_chain.forward_kinematics(
        qpos[:, kinematics.active_ancestor_joint_idxs]
    ).get_matrix()
    return Pose.create_from_pq(matrix[:, :3, 3], matrix_to_quaternion(matrix[:, :3, :3]))


def solve_ik(base_env, target_world_p: torch.Tensor):
    """Joint angles putting the TCP at `target_world_p`, or as close as the arm can reach.

    Returns `(qpos, residual)`; `residual` is how far short of the target it stopped, which is
    non-zero exactly when the target is outside the workspace. See note 2 in the module docstring.
    """
    controller = base_env.agent.controller.controllers["arm"]
    kinematics = controller.kinematics
    arm = controller.active_joint_indices.long()
    limits = base_env.agent.robot.get_qlimits()

    target = (controller.root_link.pose.inv() * Pose.create_from_pq(target_world_p)).p
    qpos = base_env.agent.robot.get_qpos().clone()
    start = _fk_at_base(kinematics, qpos).p
    best = qpos.clone()

    span = torch.linalg.norm(target - start, dim=-1).max()
    waypoints = max(1, int(torch.ceil(span / IK_STEP).item()))
    for step in range(1, waypoints + 1):
        waypoint = start + (target - start) * (step / waypoints)
        for _ in range(IK_ITERS_PER_STEP):
            current = _fk_at_base(kinematics, qpos)
            # position-only: reusing the current orientation makes the rotation delta identity,
            # which is also what PDEEPosController does for this control mode
            goal = Pose.create_from_pq(waypoint, current.q)
            qpos[:, arm] = kinematics.compute_ik(goal, qpos, current_pose=current)
            qpos = torch.clamp(qpos, limits[..., 0], limits[..., 1])
        if float(torch.linalg.norm(_fk_at_base(kinematics, qpos).p - waypoint)) > IK_TOLERANCE:
            break  # out of reach from here on; keep the furthest pose that did converge
        best = qpos.clone()

    residual = float(torch.linalg.norm(_fk_at_base(kinematics, best).p - target))
    return best, residual


class GoalMirror:
    """A second copy of the task, used only to render goal images. Never stepped.

    Its state is overwritten on every call, so it holds no episode of its own.
    """

    def __init__(self, env_factory, camera_uid: str):
        self.env = env_factory()
        self.env.reset(seed=0)  # required before set_state_dict has anything to overwrite
        self.base = self.env.unwrapped
        self.camera_uid = camera_uid
        self.last_ik_residual = 0.0

    def goal_image(self, live_state: dict) -> torch.Tensor:
        """The goal frame for the episode whose current state is `live_state`."""
        self.base.set_state_dict(_clone_state(live_state))

        cube_goal = self.base.goal_region.pose.p.clone()
        cube_goal[:, 2] = self.base.cube_half_size
        ee_target = cube_goal + torch.tensor(CUBE_TO_EE, device=cube_goal.device)
        qpos, self.last_ik_residual = solve_ik(self.base, ee_target)

        state = _clone_state(live_state)
        state["actors"]["cube"][:, :3] = cube_goal
        state["actors"]["cube"][:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=cube_goal.device)
        state["actors"]["cube"][:, 7:] = 0.0            # at rest
        state["articulations"]["panda"][:, 13:22] = qpos  # [root pose 7 | root vel 6 | qpos 9 | qvel 9]
        state["articulations"]["panda"][:, 22:] = 0.0
        self.base.set_state_dict(state)

        return self.base.get_obs()["sensor_data"][self.camera_uid]["rgb"].clone()


def _clone_state(state: dict) -> dict:
    return {key: {name: value.clone() for name, value in group.items()}
            for key, group in state.items()}


class PushCubeDinoReward(DINORewardWrapper):
    """`DINORewardWrapper` whose goal image is rebuilt from the mirror on every reset."""

    def __init__(self, env, encoder, mirror: GoalMirror, **kwargs):
        super().__init__(env, encoder, goal_image=None, **kwargs)
        self.mirror = mirror

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        obs, info = result if isinstance(result, tuple) else (result, None)

        self.set_goal(self.mirror.goal_image(self.env.unwrapped.get_state_dict()))

        features = self._encode(self._image(obs))
        self._prev_distance = self._distance(features)
        self._report(info, self._prev_distance, features)
        if isinstance(info, dict):
            info["dino_ik_residual"] = self.mirror.last_ik_residual
        return result


def build_encoder(cfg):
    """The frozen DINOv3 backbone, wrapped in the `(N, H, W, 3) uint8 -> (N, ...)` contract.

    Frames stay uint8 through `permute`: `DINOV3EncoderModel`'s transform reaches [0, 1] with
    `v2.ToDtype(scale=True)`, which is a no-op on float input, so casting first would
    ImageNet-normalize [0, 255] and encode into a different feature space.
    """
    # `dinov3_encoder_model` reads PROJECT_ROOT at import time to find dependencies/dinov3, and
    # TD-MPC2 runs do not export it the way TSD's launchers do. Fill it in from this file's
    # location (agents/tdmpc2/tdmpc2/envs/ -> four levels up) rather than making the caller
    # remember, but never override one that is already set.
    os.environ.setdefault(
        "PROJECT_ROOT", str(pathlib.Path(__file__).resolve().parents[4])
    )
    from tsd.models.dinov3_encoder_model import DINOV3EncoderModel

    encoder_cfg = types.SimpleNamespace(
        model_name=cfg.dino_reward_model,
        checkpoint=cfg.dino_reward_checkpoint,
        token_mode="patch",
        resize_size=cfg.dino_reward_resize,
        observation_key=None,
        freeze=True,
        device=cfg.device,
    )
    resize = cfg.dino_reward_resize
    task = types.SimpleNamespace(observation_dimension=(1, 3, resize, resize))
    model = DINOV3EncoderModel(encoder_cfg, task).eval()

    def encode(images: torch.Tensor) -> torch.Tensor:
        return model.encode(images.permute(0, 3, 1, 2)[None, :, None])[0, :, 0]

    return encode, torch.device(cfg.device)


def reward_wrappers(cfg, env_factory, camera_uid: str):
    """The `reward_wrappers=` list for `custom_maniskill_tasks.make_env`."""
    encoder, device = build_encoder(cfg)
    mirror = GoalMirror(env_factory, camera_uid)

    def factory(env):
        return PushCubeDinoReward(
            env,
            encoder,
            mirror,
            camera_uid=camera_uid,
            device=device,
            shaping=cfg.dino_reward_shaping,
            reward_scale=cfg.dino_reward_scale,
        )

    return [factory]
