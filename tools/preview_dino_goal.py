"""TEMPORARY (2026-08-26): render initial vs goal frames for the DINO-reward PushCube run.

Builds the env through `tdmpc2.envs.make_env` exactly as training does, so the goal images shown
are the ones the reward is actually computed against -- not a re-derivation that could drift.

    python tools/preview_dino_goal.py [--episodes 4] [--out /tmp/dino_goal_preview.png]

Also prints the DINO distance and reward magnitudes over a random rollout, which is how
`dino_reward_scale` was picked.
"""

import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tdmpc2.envs.custom_maniskill import make_env  # noqa: E402


def upscale(frame, factor=5):
    return np.kron(frame, np.ones((factor, factor, 1), dtype=np.uint8))


def resize(frame, size):
    return np.asarray(Image.fromarray(frame).resize((size, size), Image.NEAREST))


def label(width, text, height=22):
    """A caption strip, drawn as blocky text so this needs no font file."""
    from PIL import ImageDraw

    strip = Image.new("RGB", (width, height), (24, 24, 24))
    ImageDraw.Draw(strip).text((6, 5), text, fill=(235, 235, 235))
    return np.asarray(strip)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=50)
    parser.add_argument("--out", default="/tmp/dino_goal_preview.png")
    parser.add_argument("--config", default="tdmpc2/tsd_online_training_config.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.task = "push-cube"
    cfg.obs = "rgb"
    cfg.dino_reward = True
    cfg.seed = 0

    env = make_env(cfg)
    reward_wrapper = env.get_wrapper_attr("mirror")  # resolves through the wrapper stack
    dino = env
    while not hasattr(dino, "goal_image"):
        dino = dino.env

    panels, tile = [], 300
    for episode in range(args.episodes):
        env.reset()
        maniskill = env.get_wrapper_attr("unwrapped")
        initial = dino._image(maniskill.get_obs())[0].cpu().numpy()
        goal = dino.goal_image[0].cpu().numpy()
        scene = maniskill.render()[0].cpu().numpy()

        residual = reward_wrapper.last_ik_residual
        distance = float(dino._distance(dino._encode(dino._image(maniskill.get_obs())))[0])
        print(f"episode {episode}: d_0 = {distance:.4f}   ik residual = {residual:.4f} m")

        row = np.concatenate(
            [resize(upscale(initial), tile), resize(upscale(goal), tile), resize(scene, tile)],
            axis=1,
        )
        caption = label(row.shape[1], f"episode {episode}   d_0={distance:.4f}   ik residual={residual*100:.1f} cm")
        panels.append(np.concatenate([caption, row], axis=0))

    header = label(panels[0].shape[1], "initial wrist obs (64x64)      goal wrist obs (64x64)      scene at reset", 24)
    Image.fromarray(np.concatenate([header] + panels, axis=0)).save(args.out)
    print("wrote", args.out)

    print(f"\nrandom rollout of {args.rollout_steps} steps, dino_reward_shaping="
          f"{cfg.dino_reward_shaping} scale={cfg.dino_reward_scale}:")
    env.reset()
    rewards, distances = [], []
    for _ in range(args.rollout_steps):
        _, reward, done, info = env.step(np.random.uniform(-1, 1, env.action_space.shape).astype(np.float32))
        rewards.append(float(reward))
        distances.append(float(dino._prev_distance[0]))
        if done:
            break
    rewards = np.array(rewards)
    # the discounted return is what TD-MPC2's critic has to represent, so it is the number that
    # has to sit inside vmin/vmax -- not the undiscounted sum
    frac = len(rewards) / cfg.discount_denom
    discount = min(max((frac - 1) / frac, cfg.discount_min), cfg.discount_max)
    discounted = float((rewards * discount ** np.arange(len(rewards))).sum())
    print(f"  d range       : {min(distances):.4f} .. {max(distances):.4f}")
    print(f"  reward/step   : mean {rewards.mean():+.4f}  min {rewards.min():+.4f}  max {rewards.max():+.4f}")
    print(f"  return        : undiscounted {rewards.sum():+.4f}   discounted(gamma={discount:.3f}) {discounted:+.4f}")
    print(f"  TD-MPC2 range : vmin/vmax = {cfg.vmin}/{cfg.vmax}")


if __name__ == "__main__":
    main()
