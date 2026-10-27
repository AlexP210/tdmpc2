"""The TD-MPC2 tasks are the ones S2P instantiates: `python tests/test_s2p_task_parity.py`.

`CUSTOM_MANISKILL_TASKS` is written out by hand, so this reads S2P's task configs (resolving their
`defaults:` inheritance the way Hydra does) and fails if a task id, control mode, camera view or
perturbation differs. No simulator is built.
"""

import copy
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from custom_maniskill_tasks import PERTURBATION_CLASSES, resolve_perturbations  # noqa: E402
from tdmpc2.envs.custom_maniskill import CUSTOM_MANISKILL_TASKS  # noqa: E402

S2P_TASKS = Path(__file__).resolve().parents[2] / "squeeze2plan" / "s2p" / "configs" / "task"
PAIRS = {  # tdmpc2 task -> S2P task config
    "push-cube": "maniskill_push_cube", "push-cube-dr": "maniskill_push_cube_dr",
    "lift-peg": "maniskill_lift_peg", "lift-peg-dr": "maniskill_lift_peg_dr",
    "pick-cube": "maniskill_pick_cube", "pick-cube-dr": "maniskill_pick_cube_dr",
    "place-sphere": "maniskill_place_sphere",
}


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def s2p_cfg(name: str) -> dict:
    """`name`'s `cfg`, with the configs it lists under `defaults:` merged underneath, `_self_` last."""
    config = yaml.safe_load((S2P_TASKS / f"{name}.yaml").read_text())
    merged = {}
    for entry in config.get("defaults", ["_self_"]):
        merged = _merge(merged, config if entry == "_self_" else {"cfg": s2p_cfg(entry)})
    return merged["cfg"]


def test_every_task_matches_its_s2p_config():
    assert set(PAIRS) == set(CUSTOM_MANISKILL_TASKS), "a task without an S2P counterpart, or the reverse"
    for task, name in PAIRS.items():
        mine, theirs = CUSTOM_MANISKILL_TASKS[task], s2p_cfg(name)
        assert mine["env"] == theirs["task_name"], (task, mine["env"], theirs["task_name"])
        assert mine["control_mode"] == theirs["control_mode"], task
        assert theirs["camera_view"] == "wrist", f"{name}: tdmpc2 always builds the wrist camera"
        s2p_perturbations = {c: theirs[c] for c in PERTURBATION_CLASSES}
        assert resolve_perturbations(mine.get("perturbations")) == resolve_perturbations(s2p_perturbations), (
            task, mine.get("perturbations"), s2p_perturbations,
        )


def test_dr_tasks_randomize_and_the_others_do_not():
    for task, entry in CUSTOM_MANISKILL_TASKS.items():
        randomized = resolve_perturbations(entry.get("perturbations")).is_randomized
        assert randomized == task.endswith("-dr"), task


def test_proprio_keys_cover_every_task():
    from tdmpc2.envs.custom_maniskill import PROPRIO_KEYS

    for task in CUSTOM_MANISKILL_TASKS:
        assert task in PROPRIO_KEYS, task
    for task in ("push-cube", "lift-peg", "pick-cube"):
        assert PROPRIO_KEYS[f"{task}-dr"] == PROPRIO_KEYS[task]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
