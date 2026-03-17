# TD-MPC2 Teacher for Task-Specialization Through Distillation

Adaptation of TD-MPC2 as a Teacher model for [Task-Specialization Through Distillation](https://github.com/AlexP210/tsd).

Some notable files:
- `tdmpc2/common/world_model.py`: TD-MPC2 World Model, with dynamics ensemble for estimating epistemic uncertainty.
- `tdmpc2/tdmpc2/tdmpc2.py`: Contains the `._plan()` method which uses the World Model to run MPPI planning with epistemic uncertainty as intrinsic reward. Also contains `.update()` and `._update_independent()` which updates the World Model.