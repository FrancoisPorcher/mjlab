from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import shadow_hand_lift_cube_env_cfg
from .rl_cfg import shadow_hand_lift_cube_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Lift-Cube-Shadow-Hand",
  env_cfg=shadow_hand_lift_cube_env_cfg(),
  play_env_cfg=shadow_hand_lift_cube_env_cfg(play=True),
  rl_cfg=shadow_hand_lift_cube_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)
