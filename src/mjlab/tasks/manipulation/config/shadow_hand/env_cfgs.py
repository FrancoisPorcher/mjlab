import mujoco

from mjlab.asset_zoo.robots import get_shadow_hand_robot_cfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg, TendonLengthActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg


def get_cube_spec(cube_size: float = 0.02, mass: float = 0.05) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  body.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(cube_size,) * 3,
    mass=mass,
    rgba=(0.8, 0.2, 0.2, 1.0),
  )
  return spec


def shadow_hand_lift_cube_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_lift_cube_env_cfg()

  cfg.scene.entities = {
    "robot": get_shadow_hand_robot_cfg(),
    "cube": EntityCfg(spec_fn=get_cube_spec),
  }

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = 0.2
  joint_pos_action.actuator_names = ("lh_.*",)

  cfg.actions["tendon_len"] = TendonLengthActionCfg(
    entity_name="robot",
    actuator_names=("lh_.*J0",),
    scale=0.2,
    offset=0.0,
  )

  cfg.observations["actor"].terms["ee_to_cube"].params["asset_cfg"].site_names = (
    "grasp_site",
  )
  cfg.rewards["lift"].params["asset_cfg"].site_names = ("grasp_site",)

  # Apply friction randomization to all collision geoms of the hand.
  hand_collision_geoms = ".*"
  cfg.events["fingertip_friction_slide"].params[
    "asset_cfg"
  ].geom_names = hand_collision_geoms
  cfg.events["fingertip_friction_spin"].params[
    "asset_cfg"
  ].geom_names = hand_collision_geoms
  cfg.events["fingertip_friction_roll"].params[
    "asset_cfg"
  ].geom_names = hand_collision_geoms

  # Configure collision sensor pattern.
  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "lh_palm"

  cfg.viewer.body_name = "lh_palm"
  cfg.viewer.distance = 0.7
  cfg.viewer.elevation = -15.0
  cfg.viewer.azimuth = 120.0

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}

    assert cfg.commands is not None
    cfg.commands["lift_height"].resampling_time_range = (4.0, 4.0)

  # Shadow hand only: keep smaller number of envs by default.
  cfg.scene.num_envs = 128 if not play else 1
  cfg.scene.env_spacing = 1.5

  # Tighten workspace around hand.
  assert cfg.commands is not None
  object_pose_range = cfg.commands["lift_height"].object_pose_range
  assert object_pose_range is not None
  object_pose_range.x = (0.20, 0.26)
  object_pose_range.y = (-0.06, 0.06)
  object_pose_range.z = (0.03, 0.07)
  object_pose_range.yaw = (-3.14, 3.14)

  # Remove term that assumes all joints should stay near initial configuration.
  # For this hand, tendon-coupled motion can trigger high penalty quickly.
  cfg.rewards.pop("joint_pos_limits")

  # Ensure reward term still targets robot joints.
  cfg.rewards["joint_vel_hinge"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=("lh_.*",)
  )

  return cfg
