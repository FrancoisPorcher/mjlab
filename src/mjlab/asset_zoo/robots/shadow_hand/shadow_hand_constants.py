"""Shadow Hand constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import XmlPositionActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets

##
# MJCF and assets.
##

SHADOW_HAND_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "shadow_hand" / "xmls" / "left_hand.xml"
)
assert SHADOW_HAND_XML.exists()

SHADOW_LEFT_HAND_XML: Path = SHADOW_HAND_XML
assert SHADOW_LEFT_HAND_XML.exists()

SHADOW_RIGHT_HAND_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "shadow_hand" / "xmls" / "right_hand.xml"
)
assert SHADOW_RIGHT_HAND_XML.exists()

SHADOW_LEFT_SCENE_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "shadow_hand" / "xmls" / "scene_left.xml"
)
assert SHADOW_LEFT_SCENE_XML.exists()

SHADOW_RIGHT_SCENE_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "shadow_hand" / "xmls" / "scene_right.xml"
)
assert SHADOW_RIGHT_SCENE_XML.exists()

SHADOW_BIMANUAL_SCENE_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "shadow_hand"
  / "xmls"
  / "scene_bimanual_side_by_side.xml"
)
assert SHADOW_BIMANUAL_SCENE_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(
    assets, SHADOW_HAND_XML.parent / "assets", meshdir, glob="*", recursive=True
  )
  return assets


def _add_grasp_site(spec: mujoco.MjSpec) -> None:
  palm_body = None
  for body in spec.bodies:
    if body.name == "lh_palm":
      palm_body = body
      break
  if palm_body is None:
    raise ValueError("Expected body 'lh_palm' was not found in shadow hand spec.")
  palm_body.add_site(
    name="grasp_site",
    pos=(0.01, 0.0, 0.07),
    size=(0.006, 0.006, 0.006),
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    rgba=(0.2, 0.8, 0.2, 1.0),
  )


def _add_right_grasp_site(spec: mujoco.MjSpec) -> None:
  palm_body = None
  for body in spec.bodies:
    if body.name == "rh_palm":
      palm_body = body
      break
  if palm_body is None:
    raise ValueError("Expected body 'rh_palm' was not found in right hand spec.")
  palm_body.add_site(
    name="grasp_site",
    pos=(-0.01, 0.0, 0.07),
    size=(0.006, 0.006, 0.006),
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    rgba=(0.2, 0.8, 0.2, 1.0),
  )


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(SHADOW_HAND_XML))
  _add_grasp_site(spec)
  spec.assets = get_assets(spec.meshdir)
  return spec


def get_right_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(SHADOW_RIGHT_HAND_XML))
  _add_right_grasp_site(spec)
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Keyframe config.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos={
    ".*": 0.0,
  },
  joint_vel={
    ".*": 0.0,
  },
)


##
# Final config.
##

SHADOW_HAND_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    XmlPositionActuatorCfg(
      target_names_expr=("lh_.*",),
      transmission_type=TransmissionType.JOINT,
    ),
    XmlPositionActuatorCfg(
      target_names_expr=("lh_.*J0",),
      transmission_type=TransmissionType.TENDON,
    ),
  ),
  soft_joint_pos_limit_factor=0.95,
)

SHADOW_RIGHT_HAND_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    XmlPositionActuatorCfg(
      target_names_expr=("rh_.*",),
      transmission_type=TransmissionType.JOINT,
    ),
    XmlPositionActuatorCfg(
      target_names_expr=("rh_.*J0",),
      transmission_type=TransmissionType.TENDON,
    ),
  ),
  soft_joint_pos_limit_factor=0.95,
)


def get_shadow_hand_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=INIT_STATE,
    spec_fn=get_spec,
    articulation=SHADOW_HAND_ARTICULATION,
  )


def get_shadow_right_hand_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=INIT_STATE,
    spec_fn=get_right_spec,
    articulation=SHADOW_RIGHT_HAND_ARTICULATION,
  )
