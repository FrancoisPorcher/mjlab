import argparse
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

# Headless HPC rendering backend (GPU offscreen via EGL).
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco

from mjlab.asset_zoo.robots.shadow_hand.shadow_hand_constants import (
  SHADOW_BIMANUAL_SCENE_XML,
  SHADOW_LEFT_SCENE_XML,
  SHADOW_RIGHT_SCENE_XML,
)

parser = argparse.ArgumentParser()
parser.add_argument(
  "--scene",
  choices=("left", "right", "bimanual"),
  default="left",
  help="Which Shadow Hand scene to render.",
)
args = parser.parse_args()

if args.scene == "left":
  scene_xml = SHADOW_LEFT_SCENE_XML
  output_name = "shadow_hand_left_mjlab_random_5s.mp4"
elif args.scene == "right":
  scene_xml = SHADOW_RIGHT_SCENE_XML
  output_name = "shadow_hand_right_mjlab_random_5s.mp4"
else:
  scene_xml = SHADOW_BIMANUAL_SCENE_XML
  output_name = "shadow_hand_bimanual_mjlab_random_5s.mp4"

model = mujoco.MjModel.from_xml_path(str(scene_xml))
data = mujoco.MjData(model)

print(f"=== Shadow Hand Scene (mjlab asset: {args.scene}) ===")
print(
  f"Bodies: {model.nbody}, Joints: {model.njnt}, Actuators: {model.nu}, Tendons: {model.ntendon}"
)
print(f"qpos dim: {model.nq}, ctrl dim: {model.nu}")

duration_sec = 5.0
fps = 30
frame_count = int(duration_sec * fps)
steps_per_frame = max(1, int(round(1.0 / (fps * model.opt.timestep))))

output_dir = Path("sample")
output_path = output_dir / output_name
output_dir.mkdir(parents=True, exist_ok=True)

renderer = mujoco.Renderer(model, height=480, width=640)
writer = imageio.get_writer(output_path, fps=fps)

rng = np.random.default_rng(0)
ctrl_min = model.actuator_ctrlrange[:, 0]
ctrl_max = model.actuator_ctrlrange[:, 1]
action_hold_steps = max(1, int(round(0.1 / model.opt.timestep)))
target_ctrl = rng.uniform(ctrl_min, ctrl_max)

sim_step = 0
for _ in range(frame_count):
  for _ in range(steps_per_frame):
    if sim_step % action_hold_steps == 0:
      target_ctrl = rng.uniform(ctrl_min, ctrl_max)
    data.ctrl[:] = target_ctrl
    mujoco.mj_step(model, data)
    sim_step += 1
  renderer.update_scene(data)
  frame = renderer.render()
  writer.append_data(frame)

writer.close()
renderer.close()
print(f"Saved video to {output_path}")
