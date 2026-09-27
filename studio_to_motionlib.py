#!/usr/bin/env python3
"""
Ultimate Bots Studio "SONIC MOTION" export  ->  SONIC motion_lib PKL.

Target schema (verified against sample_data/robot_filtered/*.pkl):
    <motion_name>/
        root_trans_offset  (T, 3)      float32
        root_rot           (T, 4)      float32   quaternion, XYZW (scipy convention)
        dof                (T, 29)     float32   IsaacLab joint order
        pose_aa            (T, 30, 3)  float32   root + 29 joints, MuJoCo order
        fps                int
    smpl_joints is optional: motion_lib guards it with `if "smpl_joints" in curr_motion`.

Usage:
    python studio_to_motionlib.py --compare sample_data/robot_filtered/210531/<file>.pkl
    python studio_to_motionlib.py --input /workspace/uzbek_raw/flat \
                                  --output /workspace/gr00t/data/motion_lib_custom/robot \
                                  --repeat 15
"""

import argparse
import glob
import os
import re
import sys

import joblib
import numpy as np
from scipy.spatial import transform

NUM_DOF = 29
NUM_BODIES = 30  # pelvis + 29 actuated links

# MJ_TO_IL[mj] = il   (verbatim from convert_soma_csv_to_motion_lib.py)
MJ_TO_IL = np.array(
    [0, 3, 6, 9, 13, 17,
     1, 4, 7, 10, 14, 18,
     2, 5, 8, 11, 15, 19,
     21, 23, 25, 27,
     12, 16, 20, 22, 24, 26, 28],
    dtype=np.int32,
)

# Per-DOF rotation axis, MuJoCo/MJCF actuator order
DOF_AXIS = np.array(
    [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0],
     [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0],
     [0, 0, 1], [1, 0, 0], [0, 1, 0],
     [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
     [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
    dtype=np.float32,
)

assert len(MJ_TO_IL) == NUM_DOF and DOF_AXIS.shape == (NUM_DOF, 3)


def read_csv(path):
    return np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32, ndmin=2)


def parse_fps(folder):
    p = os.path.join(folder, "info.txt")
    if not os.path.exists(p):
        return 50
    m = re.search(r"target_fps:\s*(\d+)", open(p).read())
    return int(m.group(1)) if m else 50


def build_motion(folder, il_to_mj=True):
    joint_pos = read_csv(os.path.join(folder, "joint_pos.csv"))   # (T, 29) IsaacLab order
    body_pos = read_csv(os.path.join(folder, "body_pos.csv"))     # (T, 3)  root xyz
    body_quat = read_csv(os.path.join(folder, "body_quat.csv"))   # (T, 4)  root WXYZ

    T = joint_pos.shape[0]
    if joint_pos.shape[1] != NUM_DOF:
        raise ValueError(f"expected {NUM_DOF} joints, got {joint_pos.shape[1]}")
    if body_pos.shape != (T, 3):
        raise ValueError(f"body_pos shape {body_pos.shape}, expected ({T}, 3)")
    if body_quat.shape != (T, 4):
        raise ValueError(f"body_quat shape {body_quat.shape}, expected ({T}, 4)")

    # root rotation: WXYZ (Studio / Isaac) -> XYZW (scipy, what motion_lib stores)
    root_rot = np.concatenate([body_quat[:, 1:4], body_quat[:, 0:1]], axis=1).astype(np.float32)
    norms = np.linalg.norm(root_rot, axis=1, keepdims=True)
    if np.abs(norms - 1.0).max() > 1e-3:
        print(f"  [warn] renormalizing quats (max dev {np.abs(norms - 1).max():.4f})")
    root_rot = root_rot / np.clip(norms, 1e-8, None)

    # pose_aa: (T, 30, 3) in MuJoCo order, slot 0 = root
    dof_mj = joint_pos[:, MJ_TO_IL] if il_to_mj else joint_pos
    pose_aa = np.zeros((T, NUM_BODIES, 3), dtype=np.float32)
    pose_aa[:, 0, :] = transform.Rotation.from_quat(root_rot).as_rotvec().astype(np.float32)
    pose_aa[:, 1:, :] = dof_mj[:, :, None] * DOF_AXIS[None, :, :]

    return {
        "root_trans_offset": body_pos.astype(np.float32),
        "root_rot": root_rot,
        "dof": joint_pos.astype(np.float32),
        "pose_aa": pose_aa,
        "fps": parse_fps(folder),
    }


def sanity(m):
    ok = True
    T = m["dof"].shape[0]
    z = m["root_trans_offset"][:, 2]
    print(f"  frames={T}  dur={T / m['fps']:.2f}s  fps={m['fps']}")
    print(f"  root_z [{z.min():.3f}, {z.max():.3f}] m   "
          f"dof [{m['dof'].min():+.3f}, {m['dof'].max():+.3f}] rad")
    if z.min() < 0.3 or z.max() > 1.3:
        print("  [warn] root height outside plausible G1 range"); ok = False
    if np.abs(m["dof"]).max() > 4.0:
        print("  [warn] joint magnitude looks extreme"); ok = False
    for k, v in m.items():
        if isinstance(v, np.ndarray) and not np.isfinite(v).all():
            print(f"  [warn] non-finite values in {k}"); ok = False
    return ok


def compare(path):
    d = joblib.load(path)
    for name, m in d.items():
        print(f"{name}")
        for k, v in m.items():
            if isinstance(v, np.ndarray):
                print(f"  {k:<20} {str(v.shape):<16} {v.dtype}  "
                      f"[{v.min():+.3f}, {v.max():+.3f}]")
            else:
                print(f"  {k:<20} {v}")
        break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--output")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--no-reorder", action="store_true")
    ap.add_argument("--compare")
    args = ap.parse_args()

    if args.compare:
        compare(args.compare)
        return
    if not args.input or not args.output:
        ap.error("--input and --output required (or --compare)")

    folders = sorted(
        d for d in glob.glob(os.path.join(args.input, "*"))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "joint_pos.csv"))
    )
    if not folders:
        sys.exit(f"no folders with joint_pos.csv under {args.input}")

    os.makedirs(args.output, exist_ok=True)
    print(f"Found {len(folders)} motion folder(s)\n")

    good = 0
    for folder in folders:
        name = os.path.basename(folder.rstrip("/"))
        print(f"[{name}]")
        try:
            motion = build_motion(folder, il_to_mj=not args.no_reorder)
        except Exception as e:
            print(f"  [FAIL] {e}\n")
            continue
        sanity(motion)
        for r in range(args.repeat):
            key = name if args.repeat == 1 else f"{name}_r{r:02d}"
            joblib.dump({key: motion}, os.path.join(args.output, f"{key}.pkl"))
        good += 1
        print()

    print(f"{good}/{len(folders)} converted -> {args.output}")
    print(f"wrote {good * args.repeat} PKL file(s)")


if __name__ == "__main__":
    main()
