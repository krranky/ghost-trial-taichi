# SONIC G1 — Tai Chi Form

Fine-tuning the [GEAR-SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl)
whole-body controller to perform a 30-second tai chi form on the Unitree G1 (29 DOF).

Ghost Trial hackathon (Ultimate Bots) — Martial Arts track.

- **Policy (ONNX):** https://huggingface.co/iamazim/sonic-g1-taichi
- **Dataset:** https://huggingface.co/datasets/iamazim/taichi-g1

---

## Result

On the full 30-second sequence (1,537 frames at 50 Hz):

| | Stock SONIC | Fine-tuned |
|---|---|---|
| Frames completed | 31 / 1537 | **1536 / 1537** |
| Progress rate | 0.020 | **1.000** |
| Success rate | 0.0 | **1.0** |

Stock SONIC falls about 0.6 seconds into the form. The fine-tuned policy completes it.

Across 64 evaluation rollouts on the training segments (300 steps each):

| | Stock SONIC | Fine-tuned |
|---|---|---|
| Environments terminated | 22 / 64 | **0 / 64** |
| Survival | 65.6% | **100%** |

Fine-tuned tracking error on the full sequence:

| Metric | Value |
|---|---|
| mpjpe_l (local per-joint) | 35.2 mm |
| mpjpe_pa (Procrustes-aligned) | 24.4 mm |
| Upper body, local | 22.2 mm |
| Head + wrists, local | 25.6 mm |
| Legs, local | 46.8 mm |
| Feet, local | 62.1 mm |
| mpjpe_g (global) | 323.9 mm |

At the end of training, all four documented convergence thresholds were met:
`tracking_vr_5point_local` 1.00 (> 0.80), `tracking_relative_body_pos` 0.54 (> 0.44),
`tracking_anchor_pos` 0.19 (> 0.14), `time_out` 0.95 (> 0.90).

---

## Why it's hard

Tai chi is whole-body and continuous: slow weight transfers, deep stances (the root
drops to 0.54 m), and coordinated arm sweeps that flow into each other with no reset.
It loads the legs through constant balance transitions, where an upper-body gesture
would not.

It also exposes a structural problem. SONIC episodes terminate on tracking failure,
so a policy trained against one 30-second reference only ever experiences its opening
moments — it never survives long enough to reach the later movements.

---

## Method

1. **Motion capture.** A tai chi video was lifted to G1 motion with Ultimate Bots
   Studio, producing a 1,537-frame reference at 50 Hz in the SONIC deployment format
   (root pose + 29 joint angles, IsaacLab joint order, WXYZ quaternions).

2. **Segmentation** (`split_motion.py`). The sequence was cut into six overlapping
   6-second windows with 1 second of overlap. Each transition between movements
   therefore appears in two segments rather than being severed at a boundary.

3. **Conversion** (`studio_to_motionlib.py`). Each segment was converted to SONIC's
   `motion_lib` PKL format: quaternions reordered WXYZ → XYZW, and `pose_aa` derived
   from the root rotation plus per-joint rotation axes with IsaacLab → MuJoCo joint
   reordering.

4. **Mixing.** Each segment was repeated 8 times (48 entries) and mixed with base G1
   locomotion clips, so the fine-tune preserves fundamentals rather than overfitting
   to one motion.

5. **Fine-tuning.** 16,000 PPO iterations from the released SONIC checkpoint,
   `num_envs=2048`, single RTX 4090, stopped once tracking error plateaued.

Segments are a training device only. SONIC is a tracking policy, so at inference it
receives the complete 30-second reference.

---

## Honest caveats

- **Global drift.** `mpjpe_g` reaches 324 mm over 30 seconds. Local pose is accurate,
  but the robot drifts from where the reference places it, and drift compounds on a
  sequence this long.
- **Lower body is harder.** Leg (46.8 mm) and foot (62.1 mm) error remain well above
  upper-body error (22.2 mm) — consistent with a motion whose difficulty lives there.
- **One source clip.** The dataset comes from a single performance, so the policy has
  seen one interpretation of the form at one speed.
- **Stock metrics aren't comparable.** The stock policy's tracking error is measured
  over only 31 frames before it falls, so progress rate — not mpjpe — is the fair
  comparison.

---

## Files

| File | Purpose |
|---|---|
| `split_motion.py` | Split a long Studio export into overlapping segments |
| `studio_to_motionlib.py` | Convert Studio exports to SONIC `motion_lib` PKL |
| `timeout_only.yaml` | Termination config used to render the stock failure |
| `config.yaml` | Full training configuration |
| `train_metrics.log` | Tracking metrics extracted from the training log |
| `eval_before2.txt`, `eval_after.txt` | Raw evaluation output |

---

## Reproduction

```bash
# split the reference into overlapping segments
python split_motion.py --input motion_raw/taychi --output motion_raw/taychi_split \
    --seconds 6 --overlap 1

# convert to motion_lib
python studio_to_motionlib.py --input motion_raw/taychi_split \
    --output data/motion_lib_custom/robot --repeat 8

# mix with base locomotion
cp sample_data/robot_filtered/*/*.pkl data/motion_lib_mixed/robot/
cp data/motion_lib_custom/robot/*.pkl  data/motion_lib_mixed/robot/

# fine-tune
python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    +checkpoint=sonic_release/last.pt \
    num_envs=2048 headless=True use_wandb=false \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_mixed/robot \
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered

# export ONNX
python gear_sonic/eval_agent_trl.py +checkpoint=<ckpt> +headless=True \
    ++num_envs=1 +export_onnx_only=true
```

**Environment.** Trained natively on Windows 11: Isaac Lab 2.3.2, Python 3.11,
torch 2.7.0+cu126. Windows is not an officially supported configuration for this
stack, and several compatibility fixes were needed — a Unix `resource` module stub,
enabling the URDF importer extension, pinning torch to Isaac Sim's bundled version
and `tensordict` to 0.8.3, and a warp `transform_compose` replacement. On Linux the
NGC container `nvcr.io/nvidia/isaac-lab:2.3.0` runs without these.

Motion Data by Bones Studio.
