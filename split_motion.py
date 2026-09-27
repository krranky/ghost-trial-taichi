#!/usr/bin/env python3
"""
Split a long Ultimate Bots Studio "SONIC MOTION" export into shorter segments.

Why: SONIC episodes terminate on tracking failure well before a 30 s reference
finishes, so a single long clip means the policy only ever trains on its opening
seconds. Splitting gives every phase of the sequence its own training signal.

The policy still tracks the FULL sequence at inference — segments are a
training-time device, not a deployment constraint.

Input : one Studio export folder (joint_pos.csv, joint_vel.csv, body_pos.csv,
        body_quat.csv, body_lin_vel.csv, body_ang_vel.csv, info.txt, metadata.txt)
Output: one folder per segment, same format, ready for studio_to_motionlib.py

Usage:
    # 6 s segments with 1 s overlap
    python split_motion.py --input motion_raw/taychi/<clip> \
                           --output motion_raw/taychi_split \
                           --seconds 6 --overlap 1

    # explicit frame ranges (e.g. at natural movement boundaries)
    python split_motion.py --input motion_raw/taychi/<clip> \
                           --output motion_raw/taychi_split \
                           --ranges 0-300,280-620,600-950,930-1280,1260-1537
"""

import argparse
import os
import re
import shutil

CSV_FILES = [
    "joint_pos.csv",
    "joint_vel.csv",
    "body_pos.csv",
    "body_quat.csv",
    "body_lin_vel.csv",
    "body_ang_vel.csv",
]


def read_csv_lines(path):
    """Return (header_line, [data_lines]) preserving the original text."""
    with open(path) as f:
        lines = f.read().splitlines()
    return lines[0], lines[1:]


def parse_fps(folder):
    p = os.path.join(folder, "info.txt")
    if not os.path.exists(p):
        return 50
    m = re.search(r"target_fps:\s*(\d+)", open(p).read())
    return int(m.group(1)) if m else 50


def parse_body_indexes(folder):
    p = os.path.join(folder, "metadata.txt")
    if not os.path.exists(p):
        return "[ 0]"
    m = re.search(r"Body part indexes:\s*\n(\[[^\]]*\])", open(p).read())
    return m.group(1) if m else "[ 0]"


def build_ranges(total, fps, seconds, overlap):
    """Sliding windows of `seconds` with `overlap` seconds shared between them."""
    win = int(round(seconds * fps))
    step = int(round((seconds - overlap) * fps))
    if step <= 0:
        raise ValueError("overlap must be smaller than segment length")

    ranges = []
    start = 0
    while start < total:
        end = min(start + win, total)
        # drop a trailing stub shorter than half a window
        if end - start < win // 2 and ranges:
            break
        ranges.append((start, end))
        if end >= total:
            break
        start += step
    return ranges


def write_segment(src, dst, lo, hi, fps, body_idx, name):
    os.makedirs(dst, exist_ok=True)
    n = hi - lo

    for fname in CSV_FILES:
        sp = os.path.join(src, fname)
        if not os.path.exists(sp):
            print(f"  [warn] missing {fname}, skipping")
            continue
        header, data = read_csv_lines(sp)
        if len(data) < hi:
            raise ValueError(f"{fname}: only {len(data)} rows, need {hi}")
        with open(os.path.join(dst, fname), "w") as f:
            f.write(header + "\n")
            f.write("\n".join(data[lo:hi]) + "\n")

    with open(os.path.join(dst, "info.txt"), "w") as f:
        f.write(f"Motion Information: {name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"source_fps: {fps}\n")
        f.write(f"target_fps: {fps}\n")
        f.write(f"source_frames: {n}\n")
        f.write(f"target_frames: {n}\n")

    with open(os.path.join(dst, "metadata.txt"), "w") as f:
        f.write(f"Metadata for: {name}\n")
        f.write("=" * 30 + "\n\n")
        f.write("Body part indexes:\n")
        f.write(body_idx + "\n")
        f.write(f"Total timesteps: {n}\n")

    print(f"  {name}: frames {lo}-{hi} ({n} frames, {n/fps:.2f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="one Studio export folder")
    ap.add_argument("--output", required=True, help="dir to write segment folders into")
    ap.add_argument("--seconds", type=float, default=6.0, help="segment length")
    ap.add_argument("--overlap", type=float, default=1.0, help="shared seconds between segments")
    ap.add_argument("--ranges", help="explicit frame ranges, e.g. 0-300,280-620")
    args = ap.parse_args()

    src = args.input.rstrip("/\\")
    jp = os.path.join(src, "joint_pos.csv")
    if not os.path.exists(jp):
        raise SystemExit(f"no joint_pos.csv in {src}")

    _, data = read_csv_lines(jp)
    total = len(data)
    fps = parse_fps(src)
    body_idx = parse_body_indexes(src)
    base = os.path.basename(src)

    print(f"{base}: {total} frames @ {fps} Hz = {total/fps:.2f}s\n")

    if args.ranges:
        ranges = []
        for part in args.ranges.split(","):
            lo, hi = part.split("-")
            ranges.append((int(lo), min(int(hi), total)))
    else:
        ranges = build_ranges(total, fps, args.seconds, args.overlap)

    os.makedirs(args.output, exist_ok=True)
    print(f"Writing {len(ranges)} segment(s):")

    for i, (lo, hi) in enumerate(ranges):
        name = f"{base}_seg{i:02d}"
        write_segment(src, os.path.join(args.output, name), lo, hi, fps, body_idx, name)

    print(f"\nDone -> {args.output}")
    print("Next:")
    print(f"  python studio_to_motionlib.py --input {args.output} "
          f"--output data/motion_lib_custom/robot --repeat 8")


if __name__ == "__main__":
    main()
