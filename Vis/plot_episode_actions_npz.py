#!/usr/bin/env python
"""
Plot qpos, action, and ee_pose from .npz (real_to_sim recordings).
Supports overlay plotting for real vs sim on the same axes.

Units (from real_to_sim.py):
1. Action:  Linear=mm, Rotation=rad (Delta)
2. EE Pose: Linear=mm, Rotation=rad
3. Qpos:    All=degree

Usage:

python plot_episode_actions_npz.py \
    --real /home/irom/NAS/VLA/Insertion_VLA_Sim/digital_twin/recordings/real_to_sim/real_episode_20260130_031527.npz \
    --sim /home/irom/NAS/VLA/Insertion_VLA_Sim/digital_twin/recordings/real_to_sim/sim_episode_20260130_031527.npz \
    --overlay \
    --output_dir /home/irom/NAS/VLA/Insertion_VLA_Sim/Vis/outputs/action_plots
    
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

def find_episode_file(episode_input: str, base_dir: str):
    episode_path = Path(episode_input)
    if episode_path.exists():
        return str(episode_path)

    episode_name = episode_path.name
    print(f"Searching for '{episode_name}' in {base_dir}...")
    search_pattern = f"{base_dir}/**/{episode_name}"
    matches = glob.glob(search_pattern, recursive=True)

    if not matches:
        raise FileNotFoundError(f"Episode file '{episode_name}' not found in {base_dir}")
    if len(matches) > 1:
        print(f"Warning: Found {len(matches)} matching files. Using first match: {matches[0]}")
    return matches[0]


def load_episode_data(npz_path: str):
    print(f"Loading episode from: {npz_path}")
    data = np.load(npz_path)
    qpos = data["qpos"] if "qpos" in data else None
    ee_pose = data["ee_pose"] if "ee_pose" in data else None
    action = data["action"] if "action" in data else None
    timestamps = data["timestamp"] if "timestamp" in data else None
    return qpos, ee_pose, action, timestamps


def plot_data_6d(data, data_type, timestamps=None, output_dir=None, episode_name=""):
    if data is None:
        print(f"Skipping plot for {data_type} (No data found)")
        return

    num_frames = len(data)
    data_dim = data.shape[1]
    x_values = np.arange(num_frames)
    x_label = "Frame Index"

    if data_type == "action":
        dim_labels = ['dx (mm)', 'dy (mm)', 'dz (mm)', 'dRx (rad)', 'dRy (rad)', 'dRz (rad)']
        title_color = "red"
        main_title = "Action (Delta: mm / rad)"
    elif data_type == "ee_pose":
        dim_labels = ['X (mm)', 'Y (mm)', 'Z (mm)', 'Rx (rad)', 'Ry (rad)', 'Rz (rad)']
        title_color = "green"
        main_title = "EE Pose (Pos: mm / Rot: rad)"
    elif data_type == "ee_pose_delta":
        dim_labels = ['dX (mm)', 'dY (mm)', 'dZ (mm)', 'dRx (rad)', 'dRy (rad)', 'dRz (rad)']
        title_color = "darkgreen"
        main_title = "EE Pose Delta (per frame)"
    elif data_type == "qpos":
        dim_labels = [f"Joint {i+1} (deg)" for i in range(6)]
        title_color = "blue"
        main_title = "Joint Positions (Degree)"
    else:
        dim_labels = [f"Dim {i}" for i in range(6)]
        title_color = "black"
        main_title = f"{data_type} Values"

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f"{main_title} - {episode_name}", fontsize=16, color=title_color, fontweight="bold")

    for i in range(min(data_dim, 6)):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        ax.plot(x_values, data[:, i], linewidth=1.5, color=title_color)
        ax.set_xlabel(x_label)
        ax.set_ylabel(dim_labels[i])
        ax.set_title(dim_labels[i])
        ax.grid(True, alpha=0.3)

        mean_val = data[:, i].mean()
        min_val = data[:, i].min()
        max_val = data[:, i].max()
        stats_text = f"Mean: {mean_val:.2f}\nRange: [{min_val:.2f}, {max_val:.2f}]"
        ax.legend([stats_text], loc="upper right", fontsize="small", framealpha=0.8)

    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = f"{data_type}_{episode_name}.png"
        save_path = output_path / filename
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved {data_type} plot to: {save_path}")
    else:
        plt.show()

    plt.close()

def plot_data_6d_overlay(real_data, sim_data, data_type, output_dir=None, episode_name=""):
    if real_data is None and sim_data is None:
        print(f"Skipping plot for {data_type} (No data found)")
        return

    if real_data is not None:
        num_frames = len(real_data)
        data_dim = real_data.shape[1]
    else:
        num_frames = len(sim_data)
        data_dim = sim_data.shape[1]

    x_values = np.arange(num_frames)
    x_label = "Frame Index"

    if data_type == "action":
        dim_labels = ['dx (mm)', 'dy (mm)', 'dz (mm)', 'dRx (rad)', 'dRy (rad)', 'dRz (rad)']
        title_color = "red"
        main_title = "Action (Delta: mm / rad)"
    elif data_type == "ee_pose":
        dim_labels = ['X (mm)', 'Y (mm)', 'Z (mm)', 'Rx (rad)', 'Ry (rad)', 'Rz (rad)']
        title_color = "green"
        main_title = "EE Pose (Pos: mm / Rot: rad)"
    elif data_type == "ee_pose_delta":
        dim_labels = ['dX (mm)', 'dY (mm)', 'dZ (mm)', 'dRx (rad)', 'dRy (rad)', 'dRz (rad)']
        title_color = "darkgreen"
        main_title = "EE Pose Delta (per frame)"
    elif data_type == "qpos":
        dim_labels = [f"Joint {i+1} (deg)" for i in range(6)]
        title_color = "blue"
        main_title = "Joint Positions (Degree)"
    else:
        dim_labels = [f"Dim {i}" for i in range(6)]
        title_color = "black"
        main_title = f"{data_type} Values"

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f"{main_title} - {episode_name}", fontsize=16, color=title_color, fontweight="bold")

    for i in range(min(data_dim, 6)):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        if real_data is not None:
            ax.plot(x_values, real_data[:, i], linewidth=1.5, color="tab:blue", label="real")
        if sim_data is not None:
            sim_x = np.arange(len(sim_data))
            ax.plot(sim_x, sim_data[:, i], linewidth=1.5, color="tab:orange", label="sim")

        ax.set_xlabel(x_label)
        ax.set_ylabel(dim_labels[i])
        ax.set_title(dim_labels[i])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize="small", framealpha=0.8)

    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = f"{data_type}_{episode_name}_overlay.png"
        save_path = output_path / filename
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved {data_type} overlay plot to: {save_path}")
    else:
        plt.show()

    plt.close()

def compute_delta_6d(data):
    if data is None:
        return None
    delta = np.zeros_like(data)
    delta[1:] = data[1:] - data[:-1]
    return delta


def main():
    parser = argparse.ArgumentParser(description="Plot .npz episode (real_to_sim)")
    parser.add_argument("--episode", type=str, required=False, help="Path/filename of .npz")
    parser.add_argument("--real", type=str, required=False, help="Path/filename of real .npz")
    parser.add_argument("--sim", type=str, required=False, help="Path/filename of sim .npz")
    parser.add_argument("--overlay", action="store_true", help="Overlay real vs sim plots")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--base_dir", type=str, default="/home/irom/NAS/VLA/Insertion_VLA_Sim/digital_twin/recordings")
    args = parser.parse_args()

    try:
        if args.overlay:
            if not args.real or not args.sim:
                raise ValueError("Overlay mode requires --real and --sim paths.")

            real_path = find_episode_file(args.real, args.base_dir)
            sim_path = find_episode_file(args.sim, args.base_dir)
            episode_name = f"{Path(real_path).stem}_vs_{Path(sim_path).stem}"

            real_q, real_ee, real_act, _ = load_episode_data(real_path)
            sim_q, sim_ee, sim_act, _ = load_episode_data(sim_path)

            plot_data_6d_overlay(real_q, sim_q, "qpos", args.output_dir, episode_name)
            plot_data_6d_overlay(real_ee, sim_ee, "ee_pose", args.output_dir, episode_name)
            plot_data_6d_overlay(compute_delta_6d(real_ee), compute_delta_6d(sim_ee),
                                 "ee_pose_delta", args.output_dir, episode_name)
            plot_data_6d_overlay(real_act, sim_act, "action", args.output_dir, episode_name)
        else:
            if not args.episode:
                raise ValueError("Single mode requires --episode path.")
            episode_path = find_episode_file(args.episode, args.base_dir)
            episode_name = Path(episode_path).stem

            qpos, ee_pose, action, timestamps = load_episode_data(episode_path)

            plot_data_6d(qpos, "qpos", timestamps, args.output_dir, episode_name)
            plot_data_6d(ee_pose, "ee_pose", timestamps, args.output_dir, episode_name)
            plot_data_6d(compute_delta_6d(ee_pose), "ee_pose_delta", timestamps, args.output_dir, episode_name)
            plot_data_6d(action, "action", timestamps, args.output_dir, episode_name)

        print("\nAll plots generated successfully!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
