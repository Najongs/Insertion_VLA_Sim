import h5py
import cv2
import numpy as np
import os
import argparse

# ==============================================================
# Real Dataset Viewer with MP4 Export
# Supports real-time viewing and video export with metadata overlay
# ==============================================================

# Phase mapping for display
PHASE_NAMES = {
    1: "Align",
    2: "Insert",
    3: "Hold",
    -1: "Unknown"
}

PHASE_COLORS = {
    1: (0, 255, 255),    # Yellow
    2: (0, 165, 255),    # Orange
    3: (0, 255, 0),      # Green
    -1: (128, 128, 128)  # Gray
}

def print_structure(name, obj):
    """HDF5 내부 구조를 출력하는 헬퍼 함수"""
    if isinstance(obj, h5py.Group):
        print(f"📁 [Group] {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"   💾 [Dataset] {name} | Shape: {obj.shape} | Type: {obj.dtype}")

def create_info_panel(current_step, total_steps, cur_q, cur_delta_act, cur_ee, cur_phase, width, is_playing=True):
    """
    Create metadata overlay panel.

    Args:
        current_step: Current frame index
        total_steps: Total number of frames
        cur_q: Joint positions (qpos)
        cur_delta_act: Delta action (change in EE position)
        cur_ee: End-effector pose (6D)
        cur_phase: Phase ID (1, 2, 3, or -1)
        width: Panel width
        is_playing: Whether video is playing

    Returns:
        info_panel: numpy array (H, W, 3)
    """
    info_panel = np.zeros((180, width, 3), dtype=np.uint8)

    green = (0, 255, 0)
    white = (255, 255, 255)
    yellow = (0, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Phase information
    phase_name = PHASE_NAMES.get(cur_phase, "Unknown")
    phase_color = PHASE_COLORS.get(cur_phase, (128, 128, 128))

    # Line 1: Step and Phase
    status = 'PLAY' if is_playing else 'PAUSE'
    cv2.putText(info_panel, f"Step: {current_step}/{total_steps} ({status})",
                (20, 25), font, 0.7, green, 2)
    cv2.putText(info_panel, f"Phase: {phase_name} ({cur_phase})",
                (width - 300, 25), font, 0.7, phase_color, 2)

    # Line 2: Joint positions (qpos)
    q_str = "Qpos: " + " ".join([f"{x: .3f}" for x in cur_q])
    cv2.putText(info_panel, q_str, (20, 55), font, 0.5, white, 1)

    # Line 3: Delta Action (EE position change)
    a_str = "Delta Act: " + " ".join([f"{x: .4f}" for x in cur_delta_act])
    cv2.putText(info_panel, a_str, (20, 80), font, 0.5, yellow, 1)

    # Line 4-5: End-effector pose (6D)
    if len(cur_ee) >= 6:
        # Position (XYZ)
        ee_str_pos = f"EE Pos: X={cur_ee[0]:.4f} Y={cur_ee[1]:.4f} Z={cur_ee[2]:.4f}"
        # Orientation (Roll, Pitch, Yaw in radians)
        ee_str_rot = f"EE Rot: R={cur_ee[3]:.4f} P={cur_ee[4]:.4f} Y={cur_ee[5]:.4f} (rad)"
    else:
        # Fallback for old 3D data
        ee_str_pos = f"EE Pos: X={cur_ee[0]:.4f} Y={cur_ee[1]:.4f} Z={cur_ee[2]:.4f}"
        ee_str_rot = "EE Rot: N/A (3D data only)"

    cv2.putText(info_panel, ee_str_pos, (20, 110), font, 0.5, white, 1)
    cv2.putText(info_panel, ee_str_rot, (20, 135), font, 0.5, white, 1)

    # Line 6: Controls (only for interactive mode)
    cv2.putText(info_panel, "Controls: SPACE=Pause/Play | A=Prev | D=Next | Q=Quit | S=Save MP4",
                (20, 165), font, 0.4, (200, 200, 200), 1)

    return info_panel

def save_to_mp4(hdf5_path, output_path=None, fps=30):
    """
    Save HDF5 dataset to MP4 video with metadata overlay.

    Args:
        hdf5_path: Path to HDF5 file
        output_path: Output MP4 path (default: same as input with .mp4 extension)
        fps: Video frame rate (default: 30)
    """
    if output_path is None:
        output_path = hdf5_path.replace('.h5', '.mp4')

    print(f"\n🎬 Exporting to MP4: {output_path}")

    with h5py.File(hdf5_path, 'r') as f:
        # Load data
        qpos_data = f['observations/qpos'][:]
        ee_data = f['observations/ee_pose'][:]

        # Calculate delta actions from consecutive EE poses
        delta_actions = np.zeros_like(ee_data)
        delta_actions[1:] = ee_data[1:] - ee_data[:-1]
        delta_actions[0] = np.zeros(ee_data.shape[1])  # First step has zero delta

        # Load phase data if available
        if 'phase' in f:
            phase_data = f['phase'][:]
            has_phase = True
            print("✅ Phase information found")
        else:
            phase_data = np.full(len(qpos_data), -1, dtype=np.int32)
            has_phase = False
            print("⚠️  No phase information (will show as 'Unknown')")

        # Get camera keys
        img_grp = f['observations/images']
        cam_keys = sorted(list(img_grp.keys()))

        total_steps = len(qpos_data)
        print(f"📊 Total steps: {total_steps}")
        print(f"📷 Cameras: {cam_keys}")

        # Decode first frame to get dimensions
        frames = []
        for k in cam_keys:
            binary_data = img_grp[k][0]
            img = cv2.imdecode(binary_data, cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((480, 640, 3), dtype=np.uint8)
            frames.append(img)

        # Combine frames horizontally
        combined_img = np.hstack(frames)
        h, w, _ = combined_img.shape

        # Create info panel to get final dimensions
        info_panel = create_info_panel(
            0, total_steps, qpos_data[0], delta_actions[0],
            ee_data[0], phase_data[0], w, is_playing=False
        )

        # Final video dimensions
        final_h = h + info_panel.shape[0]
        final_w = w

        print(f"📐 Video dimensions: {final_w}x{final_h}")

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (final_w, final_h))

        if not out.isOpened():
            print(f"❌ Failed to create video writer")
            return False

        # Process each frame
        print(f"🔄 Processing frames...")
        for step in range(total_steps):
            # Decode images
            frames = []
            for k in cam_keys:
                binary_data = img_grp[k][step]
                img = cv2.imdecode(binary_data, cv2.IMREAD_COLOR)

                if img is None:
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(img, "Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Add camera name overlay
                cv2.putText(img, k, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                frames.append(img)

            # Combine frames
            combined_img = np.hstack(frames)

            # Create info panel
            info_panel = create_info_panel(
                step, total_steps,
                qpos_data[step], delta_actions[step],
                ee_data[step], phase_data[step],
                w, is_playing=False
            )

            # Combine image and info panel
            final_frame = np.vstack([combined_img, info_panel])

            # Write frame
            out.write(final_frame)

            # Progress indicator
            if (step + 1) % 30 == 0 or step == total_steps - 1:
                progress = (step + 1) / total_steps * 100
                print(f"  Progress: {step + 1}/{total_steps} ({progress:.1f}%)", end='\r')

        print()  # New line after progress
        out.release()
        print(f"✅ Video saved: {output_path}")
        return True

def view_interactive(hdf5_path):
    """
    Interactive viewer for HDF5 dataset.

    Args:
        hdf5_path: Path to HDF5 file
    """
    print(f"\n👁️  Opening interactive viewer: {hdf5_path}")

    with h5py.File(hdf5_path, 'r') as f:
        # Print structure
        print("\n=== HDF5 File Structure ===")
        f.visititems(print_structure)
        print("===========================\n")

        # Load data
        qpos_data = f['observations/qpos'][:]
        ee_data = f['observations/ee_pose'][:]

        # Calculate delta actions from consecutive EE poses
        delta_actions = np.zeros_like(ee_data)
        delta_actions[1:] = ee_data[1:] - ee_data[:-1]
        delta_actions[0] = np.zeros(ee_data.shape[1])  # First step has zero delta

        # Load phase data if available
        if 'phase' in f:
            phase_data = f['phase'][:]
            has_phase = True
            print("✅ Phase information found")
        else:
            phase_data = np.full(len(qpos_data), -1, dtype=np.int32)
            has_phase = False
            print("⚠️  No phase information")

        # Get camera keys
        img_grp = f['observations/images']
        cam_keys = sorted(list(img_grp.keys()))

        total_steps = len(qpos_data)
        print(f"🎬 Total Steps: {total_steps}")
        print(f"📷 Found Cameras: {cam_keys}")

        # Create window
        cv2.namedWindow("Dataset Viewer", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Dataset Viewer", 1200, 700)

        # Create trackbar
        def nothing(x): pass
        cv2.createTrackbar("Step", "Dataset Viewer", 0, total_steps - 1, nothing)

        is_playing = True
        current_step = 0

        while True:
            # Update trackbar position
            if is_playing:
                current_step += 1
                if current_step >= total_steps:
                    current_step = 0  # Loop
                cv2.setTrackbarPos("Step", "Dataset Viewer", current_step)
            else:
                current_step = cv2.getTrackbarPos("Step", "Dataset Viewer")

            # Decode and combine images
            frames = []
            for k in cam_keys:
                binary_data = img_grp[k][current_step]
                img = cv2.imdecode(binary_data, cv2.IMREAD_COLOR)

                if img is None:
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(img, "Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Add camera name
                cv2.putText(img, k, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                frames.append(img)

            combined_img = np.hstack(frames)
            h, w, _ = combined_img.shape

            # Create info panel
            info_panel = create_info_panel(
                current_step, total_steps,
                qpos_data[current_step], delta_actions[current_step],
                ee_data[current_step], phase_data[current_step],
                w, is_playing
            )

            # Combine
            final_view = np.vstack([combined_img, info_panel])
            cv2.imshow("Dataset Viewer", final_view)

            # Key handling
            key = cv2.waitKey(33) & 0xFF  # ~30 FPS

            if key == ord('q'):  # Quit
                break
            elif key == 32:  # Spacebar: Play/Pause
                is_playing = not is_playing
            elif key == ord('a'):  # A: Previous frame
                is_playing = False
                cv2.setTrackbarPos("Step", "Dataset Viewer", max(0, current_step - 1))
            elif key == ord('d'):  # D: Next frame
                is_playing = False
                cv2.setTrackbarPos("Step", "Dataset Viewer", min(total_steps - 1, current_step + 1))
            elif key == ord('s'):  # S: Save to MP4
                is_playing = False
                output_path = hdf5_path.replace('.h5', '_export.mp4')
                print(f"\n💾 Saving video to: {output_path}")
                if save_to_mp4(hdf5_path, output_path, fps=30):
                    print("✅ Video saved successfully!")
                else:
                    print("❌ Failed to save video")

    cv2.destroyAllWindows()

def check_display_available():
    """Check if display is available for interactive viewing."""
    import os
    # Check DISPLAY environment variable
    if 'DISPLAY' not in os.environ or not os.environ['DISPLAY']:
        return False
    # Don't actually try to create window - it might hang
    # Just check if we're in a headless environment
    return False  # Default to headless/export mode for safety

def main():
    parser = argparse.ArgumentParser(
        description="Real HDF5 Dataset Viewer and MP4 Exporter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive viewer (requires display) - defaults to specific real data file
  python data_replay_real.py

  # Specify a different file
  python data_replay_real.py /path/to/another/episode.h5

  # Export to MP4 (headless mode)
  python data_replay_real.py --export

  # Export with custom output path and fps
  python data_replay_real.py --export --output /path/to/output.mp4 --fps 60
        """
    )

    parser.add_argument(
        "hdf5_path",
        type=str,
        nargs='?',
        default='/home/najo/NAS/VLA/dataset/New_dataset/collected_data/Eye_trocar/Eye_trocar/260107/2_JYT/episode_20260107_140936.h5',
        help="Path to HDF5 episode file"
    )

    parser.add_argument(
        "--export",
        action="store_true",
        help="Export to MP4 instead of interactive viewing"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output MP4 path (default: same as input with .mp4 extension)"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Video frame rate for export (default: 30)"
    )

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.hdf5_path):
        print(f"❌ File not found: {args.hdf5_path}")
        return

    # Export or view
    if args.export:
        # Explicit export mode
        save_to_mp4(args.hdf5_path, args.output, args.fps)
    else:
        # Check if display is available
        if not check_display_available():
            print("⚠️  No display detected - running in headless mode")
            print("🎬 Automatically switching to export mode...")
            save_to_mp4(args.hdf5_path, args.output, args.fps)
        else:
            view_interactive(args.hdf5_path)

if __name__ == "__main__":
    main()
