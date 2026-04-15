"""Follow Everything — offline evaluation pipeline on CrowdBot_v2.

Usage
-----
    python run.py \\
        --processed-dir data/crowdbot \\
        --sequence      0327_rds_06   \\
        --track-id      3             \\
        [--config       configs/follow_everything.yaml] \\
        [--output       results/]     \\
        [--no-images]                 \\
        [--visualize]

The script:
  1. Loads the CrowdBot_v2 sequence (LiDAR + tracks [+ RGB/depth from bag]).
  2. Writes RGB frames to a temp dir for SAM2VideoPredictor.
  3. Runs SAM2 tracking with distance-buffer re-prompting.
  4. For each frame: EKF update → occupancy grid → FSM + planner → robot step.
  5. Prints evaluation metrics at the end.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml
from omegaconf import OmegaConf

from crowdbot.dataset import CrowdBotSequence, FrameData, list_recordings
from follow_everything.control.behavior_fsm import BehaviorFSM
from follow_everything.estimation.leader_ekf import LeaderEKF
from follow_everything.mapping.occupancy_grid import OccupancyGrid
from follow_everything.perception.sam2_tracker import SAM2Tracker
from follow_everything.planning.topological_graph import TopologicalGraphPlanner
from follow_everything.simulation.robot import DiffDriveRobot
from follow_everything.utils.geometry import (
    bbox_from_3d_in_image, rotation_matrix_2d, world_to_robot,
)
from evaluation.metrics import MetricsAccumulator

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.patches as patches
from io import BytesIO
from PIL import Image


# ---------------------------------------------------------------------------
def build_components(cfg):
    ekf = LeaderEKF(
        dt=cfg.ekf.dt,
        process_noise_std=cfg.ekf.process_noise_std,
        measurement_noise_std=cfg.ekf.measurement_noise_std,
        initial_uncertainty=cfg.ekf.initial_uncertainty,
    )
    occ = OccupancyGrid(
        resolution=cfg.occupancy_grid.resolution,
        map_size=cfg.occupancy_grid.map_size,
        inflation_radius=cfg.occupancy_grid.inflation_radius,
        min_height=cfg.occupancy_grid.min_lidar_height,
        max_height=cfg.occupancy_grid.max_lidar_height,
    )
    planner = TopologicalGraphPlanner(
        robot_radius=cfg.planning.robot_radius,
        goal_tolerance=cfg.planning.goal_tolerance,
        min_boundary_spacing=cfg.planning.min_boundary_spacing,
        los_check_resolution=cfg.planning.los_check_resolution,
        max_candidates=cfg.planning.max_candidates,
    )
    fsm = BehaviorFSM(
        d_min=cfg.control.d_min,
        d_max=cfg.control.d_max,
        nis_alpha=cfg.control.nis_alpha,
        v_alpha1=cfg.control.v_alpha1,
        v_alpha2=cfg.control.v_alpha2,
        max_linear_vel=cfg.control.max_linear_vel,
        max_angular_vel=cfg.control.max_angular_vel,
        k_angular=cfg.control.k_angular,
        lookahead=cfg.control.lookahead,
        search_timeout_s=cfg.control.search_timeout_s,
    )
    robot = DiffDriveRobot(
        dt=cfg.robot.dt,
        max_linear_vel=cfg.robot.max_linear_vel,
        max_angular_vel=cfg.robot.max_angular_vel,
    )
    tracker = SAM2Tracker(cfg.perception, cfg.sam2)
    metrics = MetricsAccumulator(
        fov_half_angle_deg=cfg.perception.fov_half_angle_deg
    )
    return ekf, occ, planner, fsm, robot, tracker, metrics


# ---------------------------------------------------------------------------
def get_initial_bbox(
    seq: CrowdBotSequence,
    frame_idx: int,
    track_id: int,
    cfg,
) -> Optional[np.ndarray]:
    """Project a 3-D track into image space to get SAM2 initialisation bbox."""
    track = seq.get_track(frame_idx, track_id)
    if track is None:
        return None

    # Camera intrinsics — use defaults from config
    fx = cfg.crowdbot.default_fx
    fy = cfg.crowdbot.default_fy
    cx = cfg.crowdbot.default_cx
    cy = cfg.crowdbot.default_cy

    # Leader 3-D centre is in robot (LiDAR) frame; approximate as camera frame
    # (assumes camera and LiDAR are approximately co-located for bbox estimation)
    t = cfg.crowdbot.lidar_to_cam_t
    cam_x = track.x - t[0]
    cam_y = track.y - t[1]
    cam_z = track.z - t[2]

    # Camera frame: z-forward, x-right, y-down — LiDAR is x-fwd, y-left, z-up
    # Approximate rotation: x_cam = x_lidar, y_cam = -z_lidar, z_cam = -y_lidar
    # (camera facing forward, tilted up slightly)
    cam_frame = np.array([cam_x, -cam_z, -cam_y])

    return bbox_from_3d_in_image(
        cx_3d=cam_frame[0], cy_3d=cam_frame[1], cz_3d=cam_frame[2],
        length=track.length, width=track.width, height=track.height,
        fx=fx, fy=fy, cx_img=cx, cy_img=cy,
        img_w=int(2 * cx), img_h=int(2 * cy),
    )


# ---------------------------------------------------------------------------
def run_sequence(
    seq: CrowdBotSequence,
    track_id: int,
    cfg,
    frames: List[FrameData],
    output_dir: Optional[Path],
    visualize: bool,
) -> None:
    ekf, occ, planner, fsm, robot, tracker, metrics = build_components(cfg)

    # --- Collect images for SAM2 -----------------------------------------
    images = [f.image for f in frames if f.image is not None]
    depth_seq = [f.depth for f in frames]
    has_images = len(images) == len(frames)

    if not has_images:
        print(
            f"[WARNING] Only {len(images)}/{len(frames)} frames have RGB images.\n"
            "          SAM2 tracking requires images. Check rosbag topics in config."
        )

    sam2_results = {}
    if has_images:
        with tempfile.TemporaryDirectory(prefix="fe_frames_") as tmp:
            frames_dir = SAM2Tracker.write_frames(images, tmp)

            # Initialisation bbox from first annotated frame
            start_frame = seq.first_frame_with_track(track_id) or 0
            init_bbox = get_initial_bbox(seq, start_frame, track_id, cfg)
            if init_bbox is None:
                print(f"[ERROR] Track ID {track_id} not found in sequence.")
                return

            print(f"Initialising SAM2 with bbox {init_bbox} at frame {start_frame}")
            sam2_results = tracker.track_sequence(
                frames_dir=frames_dir,
                initial_bbox=init_bbox,
                depth_seq=depth_seq,
            )

    # --- Initialise simulated robot at the real robot's first pose -------
    robot.reset(pose=frames[0].robot_pose.copy())

    # --- Per-frame simulation loop ---------------------------------------
    print(f"Running simulation over {len(frames)} frames ...")
    for idx, fd in enumerate(frames):
        real_pos  = fd.robot_pose[:2]          # real robot world XY at this frame
        real_yaw  = float(fd.robot_pose[2])    # real robot yaw at this frame
        R_real    = rotation_matrix_2d(real_yaw)

        # 1. Perception — leader position in world frame
        sam2 = sam2_results.get(idx)
        is_visible = sam2.is_visible if sam2 is not None else False
        leader_world_pos: Optional[np.ndarray] = None

        if is_visible and sam2.centroid_uv is not None and fd.depth is not None:
            u, v = sam2.centroid_uv
            u_i, v_i = int(u), int(v)
            h, w = fd.depth.shape
            if 0 <= v_i < h and 0 <= u_i < w:
                d = float(fd.depth[v_i, u_i])
                if d > 0.1:
                    fx, fy = cfg.crowdbot.default_fx, cfg.crowdbot.default_fy
                    cx, cy = cfg.crowdbot.default_cx, cfg.crowdbot.default_cy
                    x_lidar = d * (u - cx) / fx
                    y_lidar = d * (v - cy) / fy
                    leader_lidar = np.array([d, -x_lidar])
                    leader_world_pos = real_pos + R_real @ leader_lidar

        # Fallback: use GT track position (LiDAR frame → world frame)
        if leader_world_pos is None:
            gt = seq.get_track(fd.frame_idx, track_id)
            if gt is not None:
                leader_lidar = np.array([gt.x, gt.y])
                leader_world_pos = real_pos + R_real @ leader_lidar
                is_visible = True

        # 2. EKF in world frame
        if leader_world_pos is not None and is_visible:
            leader_state = ekf.update(leader_world_pos)
        else:
            leader_state = ekf.predict()

        # 3. Occupancy grid (LiDAR frame, centred on real robot each frame)
        occ.update(fd.lidar)

        # 4. Plan — convert simulated robot position to LiDAR frame for planner
        sim_in_lidar = rotation_matrix_2d(-real_yaw) @ (robot.pose[:2] - real_pos)
        goal_world   = leader_state.position if leader_state else robot.pose[:2]
        goal_lidar   = rotation_matrix_2d(-real_yaw) @ (goal_world - real_pos)
        path_lidar   = planner.plan(sim_in_lidar, goal_lidar, occ)
        # Convert waypoints back to world frame for the FSM controller
        path_world   = [real_pos + R_real @ wp for wp in path_lidar]

        # 5. FSM → velocity commands (world frame)
        ctrl = fsm.step(
            robot_pose=robot.pose,
            leader_state=leader_state,
            is_visible=is_visible,
            nis=ekf.nis,
            path=path_world,
        )

        # 6. Simulate robot motion
        robot.step(ctrl.linear_vel, ctrl.angular_vel)

        # 7. Collision check: transform simulated robot to LiDAR frame
        sim_relative = rotation_matrix_2d(-real_yaw) @ (robot.pose[:2] - real_pos)
        in_collision = occ.is_occupied(sim_relative[0], sim_relative[1])

        # 8. GT leader world position for metrics
        gt_track = seq.get_track(fd.frame_idx, track_id)
        leader_world = (
            real_pos + R_real @ np.array([gt_track.x, gt_track.y])
            if gt_track else None
        )

        metrics.record(
            robot_pose=robot.pose,
            leader_world_pos=leader_world,
            is_visible=is_visible,
            in_collision=in_collision,
            linear_vel=ctrl.linear_vel,
        )

        # 9. Visualization
        if visualize and output_dir:
            viz_dir = output_dir / "viz"
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            fig, (ax_img, ax_map) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Left: RGB Image with SAM2 mask/bbox
            if fd.image is not None:
                ax_img.imshow(fd.image)
                if is_visible and sam2 is not None and sam2.mask is not None:
                    mask = sam2.mask
                    overlay = np.zeros((*mask.shape, 4))
                    overlay[mask] = [1, 0, 0, 0.4]  # Red transparent mask
                    ax_img.imshow(overlay)
                    
                    if sam2.centroid_uv:
                        ax_img.plot(sam2.centroid_uv[0], sam2.centroid_uv[1], 'rx', markersize=10)
                ax_img.set_title(f"Perception (Frame {idx})")
            ax_img.axis('off')
            
            # Right: Occupancy Grid + Robot + Leader
            # Get grid in world frame (approximate/simplified for viz)
            grid = occ.inflated_grid
            res = occ.resolution
            origin = occ.origin  # current robot pos in LiDAR frame is [0,0]
            
            ax_map.imshow(grid, extent=[origin[0], origin[0] + grid.shape[1]*res, 
                                        origin[1], origin[1] + grid.shape[0]*res],
                          origin='lower', cmap='Greys', alpha=0.3)
            
            # Robot (simulated) in relative frame
            ax_map.plot(sim_relative[0], sim_relative[1], 'bo', label='Robot')
            # Robot direction
            sim_yaw = robot.pose[2] - real_yaw
            ax_map.arrow(sim_relative[0], sim_relative[1], 
                         0.3 * np.cos(sim_yaw), 0.3 * np.sin(sim_yaw), 
                         head_width=0.1, head_length=0.1, fc='blue', ec='blue')
            
            # Leader (estimated) in relative frame
            if leader_state:
                leader_rel = rotation_matrix_2d(-real_yaw) @ (leader_state.position - real_pos)
                ax_map.plot(leader_rel[0], leader_rel[1], 'rs', label='Leader (EKF)')
                
            # Goal in relative frame
            goal_rel = rotation_matrix_2d(-real_yaw) @ (goal_world - real_pos)
            ax_map.plot(goal_rel[0], goal_rel[1], 'g*', label='Goal')
            
            # Path in relative frame
            if path_lidar:
                path_arr = np.array(path_lidar)
                ax_map.plot(path_arr[:, 0], path_arr[:, 1], 'g--', alpha=0.5)
            
            ax_map.set_title(f"Planning & Control ({ctrl.state.name})")
            ax_map.set_xlim([-5, 5])
            ax_map.set_ylim([-5, 5])
            ax_map.legend(loc='upper right', fontsize='small')
            ax_map.set_aspect('equal')
            
            plt.tight_layout()
            plt.savefig(viz_dir / f"{idx:06d}.jpg")
            plt.close(fig)

        if idx % 50 == 0:
            print(f"  frame {idx:5d}/{len(frames)} | state={ctrl.state.name:12s} "
                  f"| visible={is_visible} | d={float(np.linalg.norm(leader_state.position)):.2f}m")

    # --- Results ---------------------------------------------------------
    print("\n=== Evaluation Results ===")
    print(metrics.summary_str())

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        traj = robot.trajectory
        np.save(output_dir / "robot_trajectory.npy", traj)
        m = metrics.compute()
        import json
        result_dict = {
            "sequence": seq.sequence_name,
            "track_id": track_id,
            "follow_success": m.follow_success,
            "leader_loss_ratio": m.leader_loss_ratio,
            "collision_rate": m.collision_rate,
            "mean_follow_distance": m.mean_follow_distance,
            "trajectory_smoothness": m.trajectory_smoothness,
            "n_frames": m.n_frames,
        }
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(result_dict, f, indent=2)
        print(f"\nResults saved to {output_dir}")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Follow Everything on CrowdBot_v2")
    parser.add_argument("--processed-dir", required=True,
                        help="Day directory, e.g. data/crowdbot/0410_rds_defaced_processed")
    parser.add_argument("--sequence", default=None,
                        help="Recording name (omit to list available recordings)")
    parser.add_argument("--track-id", type=int, default=None,
                        help="Pedestrian track ID to follow (omit to list available IDs)")
    parser.add_argument("--config", default="configs/follow_everything.yaml")
    parser.add_argument("--output", default=None,
                        help="Directory to write results (optional)")
    parser.add_argument("--no-images", action="store_true",
                        help="Skip loading RGB/depth from rosbag (LiDAR+tracks only)")
    parser.add_argument("--visualize", action="store_true",
                        help="Show matplotlib visualisation (not implemented yet)")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # List recordings if none specified
    if args.sequence is None:
        recordings = list_recordings(args.processed_dir)
        if not recordings:
            print(f"No recordings found under {args.processed_dir}/lidars/")
        else:
            print(f"Available recordings in {args.processed_dir}:")
            for r in recordings:
                print(f"  {r}")
        return

    print(f"Loading recording: {args.sequence}")
    seq = CrowdBotSequence(
        day_dir=args.processed_dir,
        recording=args.sequence,
        load_images=not args.no_images,
    )
    print(f"  {len(seq)} frames loaded.")

    # List track IDs if none specified
    if args.track_id is None:
        ids = seq.all_track_ids()
        print(f"Available track IDs: {ids}")
        print("Re-run with --track-id <ID> to start simulation.")
        return

    # Eagerly collect all frames (keeps bag reader alive throughout)
    frames = [seq[i] for i in range(len(seq))]

    output_dir = Path(args.output) if args.output else None
    run_sequence(seq, args.track_id, cfg, frames, output_dir, args.visualize)


if __name__ == "__main__":
    main()
