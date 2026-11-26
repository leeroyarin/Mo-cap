"""Create a Mixamo-standard 52-bone humanoid animation project and export FBX via RGB MoCap CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mocap import project_workspace as workspace
from mocap.bone_overlay import reference_frame
from mocap.bvh_bind import asset_frames, bind_pose
from mocap.mixamo_data import code_identity
from mocap.mixamo_rig import BODY_PARENTS, VERSION as MIXAMO_VERSION, fk, mannequin, skeleton, write_bvh
from mocap.result_artifacts import file_stat
from mocap.source_animation import fit, policy, source_rig, validate
from mocap.storage import atomic_json


def canonical_humanoid_rest() -> np.ndarray:
    """22-joint adult humanoid T-pose rest joints in Y-up metres, facing +Z."""
    rest = np.zeros((22, 3), dtype=np.float64)
    rest[0] = [0.00, 0.96, 0.00]      # Hips
    rest[1] = [0.095, 0.92, 0.00]     # LeftUpLeg
    rest[2] = [-0.095, 0.92, 0.00]    # RightUpLeg
    rest[3] = [0.00, 1.08, 0.00]      # Spine
    rest[4] = [0.095, 0.50, 0.00]     # LeftLeg
    rest[5] = [-0.095, 0.50, 0.00]    # RightLeg
    rest[6] = [0.00, 1.22, 0.00]      # Spine1
    rest[7] = [0.095, 0.08, 0.00]     # LeftFoot
    rest[8] = [-0.095, 0.08, 0.00]    # RightFoot
    rest[9] = [0.00, 1.38, 0.00]      # Spine2
    rest[10] = [0.095, 0.02, 0.13]    # LeftToeBase
    rest[11] = [-0.095, 0.02, 0.13]   # RightToeBase
    rest[12] = [0.00, 1.50, 0.00]     # Neck
    rest[13] = [0.07, 1.44, 0.00]     # LeftShoulder
    rest[14] = [-0.07, 1.44, 0.00]    # RightShoulder
    rest[15] = [0.00, 1.60, 0.00]     # Head
    rest[16] = [0.18, 1.44, 0.00]     # LeftArm
    rest[17] = [-0.18, 1.44, 0.00]    # RightArm
    rest[18] = [0.46, 1.44, 0.00]     # LeftForeArm
    rest[19] = [-0.46, 1.44, 0.00]    # RightForeArm
    rest[20] = [0.71, 1.44, 0.00]     # LeftHand
    rest[21] = [-0.71, 1.44, 0.00]    # RightHand
    return rest


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def synthesize_walk_and_wave(rig: dict, frames: int = 120, fps: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """Generate a smooth humanoid walk + right-hand wave in Mixamo 52-bone local/root space."""
    t = np.arange(frames, dtype=np.float64) / fps
    cycle = 2.0 * np.pi * (t / 1.0)  # 1.0 s per full L+R gait stride

    local_rot = np.broadcast_to(np.eye(3, dtype=np.float64), (frames, 22, 3, 3)).copy()

    def euler_xyz(rx=0.0, ry=0.0, rz=0.0):
        arr = np.column_stack((
            np.broadcast_to(rx, (frames,)),
            np.broadcast_to(ry, (frames,)),
            np.broadcast_to(rz, (frames,)),
        ))
        return Rotation.from_euler('xyz', arr, degrees=True).as_matrix()

    # Pelvis: subtle forward lean, lateral sway, and yaw twist
    local_rot[:, 0] = euler_xyz(
        4.0 + 1.5 * np.cos(2.0 * cycle),
        5.5 * np.sin(cycle),
        2.5 * np.cos(cycle),
    )

    # Spine counter-rotation
    local_rot[:, 3] = euler_xyz(0.0, -2.0 * np.sin(cycle), 0.0)
    local_rot[:, 6] = euler_xyz(2.0, -2.5 * np.sin(cycle), 0.0)
    local_rot[:, 9] = euler_xyz(0.0, -2.5 * np.sin(cycle), 0.0)

    # Wave envelope (frames ~1.0s to 3.65s)
    wave_blend = _smoothstep(1.0, 1.45, t) * (1.0 - _smoothstep(3.15, 3.65, t))
    wave_osc = np.sin(2.0 * np.pi * 2.2 * (t - 1.45))

    # Head & neck: stabilize forward, gentle turn during wave
    local_rot[:, 12] = euler_xyz(0.0, -3.0 * np.sin(cycle) - 6.0 * wave_blend, 0.0)
    local_rot[:, 15] = euler_xyz(-3.0 + 2.0 * wave_blend, -4.0 * wave_blend, 0.0)

    # Legs: positive X moves -Y toward -Z (extension/knee bend); negative X swings forward (+Z)
    for leg_idx, knee_idx, foot_idx, toe_idx, phase in ((1, 4, 7, 10, 0.0), (2, 5, 8, 11, np.pi)):
        phi = cycle + phase
        hip_swing = -24.0 * np.sin(phi) - 3.0
        knee_bend = 6.0 + 48.0 * np.maximum(0.0, np.cos(phi - 0.55)) ** 1.3 + 6.0 * np.maximum(0.0, -np.sin(phi))
        knee_bend = np.clip(knee_bend, 3.0, 62.0)
        ankle_pitch = -0.35 * (hip_swing + knee_bend) + 5.0 * np.sin(phi + 0.5)
        toe_pitch = 20.0 * np.maximum(0.0, np.sin(phi - 0.6))
        local_rot[:, leg_idx] = euler_xyz(hip_swing, 0.0, 0.0)
        local_rot[:, knee_idx] = euler_xyz(knee_bend, 0.0, 0.0)
        local_rot[:, foot_idx] = euler_xyz(ankle_pitch, 0.0, 0.0)
        local_rot[:, toe_idx] = euler_xyz(toe_pitch, 0.0, 0.0)

    # Left arm: lowered from +X T-pose via R_z(-72 deg), swung in sagittal YZ plane via extrinsic R_x
    left_swing = 20.0 * np.sin(cycle + np.pi)
    local_rot[:, 13] = euler_xyz(0.0, 0.0, -4.0)
    local_rot[:, 16] = euler_xyz(-left_swing, 0.0, -72.0)
    # HINGES[2] = (16, 18, 20, (0, -1, 0)): positive rotation around -Y bends +X forearm forward (+Z)
    left_elbow = np.deg2rad(20.0 + 10.0 * np.maximum(0.0, left_swing / 20.0))
    local_rot[:, 18] = Rotation.from_rotvec(np.array([0.0, -1.0, 0.0])[None, :] * left_elbow[:, None]).as_matrix()

    # Right arm: blend between sagittal walk swing (R_z(+72)) and friendly overhead wave (R_z(-38))
    right_swing = 20.0 * np.sin(cycle)
    walk_upper = euler_xyz(-right_swing, 0.0, 72.0)
    wave_upper = euler_xyz(-22.0, 0.0, -38.0 + 6.0 * wave_osc)
    upper_slerp = []
    for k in range(frames):
        w = float(wave_blend[k])
        r_walk = Rotation.from_matrix(walk_upper[k])
        r_wave = Rotation.from_matrix(wave_upper[k])
        rel = (r_walk.inv() * r_wave).as_rotvec()
        upper_slerp.append((r_walk * Rotation.from_rotvec(w * rel)).as_matrix())
    local_rot[:, 14] = euler_xyz(0.0, 0.0, 4.0 + 10.0 * wave_blend)
    local_rot[:, 17] = np.asarray(upper_slerp)

    # HINGES[3] = (17, 19, 21, (0, 1, 0)): positive rotation around +Y bends -X forearm forward/upward
    walk_elbow = 20.0 + 10.0 * np.maximum(0.0, right_swing / 20.0)
    wave_elbow = 92.0 + 22.0 * wave_osc
    right_elbow = np.deg2rad((1.0 - wave_blend) * walk_elbow + wave_blend * wave_elbow)
    local_rot[:, 19] = Rotation.from_rotvec(np.array([0.0, 1.0, 0.0])[None, :] * right_elbow[:, None]).as_matrix()
    local_rot[:, 21] = euler_xyz(0.0, 0.0, -12.0 * wave_blend * wave_osc)

    # Build source world orientations and floor-grounded root trajectory
    world_22 = local_rot.copy()
    for j, p in enumerate(BODY_PARENTS):
        if p >= 0:
            world_22[:, j] = world_22[:, p] @ world_22[:, j]

    rest_22 = rig['rest'][:22]
    standing_hips_y = float(-rest_22[[10, 11], 1].min())  # ToeBase at Y=0 in rest
    roots = np.zeros((frames, 3), dtype=np.float64)
    roots[:, 0] = 0.022 * np.sin(cycle)
    roots[:, 1] = standing_hips_y - 0.020 * (1.0 - np.cos(2.0 * cycle))
    roots[:, 2] = 0.92 * t

    goals = np.empty((frames, 22, 3), dtype=np.float64)
    goals[:, 0] = roots
    for j, p in enumerate(BODY_PARENTS):
        if p >= 0:
            goals[:, j] = goals[:, p] + np.einsum('tij,j->ti', world_22[:, p], rest_22[j] - rest_22[p])

    # Keep lowest foot/toe joint at >= 0.02 m above Y=0 floor plane smoothly
    min_foot_y = goals[:, [7, 8, 10, 11], 1].min(axis=1)
    floor_lift = 0.02 - float(np.median(min_foot_y))
    roots[:, 1] += floor_lift
    goals[:, :, 1] += floor_lift
    under = goals[:, [7, 8, 10, 11], 1].min(axis=1) < 0.015
    if under.any():
        lift = np.maximum(0.0, 0.015 - goals[:, [7, 8, 10, 11], 1].min(axis=1))
        roots[:, 1] += lift
        goals[:, :, 1] += lift[:, None]

    local_52, _, _ = fit(rig, roots, world_22, goals, fps=fps)

    # Add relaxed curl on the 30 finger joints (open slightly during right-hand wave)
    for j in range(22, 52):
        is_right = j >= 37
        is_thumb = ((j - 22) % 15) < 3
        curl_deg = 10.0 if is_thumb else 14.0
        if is_right:
            curl_deg = curl_deg * (1.0 - 0.7 * wave_blend)
        else:
            curl_deg = np.full(frames, curl_deg)
        axis = np.array([0.0, 0.0, -1.0 if not is_right else 1.0]) if not is_thumb else np.array([0.0, -1.0 if not is_right else 1.0, 0.0])
        local_52[:, j] = Rotation.from_rotvec(axis[None, :] * np.deg2rad(curl_deg)[:, None]).as_matrix()

    validate(local_52)
    return roots, local_52


def prepare_canonical_fbx_bind(rig: dict) -> dict:
    """Build fbx_bind arrays for a canonical straight-limb Mixamo rig without requiring SMPL-X vertices."""
    binding = bind_pose(rig)
    old = binding['rest']
    parents = binding['parents']
    display = []
    for j in range(len(parents)):
        children = np.flatnonzero(parents == j)
        delta = old[children].mean(0) - old[j] if len(children) else binding['ends'][j]
        if np.linalg.norm(delta) < 0.001:
            delta = delta + [0.0, 0.1, 0.0]
        display.append(reference_frame(delta))
    display = np.asarray(display)
    target = old.copy()
    target -= [target[[7, 8], 0].mean(), target[[10, 11], 1].min(), target[[7, 8], 2].mean()]
    return dict(
        fbx_rest=target,
        fbx_rest_frames=display.copy(),
        fbx_display_frames=display,
        fbx_tip_local=np.einsum('nji,nj->ni', display, binding['ends']),
    )


def build_mixamo_project(workspace_dir: Path, frames: int = 120, fps: float = 30.0) -> Path:
    """Create a complete export-ready RGB MoCap workspace with a Mixamo 52-bone animation."""
    workspace_dir = Path(workspace_dir).resolve()
    source_rest = canonical_humanoid_rest()
    rig = source_rig(skeleton(source_rest, source_anchors=False), source_rest)
    roots, local_52 = synthesize_walk_and_wave(rig, frames=frames, fps=fps)
    positions, world_52 = fk(rig, roots, local_52)

    export_binding = bind_pose(rig)
    fbx_binding = prepare_canonical_fbx_bind(rig)
    asset = mannequin(rig)

    source_clip = workspace_dir / 'source_walk.mp4'
    workspace.create(workspace_dir, source_clip, {'path': str(source_clip), 'fps': fps, 'frames': frames}, 'exmr')
    model_dir = workspace_dir / 'exmr'
    model_dir.mkdir(parents=True, exist_ok=True)
    workspace.set_active(workspace_dir, 'exmr', model_dir)

    atomic_json(
        model_dir / 'project.json',
        {
            'workspace_root': '..',
            'fps': float(fps),
            'frames': int(frames),
            'start': 0.0,
            'end': float(frames / fps),
            'pipeline': {'model': 'exmr', 'result': 'initial'},
            'single_actor': {'name': 'Person 1', 'checked': True},
        },
    )

    stem = 'initial_rig-sample'
    bvh_name = stem + '.bvh'
    asset_name = stem + '-asset.npz'
    frames_name = stem + '.npy'
    joints_name = stem + '-joints.npy'

    write_bvh(model_dir / bvh_name, rig, roots, local_52, fps)
    export_probes = np.einsum(
        'vij,vj->vi',
        export_binding['basis'][asset['probe_bones']].swapaxes(-1, -2),
        asset['probe_local'],
    )
    np.savez(
        model_dir / asset_name,
        packed=asset_frames(asset['packed'], export_binding),
        faces=asset['faces'],
        vertex_count=np.array(asset['vertex_count']),
        rest=export_binding['rest'],
        parents=rig['parents'],
        ends=export_binding['ends'],
        export_basis=export_binding['basis'],
        anatomical_calibration=rig['alignment'],
        probe_local=export_probes,
        probe_bones=asset['probe_bones'],
        sole_ids=asset['sole_ids'],
        sole_groups=asset['sole_groups'],
        source_regressor=np.empty((0, 0), np.float32),
        **fbx_binding,
    )

    frame_mats = np.broadcast_to(np.eye(4, dtype=np.float32), (frames, 52, 4, 4)).copy()
    frame_mats[:, :, :3, :3] = (world_52 @ export_binding['basis']).astype(np.float32)
    frame_mats[:, :, :3, 3] = positions.astype(np.float32)
    np.save(model_dir / frames_name, frame_mats)
    np.save(model_dir / joints_name, positions[:, :22].astype(np.float32))

    files = {
        'frames_file': frames_name,
        'joints_file': joints_name,
        'asset_file': asset_name,
        'bvh_file': bvh_name,
    }
    identity = code_identity()
    payload = {
        'version': MIXAMO_VERSION,
        'kind': 'initial',
        'frames': int(frames),
        'fps': float(fps),
        'names': rig['names'],
        'sources': {},
        **files,
        'representation': 'smplx',
        'source_model': 'exmr',
        'code': identity,
        'code_stats': {name: file_stat(ROOT / 'mocap' / name) for name in identity},
        **policy(fps),
        'geometry': '52-bone humanoid hierarchy including 30 finger channels',
        'outputs': {name: file_stat(model_dir / name) for name in files.values()},
    }
    atomic_json(model_dir / 'initial_mixamo.json', payload)
    return workspace_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=ROOT / 'projects/humanoid_walk')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/humanoid_walk_mixamo.fbx')
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--prepend-tpose', action='store_true')
    args = parser.parse_args()

    workspace_dir = build_mixamo_project(args.workspace, frames=args.frames, fps=args.fps)
    from mocap.cli import main as cli_main

    cli_args = [
        'export',
        str(workspace_dir),
        '--model',
        'exmr',
        '--format',
        'fbx',
        '--fbx-profile',
        'generic',
        '--rest-pose',
        't',
        '--output',
        str(args.output),
        '--overwrite',
        '--json',
    ]
    if args.prepend_tpose:
        cli_args.append('--prepend-tpose')
    return cli_main(cli_args)


if __name__ == '__main__':
    raise SystemExit(main())
