"""Convert extracted 3D pose landmarks to Mixamo 52-bone RGB MoCap projects and seamless-loop FBX."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mocap import project_workspace as workspace
from mocap.bvh_bind import asset_frames, bind_pose
from mocap.mixamo_data import code_identity
from mocap.mixamo_rig import BODY_PARENTS, VERSION as MIXAMO_VERSION, fk, mannequin, skeleton, unit, write_bvh
from mocap.result_artifacts import file_stat
from mocap.source_animation import fit, policy, source_rig, validate
from mocap.storage import atomic_json
from tools.create_mixamo_sample import canonical_humanoid_rest, prepare_canonical_fbx_bind


def _orthonormal_frame(x_hint: np.ndarray, y_hint: np.ndarray, primary: str = 'x') -> np.ndarray:
    """Build a right-handed 3x3 rotation matrix from two direction hints."""
    if primary == 'x':
        x = unit(x_hint, (1.0, 0.0, 0.0))
        y = y_hint - np.sum(y_hint * x, axis=-1, keepdims=True) * x
        y = unit(y, (0.0, 1.0, 0.0))
        z = unit(np.cross(x, y), (0.0, 0.0, 1.0))
        y = unit(np.cross(z, x), (0.0, 1.0, 0.0))
    else:
        y = unit(y_hint, (0.0, 1.0, 0.0))
        x = x_hint - np.sum(x_hint * y, axis=-1, keepdims=True) * y
        x = unit(x, (1.0, 0.0, 0.0))
        z = unit(np.cross(x, y), (0.0, 0.0, 1.0))
        x = unit(np.cross(y, z), (1.0, 0.0, 0.0))
    return np.stack((x, y, z), axis=-1)


def _smooth_landmarks(pts: np.ndarray, vis: np.ndarray, sigma: float = 1.6) -> np.ndarray:
    """Visibility-weighted Gaussian smoothing along time, filling NaNs cleanly."""
    n, j, _ = pts.shape
    out = np.empty_like(pts, dtype=np.float64)
    t = np.arange(n)
    for k in range(j):
        valid = np.isfinite(pts[:, k]).all(axis=1)
        if not valid.any():
            out[:, k] = 0.0
            continue
        series = np.empty((n, 3), dtype=np.float64)
        for c in range(3):
            series[:, c] = np.interp(t, t[valid], pts[valid, k, c])
        w = np.where(valid, np.clip(vis[:, k], 0.05, 1.0), 0.02)[:, None]
        num = gaussian_filter1d(series * w, sigma=sigma, axis=0, mode='nearest')
        den = gaussian_filter1d(w, sigma=sigma, axis=0, mode='nearest')
        out[:, k] = num / np.maximum(den, 1e-6)
    return out


def _slerp_seq(r0: np.ndarray, r1: np.ndarray, alpha: np.ndarray | float) -> np.ndarray:
    alpha_arr = np.broadcast_to(alpha, (len(r0),)).astype(np.float64)
    rel = np.einsum('tji,tjk->tik', r0, r1)
    rv = Rotation.from_matrix(rel).as_rotvec()
    return r0 @ Rotation.from_rotvec(rv * alpha_arr[:, None]).as_matrix()


def _limb_frames(upper_dir: np.ndarray, lower_dir: np.ndarray, rest_u: np.ndarray, axis: np.ndarray, fallback_hint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute anatomical upper and lower limb 3x3 world frames sharing an exact hinge axis."""
    u = unit(upper_dir, rest_u)
    v = unit(lower_dir, rest_u)
    n = len(u)
    normals = np.empty((n, 3), dtype=np.float64)
    prev = np.asarray(fallback_hint, dtype=np.float64)
    for i in range(n):
        cross = np.cross(u[i], v[i])
        mag = float(np.linalg.norm(cross))
        hint = prev - u[i] * float(np.dot(prev, u[i]))
        if np.linalg.norm(hint) < 1e-6:
            hint = np.asarray(fallback_hint, dtype=np.float64) - u[i] * float(np.dot(fallback_hint, u[i]))
        if np.linalg.norm(hint) < 1e-6:
            basis = np.eye(3)[int(np.argmin(np.abs(u[i])))]
            hint = basis - u[i] * float(np.dot(basis, u[i]))
        hint = unit(hint)
        if mag > 0.12:
            cand = cross / mag
            if np.dot(cand, hint) < 0 and mag < 0.25:
                cand = hint
            w = float(np.clip((mag - 0.12) / 0.18, 0.0, 1.0))
            nrm = unit((1.0 - w) * hint + w * cand)
        else:
            nrm = hint
        normals[i] = nrm
        prev = nrm
    normals = unit(gaussian_filter1d(normals, sigma=2.0, axis=0, mode='nearest'))

    def build(d: np.ndarray) -> np.ndarray:
        d = unit(d)
        nrm = unit(normals - d * np.sum(normals * d, axis=-1, keepdims=True))
        if np.allclose(rest_u, [1, 0, 0]):
            # Left arm: rest +X, hinge (0, -1, 0) -> cols: (d, -nrm, d x (-nrm))
            y = -nrm
            z = unit(np.cross(d, y))
            return np.stack((d, y, z), axis=-1)
        if np.allclose(rest_u, [-1, 0, 0]):
            # Right arm: rest -X, hinge (0, +1, 0) -> cols: (-d, nrm, (-d) x nrm)
            x = -d
            z = unit(np.cross(x, nrm))
            return np.stack((x, nrm, z), axis=-1)
        # Legs: rest -Y, hinge (+1, 0, 0) -> cols: (nrm, -d, nrm x (-d))
        y = -d
        z = unit(np.cross(nrm, y))
        return np.stack((nrm, y, z), axis=-1)

    return build(u), build(v)


def _stabilize_seated_lower_body(mp_pts: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """Anchor occluded seated lower legs toward well-observed seated frames."""
    pts = mp_pts.copy()
    leg_vis = vis[:, [25, 26, 27, 28]].mean(axis=1)
    good = leg_vis >= np.quantile(leg_vis, 0.75)
    if not good.any():
        return pts
    for j in (25, 26, 27, 28, 29, 30, 31, 32):
        ref = np.median(pts[good, j], axis=0)
        w = np.clip((vis[:, j] - 0.35) / 0.5, 0.0, 1.0)[:, None]
        # Keep seated legs heavily damped toward the clean seated pose
        w = 0.25 * w
        pts[:, j] = w * pts[:, j] + (1.0 - w) * ref[None, :]
        pts[:, j] = gaussian_filter1d(pts[:, j], sigma=4.0, axis=0, mode='nearest')
    return pts


def landmarks_to_mixamo(npz_path: Path, rig: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert MediaPipe pose_world landmarks into Mixamo 52-bone (roots, local_52, fps)."""
    data = np.load(npz_path)
    fps = float(data['fps'])
    raw_pw = data['pose_world'].astype(np.float64)
    vis = data['pose_vis'].astype(np.float64)

    # Convert MediaPipe world (X=char-left, Y=down, Z=away) -> Mixamo world (X=char-left, Y=up, Z=forward)
    pw = raw_pw.copy()
    pw[..., 1] *= -1.0
    pw[..., 2] *= -1.0

    # Trim cold-start MediaPipe video convergence if frame 0 differs strongly, or stabilize first 15 frames
    pts = _smooth_landmarks(pw, vis, sigma=1.8)
    pts = _stabilize_seated_lower_body(pts, vis)
    n = len(pts)

    world_22 = np.broadcast_to(np.eye(3, dtype=np.float64), (n, 22, 3, 3)).copy()

    mid_hip = 0.5 * (pts[:, 23] + pts[:, 24])
    mid_sh = 0.5 * (pts[:, 11] + pts[:, 12])
    hip_x = pts[:, 23] - pts[:, 24]
    sh_x = pts[:, 11] - pts[:, 12]
    torso_y = mid_sh - mid_hip

    # Seated pelvis & torso frames
    pelvis_y_hint = unit(0.65 * np.array([[0.0, 1.0, 0.0]]) + 0.35 * unit(torso_y))
    r_hips = _orthonormal_frame(hip_x, pelvis_y_hint, primary='x')
    r_chest = _orthonormal_frame(sh_x, torso_y, primary='x')

    world_22[:, 0] = r_hips
    world_22[:, 3] = _slerp_seq(r_hips, r_chest, 0.35)
    world_22[:, 6] = _slerp_seq(r_hips, r_chest, 0.70)
    world_22[:, 9] = r_chest

    # Neck & head
    ear_x = pts[:, 7] - pts[:, 8]
    mid_ear = 0.5 * (pts[:, 7] + pts[:, 8])
    nose_z = unit(pts[:, 0] - mid_ear)
    head_x = unit(0.75 * unit(ear_x) + 0.25 * unit(sh_x))
    head_y = unit(np.cross(nose_z, head_x))
    r_head = _orthonormal_frame(head_x, head_y, primary='x')
    world_22[:, 12] = _slerp_seq(r_chest, r_head, 0.45)
    world_22[:, 15] = r_head

    # Legs (seated)
    l_thigh, l_shin = _limb_frames(pts[:, 25] - pts[:, 23], pts[:, 27] - pts[:, 25], np.array([0, -1, 0]), np.array([1, 0, 0]), np.array([1, 0, 0]))
    r_thigh, r_shin = _limb_frames(pts[:, 26] - pts[:, 24], pts[:, 28] - pts[:, 26], np.array([0, -1, 0]), np.array([1, 0, 0]), np.array([1, 0, 0]))
    world_22[:, 1], world_22[:, 4] = l_thigh, l_shin
    world_22[:, 2], world_22[:, 5] = r_thigh, r_shin

    # Feet & toes planted flat on the floor aligned with hip facing
    hip_fwd = r_hips[:, :, 2]
    hip_fwd_xz = unit(np.column_stack((hip_fwd[:, 0], np.zeros(n), hip_fwd[:, 2])), (0, 0, 1))
    foot_frame = np.stack((unit(np.cross(np.array([0, 1, 0]), hip_fwd_xz)), np.broadcast_to([0.0, 1.0, 0.0], (n, 3)), hip_fwd_xz), axis=-1)
    world_22[:, 7] = foot_frame
    world_22[:, 8] = foot_frame
    world_22[:, 10] = foot_frame
    world_22[:, 11] = foot_frame

    # Arms (tracked from MediaPipe shoulder->elbow->wrist)
    l_arm, l_fore = _limb_frames(pts[:, 13] - pts[:, 11], pts[:, 15] - pts[:, 13], np.array([1, 0, 0]), np.array([0, -1, 0]), np.array([0, -1, 0]))
    r_arm, r_fore = _limb_frames(pts[:, 14] - pts[:, 12], pts[:, 16] - pts[:, 14], np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, 1, 0]))
    world_22[:, 13] = _slerp_seq(r_chest, l_arm, 0.12)
    world_22[:, 14] = _slerp_seq(r_chest, r_arm, 0.12)
    world_22[:, 16], world_22[:, 18] = l_arm, l_fore
    world_22[:, 17], world_22[:, 19] = r_arm, r_fore

    # Wrists / Hands from MediaPipe index(19/20), pinky(17/18), wrist(15/16)
    l_palm_dir = unit(0.5 * (pts[:, 19] + pts[:, 17]) - pts[:, 15])
    l_across = unit(pts[:, 19] - pts[:, 17])  # index - pinky
    l_hand_y = unit(np.cross(l_across, l_palm_dir))
    world_22[:, 20] = _orthonormal_frame(l_palm_dir, l_hand_y, primary='x')

    r_palm_dir = unit(0.5 * (pts[:, 20] + pts[:, 18]) - pts[:, 16])
    r_across = unit(pts[:, 20] - pts[:, 18])
    r_hand_y = unit(np.cross(r_palm_dir, r_across))
    world_22[:, 21] = _orthonormal_frame(-r_palm_dir, r_hand_y, primary='x')

    # Seated root height so feet rest cleanly on Y=0 floor
    rest_22 = rig['rest'][:22]
    roots = np.zeros((n, 3), dtype=np.float64)
    # Subtle seated hip translation from smoothed pelvis/torso center
    roots[:, 0] = gaussian_filter1d((pts[:, 23, 0] + pts[:, 24, 0]) * 0.5, sigma=3.0)
    roots[:, 1] = 0.54
    roots[:, 2] = 0.0

    goals = np.empty((n, 22, 3), dtype=np.float64)
    goals[:, 0] = roots
    for j, p in enumerate(BODY_PARENTS):
        if p >= 0:
            goals[:, j] = goals[:, p] + np.einsum('tij,j->ti', world_22[:, p], rest_22[j] - rest_22[p])

    min_foot_y = float(np.median(goals[:, [7, 8, 10, 11], 1].min(axis=1)))
    lift = 0.02 - min_foot_y
    roots[:, 1] += lift
    goals[:, :, 1] += lift

    local_52, _, _ = fit(rig, roots, world_22, goals, fps=fps)

    # Cello hand posture: arched fingerboard grip on left hand, bow grip on right hand
    for j in range(22, 52):
        is_right = j >= 37
        finger_local = (j - 22) % 15
        is_thumb = finger_local < 3
        seg = finger_local % 3
        if not is_right:
            curl_deg = 22.0 if is_thumb else (32.0 if seg == 0 else (38.0 if seg == 1 else 26.0))
        else:
            curl_deg = 18.0 if is_thumb else (24.0 if seg == 0 else (28.0 if seg == 1 else 20.0))
        axis = np.array([0.0, 0.0, -1.0 if not is_right else 1.0]) if not is_thumb else np.array([0.0, -1.0 if not is_right else 1.0, 0.0])
        local_52[:, j] = Rotation.from_rotvec(axis * np.deg2rad(curl_deg)).as_matrix()[None, :, :]

    validate(local_52)
    return roots, local_52, fps


def find_best_loop_window(rig: dict, roots: np.ndarray, local_52: np.ndarray, fps: float, min_sec: float = 3.2) -> tuple[int, int]:
    """Find [i0, i1) after warm-up where pose and velocity match best for a seamless loop."""
    pos, _ = fk(rig, roots, local_52)
    pts = pos[:, :22]
    vel = np.gradient(pts, axis=0)
    # Focus heavily on upper body (arms/wrists/spine/head) after frame 30 (converged tracking)
    weights = np.ones((1, 22, 1), dtype=np.float64)
    weights[:, [16, 17, 18, 19, 20, 21]] = 2.5
    start_min = min(35, len(pts) // 4)
    min_len = max(int(round(min_sec * fps)), 45)
    best_score = float('inf')
    best_pair = (start_min, len(pts))
    for i0 in range(start_min, len(pts) - min_len):
        for i1 in range(i0 + min_len, len(pts) - 4):
            dp = np.linalg.norm((pts[i1] - pts[i0]) * weights[0], axis=-1).mean()
            dv = np.linalg.norm((vel[i1] - vel[i0]) * weights[0], axis=-1).mean()
            # Slight preference for longer loops when quality is comparable
            score = dp + 0.35 * dv - 0.00015 * (i1 - i0)
            if score < best_score:
                best_score = score
                best_pair = (i0, i1)
    return best_pair


def make_seamless_loop(rig: dict, roots: np.ndarray, local_52: np.ndarray, blend_frames: int = 12) -> tuple[np.ndarray, np.ndarray, dict]:
    """Crossfade and circularly smooth start/end so frame N-1 -> frame 0 is C1-continuous."""
    n = len(roots)
    b = min(blend_frames, n // 4)
    r_out = roots.copy()
    l_out = local_52.copy()

    # Geodesic raised-cosine crossfade over first/last b frames using cyclic anchor midpoint
    # Smoothly pull start (0..b-1) and end (n-b..n-1) toward their shared periodic average
    mid_root = 0.5 * (roots[0] + roots[-1])
    for k in range(b):
        # w goes from 0.5 at boundary (k=0) to 0.0 at interior (k=b-1)
        w = 0.5 * 0.5 * (1.0 + np.cos(np.pi * k / b))
        r_out[k] = (1.0 - w) * roots[k] + w * roots[n - 1 - k]
        r_out[n - 1 - k] = (1.0 - w) * roots[n - 1 - k] + w * roots[k]
        for j in range(52):
            ra = Rotation.from_matrix(local_52[k, j])
            rb = Rotation.from_matrix(local_52[n - 1 - k, j])
            delta = (ra.inv() * rb).as_rotvec()
            l_out[k, j] = (ra * Rotation.from_rotvec(w * delta)).as_matrix()
            l_out[n - 1 - k, j] = (rb * Rotation.from_rotvec(-w * delta)).as_matrix()

    # Exact endpoint closure + gentle wrap smoothing across the seam
    r_out = gaussian_filter1d(r_out, sigma=1.2, axis=0, mode='wrap')
    for j in range(52):
        rv = Rotation.from_matrix(l_out[:, j]).as_rotvec()
        # Unwrap rotvec sign continuity before wrap filtering
        for t in range(1, n):
            if np.dot(rv[t], rv[t - 1]) < 0 and np.linalg.norm(rv[t]) > 2.5:
                theta = np.linalg.norm(rv[t])
                rv[t] = rv[t] * ((theta - 2.0 * np.pi) / theta)
        rv_s = gaussian_filter1d(rv, sigma=1.2, axis=0, mode='wrap')
        l_out[:, j] = Rotation.from_rotvec(rv_s).as_matrix()

    validate(l_out)
    pos, world = fk(rig, r_out, l_out)
    pos_jump_mm = float(np.linalg.norm(pos[-1, :22] - pos[0, :22], axis=-1).max() * 1000.0)
    rel = np.swapaxes(world[-1], -1, -2) @ world[0]
    ang_jump_deg = float(np.rad2deg(Rotation.from_matrix(rel).magnitude()).max())
    metrics = {'loop_frames': n, 'seam_max_pos_jump_mm': round(pos_jump_mm, 3), 'seam_max_rot_jump_deg': round(ang_jump_deg, 3)}
    return r_out, l_out, metrics


def save_and_export(workspace_dir: Path, rig: dict, roots: np.ndarray, local_52: np.ndarray, fps: float, source_video: Path, output_fbx: Path, compatible_fbx: Path | None = None) -> None:
    """Commit RGB MoCap project and export Mixamo-standard FBX via CLI."""
    frames = len(roots)
    positions, world_52 = fk(rig, roots, local_52)
    export_binding = bind_pose(rig)
    fbx_binding = prepare_canonical_fbx_bind(rig)
    asset = mannequin(rig)

    workspace.create(workspace_dir, source_video, {'path': str(source_video), 'fps': fps, 'frames': frames}, 'exmr')
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
            'single_actor': {'name': 'Cellist', 'checked': True},
        },
    )

    stem = 'initial_rig-mocap'
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

    from mocap.cli import main as cli_main
    code = cli_main([
        'export', str(workspace_dir), '--model', 'exmr',
        '--format', 'fbx', '--fbx-profile', 'generic', '--rest-pose', 't',
        '--output', str(output_fbx), '--overwrite', '--json',
    ])
    if code != 0:
        raise RuntimeError(f'FBX export failed with exit code {code}')
    if compatible_fbx is not None:
        code2 = cli_main([
            'export', str(workspace_dir), '--model', 'exmr',
            '--format', 'fbx', '--fbx-profile', 'generic', '--rest-pose', 't',
            '--symmetric-rest', '--fbx-compatibility',
            '--output', str(compatible_fbx), '--overwrite', '--json',
        ])
        if code2 != 0:
            raise RuntimeError(f'Compatible FBX export failed with exit code {code2}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('landmarks', type=Path)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, default=ROOT / 'projects/cellist_mocap')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/cellist_mocap_mixamo.fbx')
    parser.add_argument('--loop-workspace', type=Path, default=ROOT / 'projects/cellist_mocap_loop')
    parser.add_argument('--loop-output', type=Path, default=ROOT / 'artifacts/cellist_mocap_loop_mixamo.fbx')
    parser.add_argument('--compatible', action='store_true')
    args = parser.parse_args()

    source_rest = canonical_humanoid_rest()
    # Slight female stature scaling (~1.65 m)
    source_rest *= 0.96
    rig = source_rig(skeleton(source_rest, source_anchors=False), source_rest)

    roots, local_52, fps = landmarks_to_mixamo(args.landmarks, rig)
    # Trim initial 20 frames if warm-up moved rapidly? Check warmed-up landmarks first:
    save_and_export(
        args.workspace, rig, roots, local_52, fps, args.video, args.output,
        compatible_fbx=args.output.with_name(args.output.stem + '_compatible.fbx') if args.compatible else None,
    )

    i0, i1 = find_best_loop_window(rig, roots, local_52, fps)
    loop_roots, loop_local, metrics = make_seamless_loop(rig, roots[i0:i1], local_52[i0:i1])
    save_and_export(
        args.loop_workspace, rig, loop_roots, loop_local, fps, args.video, args.loop_output,
        compatible_fbx=args.loop_output.with_name(args.loop_output.stem + '_compatible.fbx') if args.compatible else None,
    )
    print('LOOP_WINDOW:', i0, i1, 'METRICS:', metrics)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
