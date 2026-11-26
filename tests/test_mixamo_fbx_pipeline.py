"""End-to-end verification of POSIX CLI Mixamo 52-bone humanoid animation and FBX 7.4 output."""
from __future__ import annotations

from io import StringIO
from contextlib import redirect_stdout
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
import zlib

import numpy as np

from mocap.cli import main as cli_main
from mocap.mixamo_rig import BODY_NAMES, BODY_PARENTS
from tools.create_mixamo_sample import build_mixamo_project


EXPECTED_MIXAMO_52 = list(BODY_NAMES) + [
    f'{side}Hand{finger}{seg}'
    for side in ('Left', 'Right')
    for finger in ('Thumb', 'Index', 'Middle', 'Ring', 'Pinky')
    for seg in (1, 2, 3)
]


def _read_fbx_property(data: bytes, offset: int):
    code = chr(data[offset])
    offset += 1
    if code == 'C':
        return bool(struct.unpack_from('<?', data, offset)[0]), offset + 1
    if code == 'Y':
        return struct.unpack_from('<h', data, offset)[0], offset + 2
    if code == 'I':
        return struct.unpack_from('<i', data, offset)[0], offset + 4
    if code == 'L':
        return struct.unpack_from('<q', data, offset)[0], offset + 8
    if code == 'F':
        return struct.unpack_from('<f', data, offset)[0], offset + 4
    if code == 'D':
        return struct.unpack_from('<d', data, offset)[0], offset + 8
    if code in ('S', 'R'):
        length = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        raw = data[offset : offset + length]
        return (raw.decode('utf-8', errors='replace') if code == 'S' else raw), offset + length
    if code in ('f', 'd', 'l', 'i', 'b'):
        arr_len, encoding, comp_len = struct.unpack_from('<III', data, offset)
        offset += 12
        payload = data[offset : offset + comp_len]
        offset += comp_len
        if encoding == 1:
            payload = zlib.decompress(payload)
        dtype = {'f': np.float32, 'd': np.float64, 'l': np.int64, 'i': np.int32, 'b': np.uint8}[code]
        return np.frombuffer(payload, dtype=dtype, count=arr_len), offset
    raise ValueError(f'Unsupported FBX property code: {code!r}')


def _read_fbx_node(data: bytes, offset: int):
    end_offset, num_props, prop_len, name_len = struct.unpack_from('<IIIB', data, offset)
    if end_offset == 0:
        return None, offset + 13
    cursor = offset + 13
    name = data[cursor : cursor + name_len].decode('ascii')
    cursor += name_len
    props = []
    prop_end = cursor + prop_len
    for _ in range(num_props):
        val, cursor = _read_fbx_property(data, cursor)
        props.append(val)
    if cursor != prop_end:
        raise ValueError(f'FBX property byte mismatch for {name}')
    children = []
    while cursor < end_offset:
        child, cursor = _read_fbx_node(data, cursor)
        if child is None:
            break
        children.append(child)
    if cursor != end_offset:
        raise ValueError(f'FBX node end mismatch for {name}')
    return (name, props, children), end_offset


def parse_fbx_binary(path: Path):
    data = path.read_bytes()
    header = b'Kaydara FBX Binary  \x00\x1a\x00'
    if not data.startswith(header):
        raise ValueError('Invalid FBX binary header')
    version = struct.unpack_from('<I', data, len(header))[0]
    cursor = len(header) + 4
    nodes = []
    while cursor < len(data):
        node, cursor = _read_fbx_node(data, cursor)
        if node is None:
            break
        nodes.append(node)
    return version, nodes


class MixamoFbxPipelineTests(unittest.TestCase):
    def test_cli_export_produces_valid_mixamo_52_fbx_and_bvh(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = build_mixamo_project(root / 'walk_project', frames=60, fps=30.0)
            fbx_out = root / 'walk_mixamo.fbx'
            bvh_out = root / 'walk_mixamo.bvh'

            stdout = StringIO()
            with redirect_stdout(stdout):
                code = cli_main([
                    'export', str(workspace), '--model', 'exmr',
                    '--format', 'fbx', '--fbx-profile', 'generic', '--rest-pose', 't',
                    '--output', str(fbx_out), '--json',
                ])
            self.assertEqual(code, 0)
            result = json.loads(stdout.getvalue())
            self.assertTrue(result['ok'])
            self.assertEqual(Path(result['outputs'][0]), fbx_out.resolve())
            self.assertTrue(fbx_out.is_file())

            version, top_nodes = parse_fbx_binary(fbx_out)
            self.assertEqual(version, 7400)
            by_name = {name: (props, children) for name, props, children in top_nodes}

            # GlobalSettings: Y-up, cm unit scale, 30 fps
            gs_props = {}
            for cname, cprops, cchildren in by_name['GlobalSettings'][1]:
                if cname == 'Properties70':
                    for _, pprops, _ in cchildren:
                        gs_props[pprops[0]] = pprops[4:]
            self.assertEqual(gs_props['UpAxis'][0], 1)
            self.assertAlmostEqual(gs_props['UnitScaleFactor'][0], 100.0)
            self.assertAlmostEqual(gs_props['CustomFrameRate'][0], 30.0)

            # Objects: 52 Mixamo LimbNodes, BindPose, Motion AnimStack, 159 curves
            objects = by_name['Objects'][1]
            limb_models = {}
            end_models = []
            bind_poses = []
            curves = []
            anim_stacks = []
            for oname, oprops, ochildren in objects:
                if oname == 'Model':
                    label = oprops[1].split('\x00')[0]
                    kind = oprops[2]
                    if kind == 'LimbNode':
                        if label.endswith('_end'):
                            end_models.append(label)
                        else:
                            limb_models[oprops[0]] = label
                elif oname == 'Pose' and oprops[2] == 'BindPose':
                    bind_poses.append(ochildren)
                elif oname == 'AnimationStack':
                    anim_stacks.append(oprops[1].split('\x00')[0])
                elif oname == 'AnimationCurve':
                    kv = {cn: cp[0] for cn, cp, _ in ochildren if cn in ('KeyTime', 'KeyValueFloat')}
                    curves.append(kv)

            self.assertEqual(len(limb_models), 52)
            self.assertEqual(list(limb_models.values()), EXPECTED_MIXAMO_52)
            self.assertEqual(len(end_models), 13)  # Head, L/R ToeBase, 10 finger tips
            self.assertEqual(anim_stacks, ['Motion'])
            self.assertEqual(len(bind_poses), 1)
            pose_nodes = [c for c in bind_poses[0] if c[0] == 'PoseNode']
            self.assertEqual(len(pose_nodes), 53)  # SourceAnimation null + 52 LimbNodes
            self.assertEqual(len(curves), 53 * 3)  # Hips T + 52 bones R

            for kv in curves:
                self.assertEqual(len(kv['KeyTime']), 60)
                self.assertEqual(len(kv['KeyValueFloat']), 60)
                self.assertTrue(np.isfinite(kv['KeyValueFloat']).all())

            # Root translation and limb rotations actually animate
            hips_tx = curves[0]['KeyValueFloat']
            hips_tz = curves[2]['KeyValueFloat']
            self.assertGreater(float(hips_tz[-1] - hips_tz[0]), 1.0)
            self.assertGreater(float(hips_tx.max() - hips_tx.min()), 0.01)

            # Verify parent connections preserve BODY_PARENTS hierarchy
            connections = [(p[1], p[2]) for _, p, _ in by_name['Connections'][1] if p[0] == 'OO']
            id_by_name = {v: k for k, v in limb_models.items()}
            for j, parent_idx in enumerate(BODY_PARENTS):
                expected_parent = 900 if parent_idx < 0 else id_by_name[EXPECTED_MIXAMO_52[parent_idx]]
                self.assertIn((id_by_name[EXPECTED_MIXAMO_52[j]], expected_parent), connections)

            # Also verify BVH export works via CLI on the same workspace
            stdout_bvh = StringIO()
            with redirect_stdout(stdout_bvh):
                code_bvh = cli_main([
                    'export', str(workspace), '--model', 'exmr',
                    '--format', 'bvh', '--output', str(bvh_out), '--json',
                ])
            self.assertEqual(code_bvh, 0)
            self.assertTrue(bvh_out.is_file())
            self.assertIn('ROOT Hips', bvh_out.read_text(encoding='ascii'))


if __name__ == '__main__':
    unittest.main()
