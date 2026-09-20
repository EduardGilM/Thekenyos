"""CPU checks for the kiwi-street render script. No GL, no RELIC."""
from __future__ import annotations

import importlib.util
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from treesim.config import FruitParams
from treesim.kiwi_material import STEM_LENGTH
from treesim.orchard_terrain import floor_kwargs_for_plantation, sample_orchard_floor
from treesim.pergola import generate, place_fruit

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_kiwi_street.py"
_SPEC = importlib.util.spec_from_file_location("render_kiwi_street", _SCRIPT)
street = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(street)

# RELIC Spot leg vectors (uleg->lleg and lleg->foot origins, x/z), metres.
V1 = (0.025, -0.3205)
V2 = (0.0, -0.3365)


def _small_scene():
    floor = sample_orchard_floor(
        42, canopy_height_m=street.CANOPY_Z_M, slope_deg=0.0,
        noise_m=0.0, rut_depth_m=0.0, **floor_kwargs_for_plantation(
            street.ROWS, street.COLUMNS, street.SPACING_M, margin_m=4.0),
    )
    skeleton = generate(
        height=street.CANOPY_Z_M, seed=42, rows=street.ROWS,
        columns=street.COLUMNS, spacing=street.SPACING_M,
        ground_z=floor.ground_z, canopy_z=floor.canopy_z,
    )
    return floor, skeleton


class KiwiStreetRenderTest(unittest.TestCase):
    def test_fruit_hang_matches_place_fruit(self):
        _, skeleton = _small_scene()
        fruit = place_fruit(skeleton, FruitParams(
            max_count=8, joint="free", colors=((0.39, 0.27, 0.12),),
        ), seed=42)
        self.assertTrue(fruit)
        for item in fruit:
            center = street.fruit_center(item)
            np.testing.assert_allclose(
                center,
                item.attach - np.array([0.0, 0.0, STEM_LENGTH + float(item.radii[2])]),
            )
            self.assertGreater(item.attach[2], center[2] + float(item.radii[2]) - 1e-9)

    def test_ground_soil_follows_post_rows(self):
        xs, ys, _ = street.row_layout()

        class _Floor:
            half_extent_m = 12.0

        png = street.ground_texture(_Floor(), xs, ys, seed=42, n=512)
        rgb = np.asarray(Image.open(BytesIO(png)))
        n = rgb.shape[0]
        half = 12.0

        def sample(x, y):
            j = int(round((x + half) / (2 * half) * (n - 1)))
            i = int(round((half - y) / (2 * half) * (n - 1)))
            return rgb[i, j].astype(np.float64) / 255.0

        probe_x = (-4.0, -2.0, 0.0, 2.0, 4.0)
        aisle = np.mean([sample(x, 0.0) for x in probe_x], axis=0)
        row = np.mean([sample(x, ys[1]) for x in probe_x], axis=0)
        self.assertGreater(row[0] / max(row[1], 1e-6), aisle[0] / max(aisle[1], 1e-6))
        self.assertGreater(row[0], row[1])

    def test_mjcf_uses_round_posts_leaf_meshes_and_place_fruit_hang(self):
        floor, skeleton = _small_scene()
        fruit = place_fruit(skeleton, FruitParams(
            max_count=4, joint="free", colors=((0.39, 0.27, 0.12),),
        ), seed=42)
        xs, ys, aisles = street.row_layout()
        self.assertEqual(len(aisles), street.AISLES)
        xml = street.mjcf(floor, skeleton, fruit, [], xs, ys)
        self.assertIn('type="cylinder"', xml)
        self.assertIn('material="wood"', xml)
        self.assertIn('material="concrete"', xml)
        self.assertIn('<mesh name="leaf0"', xml)
        self.assertNotIn("spotwrap", xml)
        self.assertIn("spotwrap", street.mjcf(floor, skeleton, fruit, [], xs, ys,
                                              with_spot_wrap=True))
        self.assertGreater(xml.count('type="cylinder"'), street.ROWS * street.COLUMNS)
        center = street.fruit_center(fruit[0])
        self.assertIn(f"{center[0]:.5f} {center[1]:.5f} {center[2]:.5f}", xml)
        self.assertGreaterEqual(street.CAMERA_AZIMUTH_DEG, 4.0)
        self.assertLess(street.CAMERA_AZIMUTH_DEG, 28.0)

    def test_leaf_blade_is_a_valid_single_sided_mesh(self):
        verts, uvs, faces = street.leaf_blade(0.22, 0.17)
        self.assertEqual(len(verts), len(uvs))
        self.assertEqual(faces.min(), 0)
        self.assertEqual(faces.max(), len(verts) - 1)
        self.assertTrue(np.isfinite(verts).all())
        self.assertAlmostEqual(float(verts[:, 2].max()), 0.22)
        self.assertLess(float(np.abs(verts[:, 0]).max()), 0.5 * 0.17 * 1.05)

    def test_leg_ik_recovers_forward_kinematics(self):
        hy_range, kn_range = (-0.9, 2.3), (-2.8, -0.25)
        for hy, kn in ((0.5, -1.0), (0.9, -1.6), (0.2, -0.7)):
            target = street.leg_fk(V1, V2, hy, kn)
            q = street.leg_ik(V1, V2, target, (0.5, -1.0), hy_range, kn_range)
            np.testing.assert_allclose(street.leg_fk(V1, V2, *q), target, atol=1e-4)
        with self.assertRaises(ValueError):
            street.leg_ik(V1, V2, (np.nan, 0.0), (0.5, -1.0), hy_range, kn_range)

    def test_trot_foot_offset_is_continuous_and_lifts_only_in_swing(self):
        stride, lift = 0.44, 0.09
        prev = street.trot_foot_offset(0.0, stride, lift)
        for phase in np.linspace(0.0, 1.0, 401)[1:]:
            cur = street.trot_foot_offset(phase, stride, lift)
            self.assertLess(abs(cur[0] - prev[0]), 0.02)
            self.assertLess(abs(cur[1] - prev[1]), 0.01)
            if phase < 0.5:
                self.assertEqual(cur[1], 0.0)
            else:
                self.assertGreaterEqual(cur[1], -1e-9)
            prev = cur
        self.assertAlmostEqual(street.trot_foot_offset(0.75, stride, lift)[1], lift)

    def test_stance_foot_stays_planted_in_world_frame(self):
        speed, freq = 0.72, 1.58
        stride = speed / freq
        world = []
        for t in np.linspace(0.0, 0.5 / freq, 25):
            dx, _dz = street.trot_foot_offset(freq * t, stride, 0.09)
            world.append(speed * t + dx)
        self.assertLess(np.ptp(world), 1e-9)

    def test_camera_path_stays_under_the_roof(self):
        _, _, aisles = street.row_layout()
        for t in np.linspace(0.0, 12.0, 49):
            lookat, distance, _az, elevation = street.camera_path(t, 12.0, aisles)
            height = lookat[2] + distance * np.sin(np.radians(-elevation))
            self.assertLess(height, street.CANOPY_Z_M - 0.15)
            self.assertGreater(height, 0.3)

    def test_scripted_arm_stays_under_the_beams(self):
        # Home pose: hand in front of and above the shoulder, well under the roof.
        home = street.arm_hand_height(-0.9, 1.8, -0.9)
        self.assertGreater(home, street.ARM_SHOULDER_Z_M)
        # Straight up is the geometric maximum of the chain and must exceed it.
        straight_up = street.arm_hand_height(-np.pi / 2, 0.0, 0.0)
        self.assertAlmostEqual(
            straight_up, street.ARM_SHOULDER_Z_M + street.ARM_UPPER_M[0]
            + street.ARM_FORE_M[0] + street.ARM_HAND_M[0], places=6)
        self.assertLess(street.arm_wander_max_hand_z(), street.ARM_HAND_MAX_Z_M)
        self.assertGreater(
            street.arm_wander_max_hand_z({**street.SPOT_ARM_WANDER, "arm_sh1": (-1.6, 0.4)}),
            street.ARM_HAND_MAX_Z_M)

    def test_upright_leaves_drops_hanging_shoot_leaves(self):
        class _Leaf:
            def __init__(self, frame):
                self.frame = frame

        up = _Leaf((0.0, 0.0, 0.0, 1.0))                     # +Z stays +Z
        down = _Leaf((1.0, 0.0, 0.0, 0.0))                   # 180 deg about X
        s = np.sin(np.pi / 4)
        sideways = _Leaf((0.0, s, 0.0, s))                   # +Z -> +X
        np.testing.assert_allclose(street.leaf_out_dir(up.frame), (0, 0, 1), atol=1e-12)
        np.testing.assert_allclose(street.leaf_out_dir(down.frame), (0, 0, -1), atol=1e-12)
        kept = street.upright_leaves([up, down, sideways])
        self.assertEqual(kept, [up, sideways])

    def test_walkers_stand_on_the_rendered_floor(self):
        floor, _ = _small_scene()
        xs, ys, aisles = street.row_layout()
        heights = street.street_heights(floor, xs, ys)
        ground_z = street.street_ground_z(floor, xs, ys)
        x = np.asarray(floor.x_m)
        y = np.asarray(floor.y_m)
        for i, j in ((10, 20), (heights.shape[0] // 2, heights.shape[1] // 3)):
            self.assertAlmostEqual(ground_z(float(x[j]), float(y[i])), float(heights[i, j]), places=9)
        # Wheel ruts are part of the drawn floor and must be seen by the feet.
        aisle = aisles[1]
        self.assertLess(ground_z(0.3, aisle + 0.64), ground_z(0.3, aisle) - 0.01)
        with self.assertRaises(ValueError):
            ground_z(float("nan"), 0.0)

    def test_shadow_eye_sits_upstream_of_a_camera_centred_focus(self):
        focus = street.shadow_focus((2.0, 1.0, 0.6), 10.0, 0.0, -5.0)
        np.testing.assert_allclose(focus, (5.0, 1.0, street.SHADOW_FOCUS_Z_M))
        light_dir = (-0.34, 0.58, -0.74)
        eye = street.shadow_light_pos(focus, light_dir)
        d = np.asarray(light_dir) / np.linalg.norm(light_dir)
        np.testing.assert_allclose(np.dot(focus - eye, d), street.SHADOW_BACK_M)
        self.assertGreater(eye[2], street.CANOPY_Z_M + 1.0)
        self.assertLess(street.SHADOW_BACK_M, street.SHADOW_HALF_M)
        with self.assertRaises(ValueError):
            street.shadow_light_pos(focus, (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
