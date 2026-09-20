"""CPU checks for the beauty-frames hillside basket recording."""
import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from treesim.config import FoliageParams
from treesim.foliage import place_canopy_leaves, place_leaves
from treesim.orchard_terrain import sample_orchard_floor
from treesim.pergola import generate


class RecordBasketOrchardTest(unittest.TestCase):
    def test_scenery_has_hillside_tied_fruit_and_leaf_roof(self):
        from scripts.record_basket_orchard import HARVEST_XY, build_scenery
        with tempfile.TemporaryDirectory() as directory:
            scenery = build_scenery(
                seed=42, extra_fruit=80, rows=2, columns=2, spacing=5.0,
                slope_deg=3.5, slope_azimuth_deg=38.0, noise_m=0.03, rut_depth_m=0.05,
                landform_m=0.45, landform_wavelength_m=18.0, canopy_spacing=0.25,
                asset_dir=Path(directory))
            self.assertGreater(float(scenery.floor.heights_m.max() - scenery.floor.heights_m.min()), 0.08)
            self.assertGreater(len(scenery.extra_fruit), 20)
            self.assertGreater(len(scenery.leaves), 80)
            self.assertEqual(len(scenery.harvest_positions), 6)
            np.testing.assert_allclose(scenery.harvest_positions[:, :2], HARVEST_XY, atol=1e-9)
            for fruit in scenery.extra_fruit:
                gap = np.min(np.linalg.norm(fruit.center[:2] - HARVEST_XY, axis=1))
                self.assertGreater(gap, 0.27)
            self.assertTrue(scenery.ground_png.is_file())
            self.assertTrue(scenery.earth_png.is_file())

    def test_textured_canopy_uses_material_tiles_not_leaf_meshes(self):
        from scripts.record_basket_orchard import apply_scenery, build_scenery
        xml = '''<mujoco>
  <asset><texture name="sky" type="skybox" builtin="gradient" rgb1="0 0 0" rgb2="1 1 1"/></asset>
  <visual><global offwidth="128" offheight="128"/><headlight/></visual>
  <worldbody>
    <geom type="plane" size="4 4 0.1"/>
    <light pos="0 0 4"/>
    <body name="assisted_kiwi_0" pos="-1 -0.65 1.47"><geom name="assisted_kiwi_0" type="ellipsoid" size=".03 .03 .04"/></body>
    <site name="anchor_0" pos="-1 -0.65 1.47"/>
  </worldbody>
</mujoco>'''
        with tempfile.TemporaryDirectory() as directory:
            scenery = build_scenery(
                seed=7, extra_fruit=8, rows=2, columns=2, spacing=5.0,
                slope_deg=2.0, slope_azimuth_deg=0.0, noise_m=0.02, rut_depth_m=0.04,
                landform_m=0.3, landform_wavelength_m=18.0, canopy_spacing=0.15,
                asset_dir=Path(directory), leaf_draw='texture')
            self.assertEqual(len(scenery.leaves), 0)
            self.assertGreater(len(scenery.canopy_tiles), 3)
            self.assertTrue(scenery.canopy_png.is_file())
            root = ET.fromstring(xml)
            apply_scenery(root, scenery)
        self.assertIsNotNone(root.find("asset/material[@name='kiwi_canopy']"))
        self.assertGreater(len(root.findall("worldbody/geom[@name='canopy_tile_0']")), 0)
        self.assertEqual(len(root.findall("worldbody/geom[@mesh]")), 0)

    def test_dress_keeps_six_harvest_bodies_and_adds_hfield(self):
        from scripts.record_basket_orchard import apply_scenery, build_scenery
        xml = '''<mujoco>
  <asset><texture name="sky" type="skybox" builtin="gradient" rgb1="0 0 0" rgb2="1 1 1"/></asset>
  <visual><global offwidth="128" offheight="128"/><headlight/></visual>
  <worldbody>
    <geom type="plane" size="4 4 0.1"/>
    <geom type="capsule" fromto="-1.7 -1.6 0 -1.7 -1.6 1.65" size="0.05"/>
    <geom type="ellipsoid" pos="0 0 1.7" size="0.13 0.08 0.008" contype="0" group="2"/>
    <light pos="0 0 4"/>
    <body name="assisted_kiwi_0" pos="-1 -0.65 1.47"><geom name="assisted_kiwi_0" type="ellipsoid" size=".03 .03 .04"/></body>
    <site name="anchor_0" pos="-1 -0.65 1.47"/>
  </worldbody>
</mujoco>'''
        with tempfile.TemporaryDirectory() as directory:
            scenery = build_scenery(
                seed=7, extra_fruit=12, rows=2, columns=2, spacing=5.0,
                slope_deg=2.0, slope_azimuth_deg=0.0, noise_m=0.02, rut_depth_m=0.04,
                landform_m=0.3, landform_wavelength_m=18.0, canopy_spacing=0.0,
                asset_dir=Path(directory))
            root = ET.fromstring(xml)
            apply_scenery(root, scenery)
        world = root.find('worldbody')
        self.assertIsNotNone(world.find("geom[@name='orchard_ground']"))
        self.assertIsNotNone(world.find("geom[@name='earth_mass']"))
        kiwi = world.find("body[@name='assisted_kiwi_0']")
        pos = np.fromstring(kiwi.get('pos'), sep=' ')
        np.testing.assert_allclose(pos[:2], scenery.harvest_positions[0, :2], atol=1e-4)
        self.assertGreater(len(world.findall("geom[@type='mesh']")), 0)

    def test_dress_adds_storage_and_wrist_cameras(self):
        from scripts.record_basket_orchard import CENTER, PREFIX, SIZE, apply_scenery, build_scenery
        xml = f'''<mujoco>
  <asset><texture name="sky" type="skybox" builtin="gradient" rgb1="0 0 0" rgb2="1 1 1"/></asset>
  <visual><global offwidth="128" offheight="128"/><headlight/></visual>
  <worldbody>
    <geom type="plane" size="4 4 0.1"/>
    <body name="{PREFIX}body" pos="0 0 0.6">
      <body name="{PREFIX}arm_link_wr1" pos="0.3 0 0.4">
        <site name="tcp" pos="0.195 0 0.005"/>
      </body>
    </body>
  </worldbody>
</mujoco>'''
        with tempfile.TemporaryDirectory() as directory:
            scenery = build_scenery(
                seed=3, extra_fruit=4, rows=2, columns=2, spacing=5.0,
                slope_deg=1.0, slope_azimuth_deg=0.0, noise_m=0.02, rut_depth_m=0.04,
                landform_m=0.25, landform_wavelength_m=18.0, canopy_spacing=0.0,
                asset_dir=Path(directory), leaf_draw='mesh')
            root = ET.fromstring(xml)
            apply_scenery(root, scenery)
        chassis = root.find(f'.//body[@name="{PREFIX}body"]')
        storage = chassis.find("camera[@name='basket_cam']")
        self.assertIsNotNone(storage)
        pos = np.fromstring(storage.get('pos'), sep=' ')
        axes = np.fromstring(storage.get('xyaxes'), sep=' ')
        look = -np.cross(axes[:3], axes[3:])
        look /= max(float(np.linalg.norm(look)), 1e-9)
        self.assertGreater(pos[2], CENTER[2] + SIZE[2] + 0.2)
        self.assertLess(look[2], -0.5)
        self.assertLess(np.linalg.norm(pos[:2] - CENTER[:2]), 0.25)
        wrist = root.find(f'.//body[@name="{PREFIX}arm_link_wr1"]')
        cam = wrist.find("camera[@name='ee_cam']")
        self.assertIsNotNone(cam)
        self.assertAlmostEqual(float(cam.get('fovy')), 46.4, places=1)

    def test_storage_camera_looks_down_into_the_basket(self):
        from scripts.record_basket_orchard import CENTER, SIZE, basket_camera
        import mujoco

        class _Env:
            chassis = 0
            model = mujoco.MjModel.from_xml_string(
                '<mujoco><worldbody><body name="b" pos="1 2 0.8"><geom size=".05"/></body></worldbody></mujoco>')
            data = type('Data', (), {
                'xpos': np.array([[1.0, 2.0, 0.8]]),
                'xmat': np.eye(3).ravel()[None, :],
            })()

        camera = basket_camera(_Env())
        from scripts.record_basket_orchard import free_camera_eye
        basket = np.array([1.0, 2.0, 0.8]) + CENTER
        eye = free_camera_eye(camera.lookat, camera.distance, camera.azimuth, camera.elevation)
        self.assertGreater(eye[2], basket[2] + SIZE[2] + 0.2)
        self.assertLess(np.linalg.norm(camera.lookat[:2] - basket[:2]), 0.15)
        self.assertGreater(camera.lookat[2], basket[2] - 0.05)
        self.assertLess(camera.lookat[2], basket[2] + SIZE[2])

    def test_storage_inset_sees_basket_on_a_large_extent_scene(self):
        from scripts.record_basket_orchard import (
            CENTER, SIZE, _close_near_plane, _scene_option, add_storage_camera,
        )
        import mujoco

        chassis = ET.Element('body', name='spot_with_arm_body', pos='0 0 0.6')
        add_storage_camera(chassis)
        floor = CENTER + np.array([0.0, 0.0, SIZE[2] / 2])
        xml = f'''<mujoco>
  <statistic extent="40"/>
  <visual><global offwidth="160" offheight="90"/><map znear="0.01"/><headlight/></visual>
  <worldbody>
    <light pos="0 2 4"/>
    <geom type="plane" size="20 20 0.1" rgba="0.15 0.45 0.18 1"/>
    {ET.tostring(chassis, encoding='unicode')}
    <geom type="box" size="{SIZE[0]/2:.4f} {SIZE[1]/2:.4f} {SIZE[2]/2:.4f}"
          pos="{floor[0]:.4f} {floor[1]:.4f} {0.6 + floor[2]:.4f}"
          rgba="1 0.72 0.025 1"/>
  </worldbody>
</mujoco>'''
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, height=90, width=160)
        with _close_near_plane(model):
            renderer.update_scene(data, camera='basket_cam', scene_option=_scene_option())
            rgb = renderer.render()
        yellow = (rgb[:, :, 0] > 90) & (rgb[:, :, 1] > 50) & (rgb[:, :, 2] < 80)
        self.assertGreater(float(yellow.mean()), 0.08)

    def test_landscape_camera_keeps_robot_in_the_orchard_shot(self):
        from scripts.record_basket_orchard import free_camera_eye, landscape_camera

        class _Scenery:
            def ground_z(self, x, y):
                return 0.25 + 0.01 * float(x)

            def canopy_z(self, x, y):
                return 1.6 + self.ground_z(x, y)

        class _Env:
            scenery = _Scenery()
            chassis = 0
            data = type('Data', (), {'xpos': np.array([[0.4, -0.2, 0.85]])})()

        lookat, distance, azimuth, elevation = landscape_camera(_Env())
        eye = free_camera_eye(lookat, distance, azimuth, elevation)
        chassis = np.array([0.4, -0.2])
        canopy = _Env.scenery.canopy_z(eye[0], eye[1])
        self.assertGreater(distance, 4.0)
        self.assertLess(distance, 10.0)
        self.assertLess(eye[2], canopy - 0.2)
        self.assertGreater(np.linalg.norm(eye[:2] - chassis), 4.0)
        self.assertLess(np.linalg.norm(eye[:2] - chassis), 9.0)
        self.assertLess(np.linalg.norm(lookat[:2] - chassis), 4.0)

    def test_kiwi_leaf_roof_follows_tied_canes(self):
        floor = sample_orchard_floor(42, slope_deg=0.0, noise_m=0.0, rut_depth_m=0.0,
                                     landform_m=0.0)
        skel = generate(height=1.6, seed=42, rows=2, columns=2, spacing=5.0,
                        ground_z=floor.ground_z, canopy_z=floor.canopy_z)
        fp = FoliageParams(enabled=True, leaves_per_terminal=4, min_order_for_leaves=2,
                           leaf_length=0.22, leaf_width=0.17, leaf_shape='cordate',
                           canopy_spacing_m=0.4)
        leaves = place_leaves(skel, fp, seed=42, height_z=floor.canopy_z)
        roof = place_canopy_leaves(skel, fp, seed=42, height_z=floor.canopy_z)
        self.assertGreater(len(leaves), 10)
        self.assertGreater(len(roof), 20)


if __name__ == '__main__':
    unittest.main()
