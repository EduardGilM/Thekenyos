#!/usr/bin/env python3
"""Import Newton USD into Blender, add procedural PBR materials, and render a frame."""
import argparse
import math
import sys

import bpy
from mathutils import Vector


def material(name, color, roughness, noise_scale=0.0, bump=0.0, metallic=0.0):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*color, 1)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (*color, 1)
    bsdf.inputs['Roughness'].default_value = roughness
    bsdf.inputs['Metallic'].default_value = metallic
    if noise_scale:
        noise = m.node_tree.nodes.new('ShaderNodeTexNoise')
        noise.inputs['Scale'].default_value = noise_scale
        noise.inputs['Detail'].default_value = 5.0
        noise.inputs['Roughness'].default_value = 0.7
        ramp = m.node_tree.nodes.new('ShaderNodeValToRGB')
        lo = tuple(max(0, c * 0.55) for c in color)
        hi = tuple(min(1, c * 1.35) for c in color)
        ramp.color_ramp.elements[0].color = (*lo, 1)
        ramp.color_ramp.elements[1].color = (*hi, 1)
        m.node_tree.links.new(noise.outputs['Fac'], ramp.inputs['Fac'])
        m.node_tree.links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
        if bump:
            bn = m.node_tree.nodes.new('ShaderNodeBump')
            bn.inputs['Strength'].default_value = bump
            bn.inputs['Distance'].default_value = 0.08
            m.node_tree.links.new(noise.outputs['Fac'], bn.inputs['Height'])
            m.node_tree.links.new(bn.outputs['Normal'], bsdf.inputs['Normal'])
    return m


def assign(obj, mat):
    if obj.type == 'MESH':
        obj.data = obj.data.copy()
        obj.data.materials.clear()
        obj.data.materials.append(mat)


def look_at(obj, point):
    obj.rotation_euler = (Vector(point) - obj.location).to_track_quat('-Z', 'Y').to_euler()


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument('usd')
    p.add_argument('--output', required=True)
    p.add_argument('--blend')
    p.add_argument('--frame', type=int, default=0)
    args = p.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(filepath=args.usd, import_materials=True)

    soil = material('Orchard soil', (0.17, 0.075, 0.025), 0.93, 2.8, 0.65)
    wood = material('Weathered trellis wood', (0.24, 0.09, 0.025), 0.78, 5.0, 0.25)
    steel = material('Galvanized wire', (0.32, 0.35, 0.37), 0.32, 18.0, 0.08, 0.7)
    vine = material('Kiwi vine', (0.16, 0.055, 0.018), 0.86, 7.0, 0.35)
    leaf = material('Kiwi leaves', (0.055, 0.22, 0.025), 0.72, 4.0, 0.18)
    fruit = material('Kiwi skin', (0.27, 0.16, 0.055), 0.94, 24.0, 0.85)

    for o in bpy.data.objects:
        if o.type != 'MESH':
            continue
        dims = sorted(float(x) for x in o.dimensions)
        if dims[2] > 20.0:
            mat = soil
        elif dims[2] <= 0.09 and dims[0] >= 0.03:
            mat = fruit
        elif dims[2] > 3.0 * max(dims[1], 1e-6):
            diameter, length = dims[1], dims[2]
            mat = wood if diameter >= 0.035 else steel if diameter <= 0.010 and length > 1 else vine
        else:
            mat = leaf
        assign(o, mat)

    world = bpy.data.worlds.new('Orchard sky')
    bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.18, 0.32, 0.52, 1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.35

    sun_data = bpy.data.lights.new('Sun', 'SUN')
    sun_data.energy = 3.0
    sun_data.angle = math.radians(4)
    sun = bpy.data.objects.new('Sun', sun_data)
    bpy.context.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(28), 0, math.radians(-35))

    camera_data = bpy.data.cameras.new('Camera')
    camera = bpy.data.objects.new('Camera', camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (22, -18, 7.5)
    camera.data.lens = 46
    look_at(camera, (6.5, 8.0, 1.0))
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    scene.frame_set(args.frame)
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = args.output
    scene.render.film_transparent = False
    scene.view_settings.look = 'AgX - Medium High Contrast'
    if args.blend:
        bpy.ops.wm.save_as_mainfile(filepath=args.blend)
    bpy.ops.render.render(write_still=True)


if __name__ == '__main__':
    main()
