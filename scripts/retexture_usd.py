#!/usr/bin/env python3
"""Import Newton USD into Blender, add orchard PBR materials, and render a frame."""
import argparse
import math
import sys

import bpy
from mathutils import Vector


def principled(name, color, roughness, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (*color, 1)
    shader.inputs['Roughness'].default_value = roughness
    shader.inputs['Metallic'].default_value = metallic
    return mat, shader


def noise_color(mat, shader, scale, dark, light, bump=0.0, detail=5.0):
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    coord = nodes.new('ShaderNodeTexCoord')
    noise = nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = scale
    noise.inputs['Detail'].default_value = detail
    noise.inputs['Roughness'].default_value = 0.72
    ramp = nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].color = (*dark, 1)
    ramp.color_ramp.elements[1].color = (*light, 1)
    links.new(coord.outputs['Object'], noise.inputs['Vector'])
    links.new(noise.outputs['Fac'], ramp.inputs['Fac'])
    links.new(ramp.outputs['Color'], shader.inputs['Base Color'])
    if bump:
        bump_node = nodes.new('ShaderNodeBump')
        bump_node.inputs['Strength'].default_value = bump
        bump_node.inputs['Distance'].default_value = 0.025
        links.new(noise.outputs['Fac'], bump_node.inputs['Height'])
        links.new(bump_node.outputs['Normal'], shader.inputs['Normal'])


def build_materials():
    ground, g = principled('Grassed orchard soil', (0.11, 0.22, 0.035), 0.96)
    noise_color(ground, g, 0.7, (0.075, 0.025, 0.008), (0.16, 0.32, 0.045), 0.62, 7.0)
    g.inputs['Diffuse Roughness'].default_value = 0.8

    wood, w = principled('Weathered trellis wood', (0.22, 0.075, 0.018), 0.82)
    noise_color(wood, w, 5.5, (0.08, 0.018, 0.004), (0.36, 0.14, 0.025), 0.28)

    steel, s = principled('Galvanized support wire', (0.34, 0.39, 0.42), 0.28, 0.82)
    noise_color(steel, s, 22.0, (0.18, 0.22, 0.24), (0.55, 0.59, 0.61), 0.08)

    vine, v = principled('Kiwi vine bark', (0.17, 0.045, 0.012), 0.88)
    noise_color(vine, v, 9.0, (0.055, 0.012, 0.003), (0.30, 0.09, 0.015), 0.42)

    leaf, l = principled('Kiwi leaf', (0.025, 0.20, 0.012), 0.63)
    noise_color(leaf, l, 5.0, (0.008, 0.055, 0.003), (0.07, 0.38, 0.018), 0.12)
    l.inputs['Subsurface Weight'].default_value = 0.08
    l.inputs['Subsurface Radius'].default_value = (0.6, 1.0, 0.35)
    l.inputs['Sheen Weight'].default_value = 0.12
    l.inputs['Thin Wall'].default_value = True

    fruit, f = principled('Kiwi skin', (0.30, 0.14, 0.035), 0.92)
    noise_color(fruit, f, 35.0, (0.10, 0.035, 0.006), (0.43, 0.23, 0.065), 0.82, 8.0)
    f.inputs['Coat Weight'].default_value = 0.04
    f.inputs['Coat Roughness'].default_value = 0.72
    return dict(ground=ground, wood=wood, steel=steel, vine=vine, leaf=leaf, fruit=fruit)


def classify(obj):
    dims = sorted(float(x) for x in obj.dimensions)
    if dims[2] > 20.0:
        return 'ground'
    if dims[2] <= 0.09 and dims[0] >= 0.03:
        return 'fruit'
    if dims[2] > 3.0 * max(dims[1], 1e-6):
        diameter, length = dims[1], dims[2]
        if diameter >= 0.035:
            return 'wood'
        if diameter <= 0.010 and length > 1:
            return 'steel'
        return 'vine'
    return 'leaf'


def assign(obj, material, semantic):
    obj.data = obj.data.copy()
    obj.data.materials.clear()
    obj.data.materials.append(material)
    obj['orchard_material'] = semantic
    for polygon in obj.data.polygons:
        polygon.use_smooth = semantic not in ('ground', 'leaf')


def look_at(obj, point):
    obj.rotation_euler = (Vector(point) - obj.location).to_track_quat('-Z', 'Y').to_euler()


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument('usd')
    parser.add_argument('--output', required=True)
    parser.add_argument('--blend')
    parser.add_argument('--frame', type=int, default=0)
    args = parser.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(filepath=args.usd, import_materials=True)
    materials = build_materials()
    for obj in bpy.data.objects:
        if obj.type == 'POINTCLOUD':
            obj.hide_render = True
    counts = {key: 0 for key in materials}
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            semantic = classify(obj)
            assign(obj, materials[semantic], semantic)
            obj.name = f'{semantic}_{counts[semantic]:04d}'
            counts[semantic] += 1
    print('PBR semantic classes:', counts)

    world = bpy.data.worlds.new('Clear orchard sky')
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.remove(nodes.get('Background'))
    sky = nodes.new('ShaderNodeTexSky')
    sky.sky_type = 'MULTIPLE_SCATTERING'
    sky.sun_elevation = math.radians(32)
    sky.sun_rotation = math.radians(225)
    sky.altitude = 250
    sky.air_density = 1.05
    background = nodes.new('ShaderNodeBackground')
    background.inputs['Strength'].default_value = 0.32
    world.node_tree.links.new(sky.outputs['Color'], background.inputs['Color'])
    world.node_tree.links.new(background.outputs['Background'], nodes['World Output'].inputs['Surface'])

    sun_data = bpy.data.lights.new('Late morning sun', 'SUN')
    sun_data.energy = 2.2
    sun_data.angle = math.radians(5.0)
    sun = bpy.data.objects.new('Late morning sun', sun_data)
    bpy.context.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(38), 0, math.radians(-42))

    camera_data = bpy.data.cameras.new('Canopy camera')
    camera = bpy.data.objects.new('Canopy camera', camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (-1.3, 7.5, 1.18)
    camera.data.lens = 34
    camera.data.sensor_width = 36
    camera.data.dof.use_dof = True
    camera.data.dof.focus_distance = 8.0
    camera.data.dof.aperture_fstop = 7.1
    look_at(camera, (8.5, 7.5, 1.32))

    scene = bpy.context.scene
    scene.camera = camera
    scene.frame_set(args.frame)
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = args.output
    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = 'RGBA'
    scene.view_settings.look = 'AgX - Medium High Contrast'
    if args.blend:
        bpy.ops.wm.save_as_mainfile(filepath=args.blend)
    bpy.ops.render.render(write_still=True)


if __name__ == '__main__':
    main()
