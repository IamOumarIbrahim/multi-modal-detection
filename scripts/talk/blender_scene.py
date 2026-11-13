#   MMSAR - 3D hero render: UAV, camera frustum and the two 640x640 tiles on the ground
#        Oumar Ibrahim U22200741

# blender -b --factory-startup -P scripts/talk/blender_scene.py -- figures/render_uav.png [--samples 64]
# schematic, not to scale: the person is drawn about 2x larger so it is visible on a slide
# the tile footprints are exact ray casts of a 1280x720 pinhole camera (60 deg horizontal FOV,
# 40 deg below horizontal), left tile x 0..640, right tile x 640..1280, both y 40..680

import math
import os
import random
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT = argv[0] if argv else "figures/render_uav.png"
SAMPLES = int(argv[argv.index("--samples") + 1]) if "--samples" in argv else 64

BLUE = (0.165, 0.471, 0.839)    # #2a78d6 left tile
ORANGE = (0.922, 0.408, 0.204)  # #eb6834 right tile
SAND = (0.80, 0.66, 0.46)
random.seed(7)


def srgb_to_lin(c):
    return tuple(((x + 0.055) / 1.055) ** 2.4 if x > 0.04045 else x / 12.92 for x in c)


def material(name, color, rough=0.8, emission=None, strength=0.0, alpha=1.0, metallic=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (*srgb_to_lin(color), 1)
    p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = metallic
    if emission is not None:
        p.inputs["Emission Color"].default_value = (*srgb_to_lin(emission), 1)
        p.inputs["Emission Strength"].default_value = strength
    if alpha < 1:
        p.inputs["Alpha"].default_value = alpha
        if hasattr(m, "surface_render_method"):
            m.surface_render_method = "BLENDED"
        elif hasattr(m, "blend_method"):
            m.blend_method = "BLEND"
    return m


def link(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return obj


bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# ---------------------------------------------------------------- terrain
bpy.ops.mesh.primitive_plane_add(size=90, location=(0, 18, 0))
ground = bpy.context.object
ground.name = "ground"
sub = ground.modifiers.new("sub", "SUBSURF")
sub.subdivision_type = "SIMPLE"
sub.levels = sub.render_levels = 7
tex = bpy.data.textures.new("dunes", "CLOUDS")
tex.noise_scale = 6.0
disp = ground.modifiers.new("disp", "DISPLACE")
disp.texture = tex
disp.strength = 0.7
disp.mid_level = 0.5
gm = material("sand", SAND, rough=0.95)
nt = gm.node_tree
noise = nt.nodes.new("ShaderNodeTexNoise")
noise.inputs["Scale"].default_value = 1.2
ramp = nt.nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].color = (*srgb_to_lin((0.70, 0.56, 0.38)), 1)
ramp.color_ramp.elements[1].color = (*srgb_to_lin((0.87, 0.75, 0.56)), 1)
nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
nt.links.new(ramp.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
link(ground, gm)
bpy.ops.object.shade_smooth()

# rocks and dry bushes
rock_m = material("rock", (0.36, 0.30, 0.25), rough=0.9)
bush_m = material("bush", (0.42, 0.40, 0.24), rough=1.0)
for i in range(140):
    x = random.uniform(-30, 30)
    y = random.uniform(-6, 45)
    s = random.uniform(0.12, 0.45)
    if i % 3 == 0:
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=s, location=(x, y, 0.1))
        o = bpy.context.object
        o.scale = (1, random.uniform(0.7, 1.3), random.uniform(0.4, 0.8))
        link(o, rock_m)
    else:
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=s * 0.8, location=(x, y, 0.2))
        o = bpy.context.object
        o.scale = (1, 1, 0.6)
        link(o, bush_m)

# ---------------------------------------------------------------- drone and camera model
DRONE = Vector((0, -14, 12))
PITCH = math.radians(40)
W, H, HFOV = 1280, 720, math.radians(60)
f = (W / 2) / math.tan(HFOV / 2)
F = Vector((0, math.cos(PITCH), -math.sin(PITCH)))
R = Vector((1, 0, 0))
UP = R.cross(F)
DOWN = -UP


def ground_hit(u, v, z=0.42):
    d = (R * ((u - W / 2) / f) + DOWN * ((v - H / 2) / f) + F).normalized()
    t = (z - DRONE.z) / d.z
    return DRONE + d * t


body_m = material("drone", (0.12, 0.13, 0.15), rough=0.35, metallic=0.3)
accent_m = material("accent", (0.08, 0.16, 0.35), rough=0.3)
rotor_m = material("rotor", (0.6, 0.62, 0.66), rough=0.4, alpha=0.45)
bpy.ops.mesh.primitive_cube_add(size=1, location=DRONE)
b = bpy.context.object
b.scale = (0.9, 1.2, 0.28)
bev = b.modifiers.new("bev", "BEVEL")
bev.width = 0.12
bev.segments = 4
link(b, body_m)
for sx in (-1, 1):
    for sy in (-1, 1):
        a = DRONE + Vector((sx * 1.15, sy * 1.15, 0.05))
        bpy.ops.mesh.primitive_cylinder_add(radius=0.07, depth=1.6, location=(DRONE + a) / 2)
        arm = bpy.context.object
        arm.rotation_euler = (0, math.pi / 2, math.atan2(sy, sx))
        link(arm, body_m)
        bpy.ops.mesh.primitive_cylinder_add(radius=0.16, depth=0.18, location=a + Vector((0, 0, 0.1)))
        link(bpy.context.object, accent_m)
        bpy.ops.mesh.primitive_cylinder_add(radius=0.75, depth=0.02, location=a + Vector((0, 0, 0.22)))
        link(bpy.context.object, rotor_m)
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.22, location=DRONE + Vector((0, 0.35, -0.32)))
link(bpy.context.object, accent_m)

# ---------------------------------------------------------------- frustum and tile footprints
def quad(name, pts, mat):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(p) for p in pts], [], [(0, 1, 2, 3)])
    o = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(o)
    return link(o, mat)


def tube(name, a, b, mat, r=0.05):
    d = b - a
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=d.length, location=(a + b) / 2, vertices=12)
    o = bpy.context.object
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = d.to_track_quat("Z", "Y")
    o.name = name
    return link(o, mat)


fill_l = material("fill_l", BLUE, emission=BLUE, strength=1.5, alpha=0.32)
fill_r = material("fill_r", ORANGE, emission=ORANGE, strength=1.5, alpha=0.32)
edge_l = material("edge_l", BLUE, emission=BLUE, strength=1.2)
edge_r = material("edge_r", ORANGE, emission=ORANGE, strength=1.2)
ray_m = material("ray", (1, 1, 1), emission=(1, 1, 1), strength=2.5, alpha=0.6)
frame_m = material("frame", (0.9, 0.9, 0.9), emission=(0.9, 0.9, 0.9), strength=0.8, alpha=0.18)

cam_pos = DRONE + Vector((0, 0.35, -0.32))
full = [ground_hit(0, 0), ground_hit(W, 0), ground_hit(W, H), ground_hit(0, H)]
quad("frame_full", full, frame_m)
for i, p in enumerate(full):
    tube(f"ray{i}", cam_pos, p, ray_m, r=0.025)
for name, u0, fill, edge in [("left", 0, fill_l, edge_l), ("right", 640, fill_r, edge_r)]:
    pts = [ground_hit(u0, 40, 0.48), ground_hit(u0 + 640, 40, 0.48), ground_hit(u0 + 640, 680, 0.48),
           ground_hit(u0, 680, 0.48)]
    quad(f"tile_{name}", pts, fill)
    for i in range(4):
        tube(f"edge_{name}{i}", pts[i], pts[(i + 1) % 4], edge, r=0.07)

# ---------------------------------------------------------------- person (2x scale) and a dark distractor rock
person_at = ground_hit(330, 430, 0.0)
cloth = material("cloth", (0.18, 0.20, 0.17), rough=0.9)
skin = material("skin", (0.55, 0.40, 0.30), rough=0.7)
bpy.ops.mesh.primitive_cylinder_add(radius=0.28, depth=1.6, location=person_at + Vector((0, 0, 1.25)))
link(bpy.context.object, cloth)
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.25, location=person_at + Vector((0, 0, 2.3)))
link(bpy.context.object, skin)
for sx in (-0.13, 0.13):
    bpy.ops.mesh.primitive_cylinder_add(radius=0.11, depth=0.9, location=person_at + Vector((sx, 0, 0.45)))
    link(bpy.context.object, cloth)
box_m = material("box", (1, 1, 1), emission=(1, 1, 1), strength=4)
c = person_at + Vector((0, 0, 1.3))
corners = [c + Vector((-0.75, 0, -1.4)), c + Vector((0.75, 0, -1.4)), c + Vector((0.75, 0, 1.4)),
           c + Vector((-0.75, 0, 1.4))]
for i in range(4):
    tube(f"bbox{i}", corners[i], corners[(i + 1) % 4], box_m, r=0.04)

decoy_at = ground_hit(960, 380, 0.0)
bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.9, location=decoy_at + Vector((0, 0, 0.6)))
d = bpy.context.object
d.scale = (0.55, 0.5, 1.25)
link(d, material("decoy", (0.10, 0.09, 0.08), rough=0.9))

# ---------------------------------------------------------------- light, world, camera
bpy.ops.object.light_add(type="SUN", location=(0, 0, 30))
sun = bpy.context.object
sun.data.energy = 4.0
sun.data.angle = math.radians(4)
sun.rotation_euler = (math.radians(50), math.radians(10), math.radians(-35))
world = bpy.data.worlds.new("sky")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
sep = world.node_tree.nodes.new("ShaderNodeSeparateXYZ")
mapping = world.node_tree.nodes.new("ShaderNodeMapping")
coord = world.node_tree.nodes.new("ShaderNodeTexCoord")
ramp = world.node_tree.nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].color = (*srgb_to_lin((0.93, 0.95, 0.97)), 1)
ramp.color_ramp.elements[1].color = (*srgb_to_lin((0.55, 0.72, 0.92)), 1)
mapping.inputs["Rotation"].default_value = (0, 0, 0)
world.node_tree.links.new(coord.outputs["Window"], mapping.inputs["Vector"])
world.node_tree.links.new(mapping.outputs["Vector"], sep.inputs["Vector"])
world.node_tree.links.new(sep.outputs["Y"], ramp.inputs["Fac"])
world.node_tree.links.new(ramp.outputs["Color"], bg.inputs["Color"])
bg.inputs["Strength"].default_value = 1.0

bpy.ops.object.camera_add(location=(24, -33, 19))
cam = bpy.context.object
cam.data.lens = 32
target = Vector((-3.5, 3, 3.5))
cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
scene.camera = cam

scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.film_transparent = False
for engine in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
    try:
        scene.render.engine = engine
        break
    except TypeError:
        continue
if hasattr(scene, "eevee"):
    scene.eevee.taa_render_samples = SAMPLES
    for attr in ("use_shadows", "use_raytracing"):
        if hasattr(scene.eevee, attr):
            setattr(scene.eevee, attr, True)
scene.view_settings.view_transform = "AgX" if "AgX" in [v.identifier for v in
                                                        type(scene.view_settings).bl_rna.properties["view_transform"].enum_items] else "Filmic"
scene.render.filepath = os.path.abspath(OUT)
scene.render.image_settings.file_format = "PNG"
bpy.ops.render.render(write_still=True)
print("saved", scene.render.filepath)
