import bpy
import math
import os

# Adjustable parameters
OBJECT_POSITION = (0, 0, 0)
CAMERA_POSITION = (0, 0, 10)
CAMERA_ROTATION = (0, 0, math.pi)
BACKGROUND_POSITION = (0, 0, 0)
STL_FILE_PATH = r"C:\Users\User\Downloads\fluid.stl"


def delete_default_objects():
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.data.objects:
        if obj.name in ["Cube", "Light"]:
            obj.select_set(True)
    bpy.ops.object.delete()


def import_stl(filepath):
    bpy.ops.import_mesh.stl(filepath=filepath)
    imported_object = bpy.context.selected_objects[0]
    return imported_object


def center_object_to_origin(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    obj.location = (0, 0, 0)


def scale_object(obj, scale):
    obj.scale = (scale, scale, scale)


def setup_eevee_render():
    bpy.context.scene.render.engine = "BLENDER_EEVEE"
    bpy.context.scene.eevee.use_ssr = True
    bpy.context.scene.eevee.use_ssr_refraction = True
    bpy.context.scene.render.resolution_x = 900
    bpy.context.scene.render.resolution_y = 900


def apply_glass_material(obj):
    mat = bpy.data.materials.new(name="Glass Material")

    obj.data.materials.clear()
    obj.data.materials.append(mat)

    mat.use_nodes = True
    principled = mat.node_tree.nodes.get("Principled BSDF")
    if principled:
        principled.inputs["Base Color"].default_value = (
            0.1,
            0.5,
            0.0,
            1,
        )  # Yellow color (RGB + Alpha)
        if "Transmission Weight" in principled.inputs:
            principled.inputs["Transmission Weight"].default_value = 1.0
        elif "Transmission" in principled.inputs:
            principled.inputs["Transmission"].default_value = 1.0
        principled.inputs["Roughness"].default_value = 0.0

    mat.use_screen_refraction = True


def create_checkered_background():
    bpy.ops.mesh.primitive_plane_add(size=50, location=BACKGROUND_POSITION)
    bg_plane = bpy.context.active_object

    mat = bpy.data.materials.new(name="Checkered Background")
    bg_plane.data.materials.append(mat)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    checker = nodes.new(type="ShaderNodeTexChecker")
    diffuse = nodes.new(type="ShaderNodeBsdfDiffuse")
    material_output = nodes.new(type="ShaderNodeOutputMaterial")

    mapping.inputs["Scale"].default_value = (10, 10, 10)
    checker.inputs["Color1"].default_value = (0.25, 0.25, 0.25, 1)
    checker.inputs["Color2"].default_value = (0.2, 0.2, 0.2, 1)

    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], checker.inputs["Vector"])
    links.new(checker.outputs["Color"], diffuse.inputs["Color"])
    links.new(diffuse.outputs["BSDF"], material_output.inputs["Surface"])


def setup_scene(obj):
    obj.location = OBJECT_POSITION
    create_checkered_background()

    if bpy.context.scene.camera is None:
        bpy.ops.object.camera_add()
        camera = bpy.context.active_object
    else:
        camera = bpy.context.scene.camera

    camera.location = CAMERA_POSITION
    camera.rotation_euler = CAMERA_ROTATION
    camera.data.type = "PERSP"
    camera.data.lens = 50

    bpy.context.scene.camera = camera


def add_light():
    bpy.ops.object.light_add(type="AREA", location=(0, 0, 10))
    main_light = bpy.context.active_object
    main_light.data.energy = 500.0
    main_light.data.use_shadow = False  # Disable shadows for sun light

    bpy.ops.object.light_add(type="AREA", location=(-3, 3, 10))
    fill_light = bpy.context.active_object
    fill_light.data.energy = 200.0
    fill_light.data.use_shadow = False  # Disable shadows for sun light


def setup_render_properties():
    bpy.context.scene.eevee.use_ssr = True
    bpy.context.scene.eevee.use_ssr_refraction = True


def render_scene(filepath):
    bpy.context.scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)


def rotate_object(obj, angle):
    obj.rotation_euler[1] += math.radians(angle)


def main():
    setup_eevee_render()
    # setup_world()

    # Delete default cube and light
    delete_default_objects()

    # Import STL file
    imported_obj = import_stl(STL_FILE_PATH)

    # Center object to origin
    center_object_to_origin(imported_obj)

    # Scale object
    scale_object(imported_obj, 0.03)

    apply_glass_material(imported_obj)
    setup_scene(imported_obj)
    add_light()
    setup_render_properties()

    # First snapshot (original position)
    filepath = r"C:\Users\User\Downloads\fluid_0.png"
    render_scene(filepath)
    print(f"First render saved to {filepath}")

    # # Rotate 90 degrees around Y-axis and take second snapshot
    # rotate_object(imported_obj, 90)
    # filepath = r"C:\Users\User\Downloads\fluid_shader_screenshot_2.png"
    # render_scene(filepath)
    # print(f"Second render saved to {filepath}")

    # # Rotate -180 degrees around Y-axis and take third snapshot
    # rotate_object(imported_obj, -180)
    # filepath = r"C:\Users\User\Downloads\fluid_shader_screenshot_3.png"
    # render_scene(filepath)
    # print(f"Third render saved to {filepath}")


# Run the script
if __name__ == "__main__":
    main()
