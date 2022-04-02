const std = @import("std");
const ArrayList = std.ArrayList;
const c_allocator = std.heap.c_allocator;
const Matrix = @import("maths/matrix.zig").Matrix;
const vector = @import("maths/vector.zig");
const Vec4 = vector.Vec4;
const toRadians = @import("maths/maths.zig").toRadians;
const model_loader = @import("model_loader.zig");

// extern because this data is read direct from a file
pub const VertexOtherAttributes = extern struct {
    normal: [3]f32,
    uv: [2]f32,
};

pub const Mesh = struct {
    // length of both vertex arrays must match
    vertex_positions: []const [3]f32, // c_allocator
    vertices_other_attributes: []const VertexOtherAttributes, // c_allocator

    indices: []const [3]u16, // c_allocator

    pub fn deinit(self: *Mesh) void {
        c_allocator.free(self.vertex_positions);
        c_allocator.free(self.vertices_other_attributes);
        c_allocator.free(self.indices);
    }
};

pub const Texture = struct {
    width: u32,
    height: u32,
    colours: []u32, // c_allocator

    pub fn deinit(self: *Texture) void {
        c_allocator.free(self.colours);
    }
};

pub const DrawAtom = struct {
    mesh: *Mesh,
    first_index: u32,
    indices_count: u32,
    texture: *Texture,
    transform: Matrix(f32, 4), // mvp
    object_transform: Matrix(f32, 4), // object -> world, (model matrix)
};

pub const Scene = struct {
    meshes: ArrayList(Mesh) = ArrayList(Mesh).init(c_allocator),
    textures: ArrayList(Texture) = ArrayList(Texture).init(c_allocator),

    draw_atoms: ArrayList(DrawAtom) = ArrayList(DrawAtom).init(c_allocator),

    inverse_light_direction: Vec4,

    pub fn deinit(self: *Scene) void {
        for (self.meshes.items) |*m| {
            m.deinit();
        }
        for (self.textures.items) |*t| {
            t.deinit();
        }

        self.meshes.deinit();
        self.textures.deinit();
        self.draw_atoms.deinit();
    }
};

pub fn createScene(viewport_width: u32, viewport_height: u32) !Scene {
    var to_light = Vec4{ 0.2, 1.0, 0.8, 0.0 };
    to_light = vector.normalised(to_light);

    var self = Scene{ .inverse_light_direction = to_light };
    errdefer self.deinit();

    const proj = Matrix(f32, 4).perspectiveProjection(
        @intToFloat(f32, viewport_width) / @intToFloat(f32, viewport_height),
        toRadians(45.0),
        10.0,
        0.1,
    );

    const flipY = Matrix(f32, 4).scale(Vec4{ 1.0, -1.0, 1.0, 1.0 });

    // const flipZ = Matrix(f32, 4).scale(Vec4{  1.0, 1.0, -1.0, 1.0  });
    // const zAdd1 = Matrix(f32, 4).translate(.{ 0.0, 0.0, 1.0  });
    // const flip_depth = flipZ.mul(zAdd1);

    const rotate = Matrix(f32, 4).rotateY(toRadians(15.0));
    const translate = Matrix(f32, 4).translate(.{ 0.0, -0.9, -2.5 });

    const obj_m = rotate.mul(translate);
    const m = obj_m.mul(proj).mul(flipY);

    try self.meshes.ensureUnusedCapacity(1);
    try self.textures.ensureUnusedCapacity(2);
    try self.draw_atoms.ensureUnusedCapacity(2);

    var gordon = try model_loader.load();

    self.meshes.appendAssumeCapacity(gordon.mesh);
    self.textures.appendAssumeCapacity(gordon.textures[0]);
    self.textures.appendAssumeCapacity(gordon.textures[1]);

    self.draw_atoms.appendAssumeCapacity(.{
        .mesh = &self.meshes.items[0],
        .first_index = 0,
        .indices_count = gordon.indices_counts[0],
        .texture = &self.textures.items[0],
        .object_transform = obj_m,
        .transform = m,
    });

    self.draw_atoms.appendAssumeCapacity(.{
        .mesh = &self.meshes.items[0],
        .first_index = gordon.indices_counts[0],
        .indices_count = gordon.indices_counts[1],
        .texture = &self.textures.items[1],
        .object_transform = obj_m,
        .transform = m,
    });

    return self;
}
