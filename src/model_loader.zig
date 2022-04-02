const std = @import("std");
const scene = @import("scene.zig");
const Mesh = scene.Mesh;
const Texture = scene.Texture;
const c_allocator = std.heap.c_allocator;

const c = @cImport({
    @cInclude("src/stb_image.h");
});

// 1 mesh split into 2 materials (body & head)
pub const GordonData = struct {
    mesh: Mesh,
    textures: [2]Texture,
    indices_counts: [2]u32,

    pub fn deinit(self: *GordonData) void {
        self.mesh.deinit();
        for (self.textures) |*t| {
            t.deinit();
        }
    }
};

fn readAll(file: *std.fs.File, dst: []u8) !void {
    const r = try file.readAll(dst);
    if (r != dst.len) {
        return error.ReadIncomplete;
    }
}

/// Everything is hardcoded. The model file cannot be changed without changing the constants
/// in this function and GordonData.
pub fn load() !GordonData {
    const vertices_count: u32 = 595;
    const indices_material0_count: u32 = 1458;
    const indices_material1_count: u32 = 459;
    const indices_count: u32 = indices_material0_count + indices_material1_count;

    var file = try std.fs.cwd().openFile("gordon/gordon.data", .{});
    defer file.close();

    // Mesh

    var vertex_positions = try c_allocator.alloc([3]f32, vertices_count);
    errdefer c_allocator.free(vertex_positions);

    var vertices_other_attributes = try c_allocator.alloc(scene.VertexOtherAttributes, vertices_count);
    errdefer c_allocator.free(vertices_other_attributes);

    var indices = try c_allocator.alloc([3]u16, indices_count / 3);
    errdefer c_allocator.free(indices);

    try readAll(&file, std.mem.sliceAsBytes(vertex_positions));
    try readAll(&file, std.mem.sliceAsBytes(vertices_other_attributes));
    try readAll(&file, std.mem.sliceAsBytes(indices));

    var mesh = Mesh{
        .vertex_positions = vertex_positions,
        .vertices_other_attributes = vertices_other_attributes,
        .indices = indices,
    };

    // Textures

    var x: c_int = 0;
    var y: c_int = 0;
    var n: c_int = 0;

    var body_texture_data = @alignCast(4, c.stbi_load("gordon/DM_Base.bmp_baseColor.png", &x, &y, &n, 4));
    if (body_texture_data == null) {
        return error.ImageLoadError;
    }
    errdefer c.stbi_image_free(body_texture_data);

    const body_texture_width = @intCast(u32, x);
    const body_texture_height = @intCast(u32, y);

    var body_texture = Texture{
        .width = body_texture_width,
        .height = body_texture_height,
        .colours = std.mem.bytesAsSlice(
            u32,
            std.mem.sliceAsBytes(body_texture_data[0 .. body_texture_width * body_texture_height * 4]),
        ),
    };

    var head_texture_data = @alignCast(4, c.stbi_load("gordon/DM_Face.bmp_baseColor.png", &x, &y, &n, 4));
    if (head_texture_data == null) {
        return error.ImageLoadError;
    }
    errdefer c.stbi_image_free(head_texture_data);

    const head_texture_width = @intCast(u32, x);
    const head_texture_height = @intCast(u32, y);

    var head_texture = Texture{
        .width = head_texture_width,
        .height = head_texture_height,
        .colours = std.mem.bytesAsSlice(
            u32,
            std.mem.sliceAsBytes(head_texture_data[0 .. head_texture_width * head_texture_height * 4]),
        ),
    };

    return GordonData{
        .mesh = mesh,
        .textures = [2]Texture{ body_texture, head_texture },
        .indices_counts = [2]u32{ indices_material0_count, indices_material1_count },
    };
}
