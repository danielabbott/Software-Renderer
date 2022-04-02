const std = @import("std");
const ArrayList = std.ArrayList;
const c_allocator = std.heap.c_allocator;
const TriangleBins = @import("bins.zig").TriangleBins;
const Scene = @import("scene.zig").Scene;
const vector = @import("maths/vector.zig");
const Vec4 = vector.Vec4;
const colour = @import("colour.zig");

pub fn processVertices(bins: *TriangleBins, scene: *Scene) !void {
    const vp_w_2 = @intToFloat(f32, bins.viewport_width) * 0.5;
    const vp_h_2 = @intToFloat(f32, bins.viewport_height) * 0.5;

    for (bins.bins.items) |*bin| {
        bin.triangles.clearRetainingCapacity();
    }

    for (scene.draw_atoms.items) |atom| {
        var mesh = atom.mesh;
        std.debug.assert(mesh.vertex_positions.len == mesh.vertices_other_attributes.len);

        for (mesh.indices[atom.first_index / 3 .. (atom.first_index + atom.indices_count) / 3]) |idx3| {
            var verts: [3]TriangleBins.VertexOutput = undefined;

            for (idx3) |idx_, i| {
                const idx = std.math.min(idx_, @intCast(u16, mesh.vertex_positions.len));

                var v = Vec4{
                    mesh.vertex_positions[idx][0],
                    mesh.vertex_positions[idx][1],
                    mesh.vertex_positions[idx][2],
                    1.0,
                };
                v = vector.mulMat(v, atom.transform);

                const reciprocal_w = 1.0 / v[3];
                v[0] *= reciprocal_w;
                v[1] *= reciprocal_w;
                v[2] *= reciprocal_w;

                // e.g. [-1,1] -> [0,640]
                verts[i].position_pixel_space[0] = (v[0] + 1.0) * vp_w_2;
                verts[i].position_pixel_space[1] = (v[1] + 1.0) * vp_h_2;
                verts[i].depth = v[2];
                verts[i].reciprocal_w = reciprocal_w;
            }

            var v01 = [2]f32{
                verts[1].position_pixel_space[0] - verts[0].position_pixel_space[0],
                verts[1].position_pixel_space[1] - verts[0].position_pixel_space[1],
            };

            // Same formula that is used for fragment triangle coverage test
            const winding_order = (v01[0]) * (verts[2].position_pixel_space[1] - verts[0].position_pixel_space[1]) >
                (v01[1]) * (verts[2].position_pixel_space[0] - verts[0].position_pixel_space[0]);

            if (!winding_order) {
                continue;
            }

            for (idx3) |idx_, i| {
                const idx = std.math.min(idx_, @intCast(u16, mesh.vertex_positions.len));

                const vertex = mesh.vertices_other_attributes[idx];

                var normal = Vec4{ vertex.normal[0], vertex.normal[1], vertex.normal[2], 0.0 };
                normal = vector.mulMat(normal, atom.object_transform); // assumes uniform scale

                verts[i].light = std.math.max(vector.dot(normal, scene.inverse_light_direction), 0.0) * 0.75 + 0.25;

                std.mem.copy(f32, verts[i].uv[0..2], vertex.uv[0..2]);
                verts[i].texture = atom.texture;
            }

            for (bins.bins.items) |*bin| {
                const tile_left = bin.x_f32;
                const tile_top = bin.y_f32;
                const tile_right = bin.right_f32;
                const tile_bottom = bin.bottom_f32;

                if (verts[0].position_pixel_space[0] >= tile_left or verts[1].position_pixel_space[0] >= tile_left or verts[2].position_pixel_space[0] >= tile_left) {
                    if (verts[0].position_pixel_space[0] <= tile_right or verts[1].position_pixel_space[0] <= tile_right or verts[2].position_pixel_space[0] <= tile_right) {
                        if (verts[0].position_pixel_space[1] >= tile_top or verts[1].position_pixel_space[1] >= tile_top or verts[2].position_pixel_space[1] >= tile_top) {
                            if (verts[0].position_pixel_space[1] <= tile_bottom or verts[1].position_pixel_space[1] <= tile_bottom or verts[2].position_pixel_space[1] <= tile_bottom) {
                                try bin.triangles.append(verts);
                            }
                        }
                    }
                }
            }
        }
    }
}
