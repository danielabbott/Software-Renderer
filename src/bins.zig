const std = @import("std");
const ArrayList = std.ArrayList;
const c_allocator = std.heap.c_allocator;
const png = @import("png.zig");
const Texture = @import("scene.zig").Texture;

pub const TriangleBins = struct {
    pub const VertexOutput = struct {
        position_pixel_space: [2]f32,
        depth: f32,
        reciprocal_w: f32,
        light: f32,
        uv: [2]f32,
        texture: *Texture, // TODO this does not need to be stored per triangle
    };
    pub const TriangleOutput = [3]VertexOutput;

    pub const Bin = struct {
        x: u32,
        y: u32,
        w: u32,
        h: u32,

        // Same values as before but floats
        x_f32: f32,
        y_f32: f32,
        w_f32: f32,
        h_f32: f32,

        right_f32: f32,
        bottom_f32: f32,

        triangles: ArrayList(TriangleOutput) = ArrayList(TriangleOutput).init(c_allocator),
        colour_buffer: []u32,
    };

    bins: ArrayList(Bin) = ArrayList(Bin).init(c_allocator),
    rows: u32,
    columns: u32,
    viewport_width: u32,
    viewport_height: u32,

    pub fn init(viewport_width: u32, viewport_height: u32) !TriangleBins {
        // Split viewport into 64x64 tiles
        // If it the width or height isn't a multiple of 64 then either the last
        // 1 or 2 rows/columns will be a different size (<= 64)

        var rows: u32 = (viewport_height + 63) / 64;
        var cols: u32 = (viewport_width + 63) / 64;

        var self = TriangleBins{
            .rows = rows,
            .columns = cols,
            .viewport_width = viewport_width,
            .viewport_height = viewport_height,
        };

        errdefer {
            for (self.bins.items) |*b| {
                c_allocator.free(b.colour_buffer);
            }
        }

        var last_row_height: u32 = 0;
        var penultimate_row_height: u32 = 64;

        if (viewport_height % 64 >= 32) {
            last_row_height = viewport_height % 64;
        } else if (viewport_height % 64 > 0) {
            const combined_size = 64 + viewport_height % 64;
            const half = combined_size / 2;
            last_row_height = half;
            penultimate_row_height = combined_size - half;
        }

        var last_col_width: u32 = 64;
        var penultimate_col_width: u32 = 64;

        if (viewport_width % 64 >= 32) {
            last_col_width = viewport_width % 64;
        } else if (viewport_width % 64 > 0) {
            const combined_size = 64 + viewport_width % 64;
            const half = combined_size / 2;
            last_col_width = half;
            penultimate_col_width = combined_size - half;
        }

        var tile_y: u32 = 0;
        var y: u32 = 0;
        while (y < rows) : (y += 1) {
            var h: u32 = if (y == rows - 1) last_row_height else if (y == rows - 2) penultimate_row_height else 64;

            var tile_x: u32 = 0;
            var x: u32 = 0;
            while (x < cols) : (x += 1) {
                var w: u32 = if (x == cols - 1) last_col_width else if (x == cols - 2) penultimate_col_width else 64;

                var colour_buffer = try c_allocator.alloc(u32, w * h);
                errdefer c_allocator.free(colour_buffer);

                try self.bins.append(.{
                    .x = tile_x,
                    .y = tile_y,
                    .w = w,
                    .h = h,
                    .x_f32 = @intToFloat(f32, tile_x),
                    .y_f32 = @intToFloat(f32, tile_y),
                    .w_f32 = @intToFloat(f32, w),
                    .h_f32 = @intToFloat(f32, h),
                    .right_f32 = @intToFloat(f32, tile_x + w),
                    .bottom_f32 = @intToFloat(f32, tile_y + h),
                    .colour_buffer = colour_buffer,
                });

                tile_x += w;
            }

            tile_y += h;
        }

        return self;
    }

    pub fn deinit(self: *TriangleBins) void {
        for (self.bins.items) |*b| {
            b.triangles.deinit();
            c_allocator.free(b.colour_buffer);
        }
        self.bins.deinit();
    }

    pub fn writePNG(self: *TriangleBins, file_path: [*:0]const u8) !void {
        var image = try c_allocator.alloc(u32, self.viewport_width * self.viewport_height);
        defer c_allocator.free(image);

        for (self.bins.items) |*b| {
            var y: u32 = 0;
            while (y < b.h) : (y += 1) {
                var dst_idx = ((self.viewport_height - (b.y + y)) - 1) * self.viewport_width + b.x;
                std.mem.copy(
                    u32,
                    image[dst_idx .. dst_idx + b.w],
                    b.colour_buffer[y * b.w .. (y + 1) * b.w],
                );
            }
        }

        try png.writePNG(file_path, image, self.viewport_width, self.viewport_height);
    }
};
