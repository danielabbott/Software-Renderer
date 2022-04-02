const std = @import("std");
const ArrayList = std.ArrayList;
const c_allocator = std.heap.c_allocator;
const TriangleBins = @import("bins.zig").TriangleBins;
const Vector = @import("maths/vector.zig").Vector;
const AlignedVector = @import("maths/vector.zig").AlignedVector;
const toRadians = @import("maths/maths.zig").toRadians;
const colour = @import("colour.zig");

const srgb_correct = true;

pub fn processFragments(bins: *TriangleBins, viewport_width: u32, viewport_height: u32, threads_count: u32) !void {
    var threads = ArrayList(std.Thread).init(c_allocator);
    defer threads.deinit();

    if (threads_count > 1) {
        try threads.ensureUnusedCapacity(threads_count - 1);
    }

    defer {
        for (threads.items) |*t| {
            t.join();
        }
    }

    var fail_flag = std.atomic.Atomic(bool).init(false);

    var i: u32 = 1;
    while (i < threads_count) : (i += 1) {
        var args = try c_allocator.create(ThreadArgs);

        args.* = .{
            .bins = bins,
            .viewport_width = viewport_width,
            .viewport_height = viewport_height,
            .thread_index = @intCast(u32, i),
            .threads_count = threads_count,
            .fail_flag = &fail_flag,
        };

        var t = std.Thread.spawn(
            .{ .stack_size = 1024 * 1024 },
            process,
            .{args},
        ) catch |e| {
            c_allocator.destroy(args);
            return e;
        };

        threads.appendAssumeCapacity(t);
    }

    var args = try c_allocator.create(ThreadArgs);
    args.* = .{
        .bins = bins,
        .viewport_width = viewport_width,
        .viewport_height = viewport_height,
        .thread_index = 0,
        .threads_count = threads_count,
        .fail_flag = &fail_flag,
    };
    try process(args);

    if (fail_flag.load(.SeqCst)) {
        return error.FragmentError;
    }
}

fn area(v0: [2]f32, v1: [2]f32, v2: [2]f32) f32 {
    const A_x = v1[0] - v0[0];
    const A_y = v1[1] - v0[1];
    const B_x = v2[0] - v0[0];
    const B_y = v2[1] - v0[1];

    return std.math.absFloat(0.5 * (A_x * B_y - A_y * B_x));
}

fn pixelIsWithinTriangle(positions: [3][2]f32, y: u32, x: u32) bool {
    const p = [2]f32{ @intToFloat(f32, x), @intToFloat(f32, y) };
    var line_side_counter: i32 = 0;

    var j: u32 = 0;
    while (j < 3) : (j += 1) {
        const k = (j + 1) % 3;

        const side = (positions[k][0] - positions[j][0]) * (p[1] - positions[j][1]) >
            (positions[k][1] - positions[j][1]) * (p[0] - positions[j][0]);

        line_side_counter += @intCast(i32, @boolToInt(side)) * 2 - 1;
    }

    // point is on same side of all 3 lines, so in the triangle
    return line_side_counter == 3 or line_side_counter == -3;
}

const ThreadArgs = struct {
    viewport_width: u32,
    viewport_height: u32,
    bins: *TriangleBins,
    thread_index: u32,
    threads_count: u32,
    fail_flag: *std.atomic.Atomic(bool),
};

fn process(args: *ThreadArgs) !void {
    process_(args) catch |e| {
        args.fail_flag.*.store(true, .SeqCst);
        return e;
    };
}

fn process_(args: *ThreadArgs) !void {
    defer c_allocator.destroy(args);

    // Transitive depth buffer
    var depth_buffer = try c_allocator.alloc(f32, 64 * 64);
    defer c_allocator.free(depth_buffer);

    for (args.bins.bins.items) |*bin, i| {
        if (@intCast(u32, i) % args.threads_count == args.thread_index) {
            try doFragmentProcessing(bin, depth_buffer);
        }
    }
}

fn doFragmentProcessing(
    bin: *TriangleBins.Bin,
    depth_buffer: []f32,
) !void {
    std.debug.assert(bin.w >= 0 and bin.w <= 64);
    std.debug.assert(bin.h >= 0 and bin.h <= 64);
    std.debug.assert(depth_buffer.len == 64 * 64);

    std.mem.set(u32, bin.colour_buffer, 0xfff03030);
    std.mem.set(f32, depth_buffer, 0.0);

    for (bin.triangles.items) |tri| {
        var positions = [3][2]f32{
            [2]f32{
                tri[0].position_pixel_space[0],
                tri[0].position_pixel_space[1],
            },
            [2]f32{
                tri[1].position_pixel_space[0],
                tri[1].position_pixel_space[1],
            },
            [2]f32{
                tri[2].position_pixel_space[0],
                tri[2].position_pixel_space[1],
            },
        };

        // Find bounding box of triangle within this tile

        var smallest_x_f: f32 = 9999999.0;
        var biggest_x_f: f32 = -9999999.0;
        var smallest_y_f: f32 = 9999999.0;
        var biggest_y_f: f32 = -9999999.0;
        {
            var i: u32 = 0;
            while (i < 3) : (i += 1) {
                smallest_x_f = if (positions[i][0] < smallest_x_f) positions[i][0] else smallest_x_f;
                smallest_y_f = if (positions[i][1] < smallest_y_f) positions[i][1] else smallest_y_f;
                biggest_x_f = if (positions[i][0] > biggest_x_f) positions[i][0] else biggest_x_f;
                biggest_y_f = if (positions[i][1] > biggest_y_f) positions[i][1] else biggest_y_f;
            }

            smallest_x_f = std.math.max(smallest_x_f, 0.0);
            smallest_y_f = std.math.max(smallest_y_f, 0.0);
            biggest_x_f = std.math.max(biggest_x_f, 0.0);
            biggest_y_f = std.math.max(biggest_y_f, 0.0);
        }

        var smallest_x = @floatToInt(u32, smallest_x_f);
        var smallest_y = @floatToInt(u32, smallest_y_f);
        var biggest_x = 1 + @floatToInt(u32, biggest_x_f);
        var biggest_y = 1 + @floatToInt(u32, biggest_y_f);

        smallest_x = if (smallest_x < bin.x) bin.x else smallest_x;
        smallest_y = if (smallest_y < bin.y) bin.y else smallest_y;
        biggest_x = if (biggest_x >= bin.x + bin.w) bin.x + bin.w - 1 else biggest_x;
        biggest_y = if (biggest_y >= bin.y + bin.h) bin.y + bin.h - 1 else biggest_y;

        // put variables in tile pixel coordinates

        smallest_x -= bin.x;
        smallest_y -= bin.y;
        biggest_x -= bin.x;
        biggest_y -= bin.y;

        positions[0][0] -= @intToFloat(f32, bin.x);
        positions[0][1] -= @intToFloat(f32, bin.y);
        positions[1][0] -= @intToFloat(f32, bin.x);
        positions[1][1] -= @intToFloat(f32, bin.y);
        positions[2][0] -= @intToFloat(f32, bin.x);
        positions[2][1] -= @intToFloat(f32, bin.y);

        //

        const v0 = positions[0];
        const v1 = positions[1];
        const v2 = positions[2];

        const triangle_area_r = 1.0 / area(v0, v1, v2);

        const texture_width = @intToFloat(f32, tri[0].texture.width);
        const texture_height = @intToFloat(f32, tri[0].texture.height);

        // Iterator variables - direction swaps each scanline

        var x_scan_start: i32 = @intCast(i32, smallest_x);
        var x_scan_end: i32 = @intCast(i32, biggest_x);
        var x_scan_incr: i32 = 1;

        var y: u32 = smallest_y;
        while (y <= biggest_y) : (y += 1) {
            var seen_coverage = false; // set to true when first triangle pixel is seen on this scanline

            var x: i32 = x_scan_start;
            while (x != x_scan_end + x_scan_incr) : (x += x_scan_incr) {
                if (!pixelIsWithinTriangle(positions, y, @intCast(u32, x))) {
                    if (seen_coverage) {
                        // already seen triangle on this scanline, rest of scanline is empty
                        break;
                    } else {
                        // keep going until triangle is found
                        continue;
                    }
                }
                seen_coverage = true;

                // interpolate parameters (non-perspective correct)

                const p = [2]f32{ @intToFloat(f32, x), @intToFloat(f32, y) };

                var weight0 = area(p, v1, v2) * triangle_area_r;
                var weight1 = area(p, v0, v2) * triangle_area_r;
                var weight2 = area(p, v0, v1) * triangle_area_r;

                // depth (perspective correct interpolation & depth buffer)

                weight0 *= tri[0].reciprocal_w;
                weight1 *= tri[1].reciprocal_w;
                weight2 *= tri[2].reciprocal_w;

                const weight_total_r = 1.0 / (weight0 + weight1 + weight2);

                weight0 *= weight_total_r;
                weight1 *= weight_total_r;
                weight2 *= weight_total_r;

                const depth =
                    weight0 * tri[0].depth +
                    weight1 * tri[1].depth +
                    weight2 * tri[2].depth;

                var depth_buffer_value = &depth_buffer[y * bin.w + @intCast(u32, x)];
                if (depth_buffer_value.* > depth) {
                    continue;
                }
                depth_buffer_value.* = depth;

                // texture

                var uv = [2]f32{
                    weight0 * tri[0].uv[0] +
                        weight1 * tri[1].uv[0] +
                        weight2 * tri[2].uv[0],
                    weight0 * tri[0].uv[1] +
                        weight1 * tri[1].uv[1] +
                        weight2 * tri[2].uv[1],
                };

                uv[0] = std.math.clamp(uv[0], 0.0, 1.0);
                uv[1] = std.math.clamp(uv[1], 0.0, 1.0);

                const uv_pixel_coords = [2]u32{
                    @floatToInt(u32, uv[0] * texture_width),
                    @floatToInt(u32, uv[1] * texture_height),
                };

                const texture_srgb = tri[0].texture.colours[uv_pixel_coords[1] * tri[0].texture.width + uv_pixel_coords[0]];

                var texture_linear: [3]f32 = undefined;

                if (srgb_correct) {
                    colour.sRGBToLinear3(texture_srgb, texture_linear[0..3]);
                } else {
                    colour.u32ToF32_3(texture_srgb, texture_linear[0..3]);
                }

                // lighting

                const light =
                    (tri[0].light * weight0) +
                    (tri[1].light * weight1) +
                    (tri[2].light * weight2);

                var final_colour = texture_linear;
                final_colour[0] *= light;
                final_colour[1] *= light;
                final_colour[2] *= light;

                if (srgb_correct) {
                    bin.colour_buffer[y * bin.w + @intCast(u32, x)] = colour.linearToSRGB3(final_colour);
                } else {
                    bin.colour_buffer[y * bin.w + @intCast(u32, x)] = colour.f32ToU32_3(final_colour);
                }
            }

            // swap direction
            var tmp = x_scan_start;
            x_scan_start = x_scan_end;
            x_scan_end = tmp;
            x_scan_incr *= -1;
        }
    }
}
