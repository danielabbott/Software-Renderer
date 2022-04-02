const std = @import("std");
const ArrayList = std.ArrayList;
const c_allocator = std.heap.c_allocator;
const Matrix = @import("maths/matrix.zig").Matrix;
const scene = @import("scene.zig");
const TriangleBins = @import("bins.zig").TriangleBins;
const processVertices = @import("vertex_process.zig").processVertices;
const processFragments = @import("fragment_process.zig").processFragments;

const benchmark = false;
const benchmark_n: u32 = if (benchmark) 300 else 1;

pub fn main() anyerror!void {
    const viewport_width = 640;
    const viewport_height = 480;

    std.log.info("Rendering at {}x{} resolution", .{ viewport_width, viewport_height });

    var test_scene = try scene.createScene(viewport_width, viewport_height);
    defer test_scene.deinit();

    const cpu_count = if (benchmark) 1 else std.Thread.getCpuCount() catch 2;

    var triangle_bins = try TriangleBins.init(viewport_width, viewport_height);
    defer triangle_bins.deinit();

    asm volatile ("" ::: "memory");
    const t0 = std.time.milliTimestamp();
    asm volatile ("" ::: "memory");

    {
        var i: u32 = 0;
        while (i < benchmark_n) : (i += 1) {
            try processVertices(&triangle_bins, &test_scene);
        }
    }

    asm volatile ("" ::: "memory");
    const t1 = std.time.milliTimestamp();
    asm volatile ("" ::: "memory");

    {
        var i: u32 = 0;
        while (i < benchmark_n) : (i += 1) {
            try processFragments(&triangle_bins, viewport_width, viewport_height, @intCast(u32, cpu_count));
        }
    }

    asm volatile ("" ::: "memory");
    const t2 = std.time.milliTimestamp();
    asm volatile ("" ::: "memory");

    try triangle_bins.writePNG("out.png");

    asm volatile ("" ::: "memory");
    const t3 = std.time.milliTimestamp();
    asm volatile ("" ::: "memory");

    if (benchmark) {
        std.debug.print("Vertex processing took {}ms {d:.2}ms\n", .{
            t1 - t0,
            @intToFloat(f32, t1 - t0) / @intToFloat(f32, benchmark_n),
        });
        std.debug.print("Fragment processing took {}ms {d:.2}ms\n", .{
            t2 - t1,
            @intToFloat(f32, t2 - t1) / @intToFloat(f32, benchmark_n),
        });
    } else {
        std.debug.print("Vertex processing took {}ms\n", .{t1 - t0});
        std.debug.print("Fragment processing took {}ms\n", .{t2 - t1});
    }
    std.debug.print("PNG creation took {}ms\n", .{t3 - t2});
}

test "" {
    _ = @import("maths/maths.zig");
}
