const std = @import("std");
const c = @cImport({
    @cInclude("stb_image_write.h");
});

pub fn writePNG(file_path: [*:0]const u8, image: []const u32, w: u32, h: u32) !void {
    std.debug.assert(image.len == w * h);

    const err = c.stbi_write_png(
        file_path,
        @intCast(c_int, w),
        @intCast(c_int, h),
        4,
        image.ptr,
        @intCast(c_int, w * 4),
    );
    if (err == 0) {
        return error.PNGWriteError;
    }
}
