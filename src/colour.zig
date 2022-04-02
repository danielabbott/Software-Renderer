const std = @import("std");
const pow = std.math.pow;

// Quick & inaccurate conversion

pub fn f32ToU32(l: f32) u8 {
    return @floatToInt(u8, std.math.clamp(l, 0.0, 1.0) * 255.0);
}
fn f32ToU32_(l: f32) u32 {
    return f32ToU32(l);
}
pub fn f32ToU32_3(l: [3]f32) u32 {
    return f32ToU32_(l[0]) | (f32ToU32_(l[1]) << 8) | (f32ToU32_(l[2]) << 16) | 0xff000000;
}

pub fn u32ToF32_3(s: u32, l: []f32) void {
    std.debug.assert(l.len == 3);

    l[0] = @intToFloat(f32, s & 0xff) * (1.0 / 255.0);
    l[1] = @intToFloat(f32, (s >> 8) & 0xff) * (1.0 / 255.0);
    l[2] = @intToFloat(f32, (s >> 16) & 0xff) * (1.0 / 255.0);
}

//
// -- srgb <--> linear functions
//

fn getLookupTableSRGBToLinear() [256]f32 {
    comptime {
        @setEvalBranchQuota(20000);

        var a: [256]f32 = undefined;

        var i: u32 = 0;
        while (i < 256) : (i += 1) {
            const s = @intToFloat(f32, i) / 255.0;
            a[i] = if (s < 0.04045)
                s / 12.92
            else
                pow(f32, ((s + 0.055) / 1.055), 2.4);
        }

        return a;
    }
}

fn getLookupTableLinearToSRGB() [1024]u8 {
    comptime {
        @setEvalBranchQuota(50000);

        var a: [1024]u8 = undefined;

        var i: u32 = 0;
        while (i < 1024) : (i += 1) {
            const l = @intToFloat(f32, i) / 1023.0;
            a[i] = @floatToInt(u8, std.math.clamp(if (l < 0.0031308)
                l * 12.92
            else
                1.055 * pow(f32, l, 1.0 / 2.4) - 0.055, 0.0, 1.0) * 255.0);
        }

        return a;
    }
}

const table_srgb_to_linear = getLookupTableSRGBToLinear();
const table_linear_to_srgb = getLookupTableLinearToSRGB();

pub fn linearToSRGB1(l: f32) u8 {
    return table_linear_to_srgb[std.math.clamp(@floatToInt(u32, l * 1023.0), 0, 1023)];
}

fn linearToSRGB1_(l: f32) u32 {
    return linearToSRGB1(l);
}

/// Input: RGBA f32 in linear colour space
/// Output: sRGB_A, alpha is most significant byte, red is least
pub fn linearToSRGB(l: [4]f32) u32 {
    const a = @floatToInt(u32, std.math.clamp(l[3], 0.0, 1.0) * 255.0);
    return linearToSRGB1_(l[0]) | (linearToSRGB1_(l[1]) << 8) | (linearToSRGB1_(l[2]) << 16) | (a << 24);
}

pub fn linearToSRGB3(l: [3]f32) u32 {
    return linearToSRGB1_(l[0]) | (linearToSRGB1_(l[1]) << 8) | (linearToSRGB1_(l[2]) << 16) | 0xff000000;
}

pub fn linearToSRGBGrey(grey: f32) u32 {
    const x = linearToSRGB1_(grey);
    return x | (x << 8) | (x << 16) | 0xff000000;
}

pub fn sRGBToLinear1(s: u8) f32 {
    return table_srgb_to_linear[s];
}

pub fn sRGBToLinear3(s: u32, l: []f32) void {
    std.debug.assert(l.len == 3);

    l[0] = sRGBToLinear1(@intCast(u8, s & 0xff));
    l[1] = sRGBToLinear1(@intCast(u8, (s >> 8) & 0xff));
    l[2] = sRGBToLinear1(@intCast(u8, (s >> 16) & 0xff));
}

pub fn sRGBToLinear(s: u32, l: []f32) void {
    std.debug.assert(l.len == 4);

    sRGBToLinear3(s, l[0..3]);
    l[2] = @intToFloat(f32, (s >> 24) & 0xff) / 255.0;
}
