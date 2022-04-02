const vector = @import("vector.zig");
pub const Vector = vector.Vector;
pub const matrix = @import("matrix.zig");
pub const Matrix = matrix.Matrix;

pub const Vec2 = vector.Vec2;
pub const Vec3 = vector.Vec3;
pub const Vec4 = vector.Vec4;
pub const DVec2 = vector.DVec2;
pub const DVec3 = vector.DVec3;
pub const DVec4 = vector.DVec4;

pub const Mat2 = Matrix(f32, 2);
pub const Mat3 = Matrix(f32, 3);
pub const Mat4 = Matrix(f32, 4);

const std = @import("std");

pub fn toRadians(angle: anytype) @TypeOf(angle) {
    return angle * (std.math.pi / 180.0);
}

pub fn toDegrees(angle: anytype) @TypeOf(angle) {
    return angle * (180.0 / std.math.pi);
}

pub fn roundUp(x: anytype, round: @TypeOf(x)) @TypeOf(x) {
    return ((x + round - 1) / round) * round;
}

pub fn mix(a: anytype, b: @TypeOf(a), x: @TypeOf(a)) @TypeOf(a) {
    return x * b + (1.0 - x) * a;
}

const testing_tolerance = std.math.sqrt(std.math.epsilon(f32));

test "Angles" {
    try std.testing.expectApproxEqRel(toRadians(@as(f32, 45.0)), 0.785398, testing_tolerance);
    try std.testing.expectApproxEqRel(toDegrees(@as(f32, 1.0)), 57.2958, testing_tolerance);
}

test "Maths" {
    _ = @import("vector.zig");
    _ = @import("matrix.zig");
}
