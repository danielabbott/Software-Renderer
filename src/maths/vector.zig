const std = @import("std");
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const expectApproxEqRel = std.testing.expectApproxEqRel;
const Matrix = @import("matrix.zig").Matrix;

const testing_tolerance = std.math.sqrt(std.math.epsilon(f32));

pub const Vec2 = @Vector(2, f32);
pub const Vec3 = @Vector(3, f32);
pub const Vec4 = @Vector(4, f32);
pub const DVec2 = @Vector(2, f64);
pub const DVec3 = @Vector(3, f64);
pub const DVec4 = @Vector(4, f64);

pub fn Element(comptime X: type) type {
    return @typeInfo(X).Vector.child;
}
pub fn vecLen(comptime X: type) u32 {
    return @typeInfo(X).Vector.len;
}

/// broadcast scalar into all elements
pub fn init(comptime T: type, value: Element(T)) @Vector(vecLen(T), Element(T)) {
    return @splat(comptime vecLen(T), value);
}

pub fn lengthSquared(a: anytype) Element(@TypeOf(a)) {
    return dot(a, a);
}

pub fn length(a: anytype) Element(@TypeOf(a)) {
    return @sqrt(lengthSquared(a));
}

pub fn normalised(a: anytype) @TypeOf(a) {
    return a * init(@TypeOf(a), 1.0 / length(a));
}

pub fn normalized(a: anytype) @TypeOf(a) {
    return normalised(a);
}

pub fn dot(a: anytype, b: @TypeOf(a)) Element(@TypeOf(a)) {
    const mul = a * b;

    var i: u32 = 0;
    var sum: Element(@TypeOf(a)) = 0;
    while (i < vecLen(@TypeOf(a))) : (i += 1) {
        sum += mul[i];
    }
    return sum;
}

pub fn cross(a: anytype, b: anytype) @TypeOf(a) {
    const l = comptime vecLen(@TypeOf(a));
    if (l < 3) {
        @compileError("Cross product is for 3D (or bigger) vectors only");
    }

    var new: @TypeOf(a) = undefined;
    var i: u32 = 0;
    while (i < l) : (i += 1) {
        new[i] = a[(i + 1) % l] * b[(i + 2) % l] - a[(i + 2) % l] * b[(i + 1) % l];
    }
    return new;
}

// Unit tests in Matrix.zig
pub fn mulMat(a: anytype, m: Matrix(Element(@TypeOf(a)), vecLen(@TypeOf(a)))) @TypeOf(a) {
    var new: @TypeOf(a) = undefined;
    const l = vecLen(@TypeOf(a));
    var i: u32 = 0;
    while (i < l) : (i += 1) {
        var sum: Element(@TypeOf(a)) = 0;
        var j: u32 = 0;
        while (j < l) : (j += 1) {
            sum += m.data[j][i] * a[j];
        }
        new[i] = sum;
    }
    return new;
}

test "Creation" {
    var v1 = init(Vec4, 1.0);
    v1 += Vec4{ 1.0, 0.0, 0.0, 0.0 };

    try expectEqual(v1[0], 2.0);
    try expectEqual(v1[1], 1.0);
    try expectEqual(v1[2], 1.0);
    try expectEqual(v1[3], 1.0);
}

test "Length" {
    const v1 = DVec3{ 1.0, 0.0, 6.0 };

    try expectApproxEqRel(lengthSquared(v1), 37.0, testing_tolerance);
    try expectApproxEqRel(length(v1), 6.0827625303, testing_tolerance);
}

test "Dot Product" {
    const v1 = Vec3{ 1.0, 2.0, 3.0 };
    const v2 = Vec3{ 4.0, 5.0, 6.0 };

    try expectApproxEqRel(dot(v1, v2), 32.0, testing_tolerance);
}

test "Cross Product" {
    const v1 = Vec3{ 1.0, 2.0, 3.0 };
    const v2 = Vec3{ 4.0, 5.0, 6.0 };
    const v3 = cross(v1, v2);

    try expectApproxEqRel(v3[0], -3.0, testing_tolerance);
    try expectApproxEqRel(v3[1], 6.0, testing_tolerance);
    try expectApproxEqRel(v3[2], -3.0, testing_tolerance);
}

test "Divide by scalar" {
    const v1 = Vec4{ 1.0, 2.0, 3.0, 4.0 };
    const v2 = init(Vec4, 2.0);
    const v3 = v1 / v2;

    try expectApproxEqRel(v3[0], 0.5, testing_tolerance);
    try expectApproxEqRel(v3[1], 1.0, testing_tolerance);
    try expectApproxEqRel(v3[2], 1.5, testing_tolerance);
    try expectApproxEqRel(v3[3], 2.0, testing_tolerance);
}
