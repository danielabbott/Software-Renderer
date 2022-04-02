const std = @import("std");
const print = std.debug.print;
const assert = std.debug.assert;
const vector = @import("vector.zig");

const testing_tolerance = std.math.sqrt(std.math.epsilon(f32));

fn Vector(comptime T: type, comptime len: u32) type {
    return @Vector(len, T);
}

pub fn Matrix(comptime T: type, comptime S: u32) type {
    if (@typeInfo(T) != .Float) {
        @compileError("Matrices must use float types");
    }
    if (S < 2) {
        @compileError("Matrices must be at least 2x2");
    }

    return struct {
        const Self = @This();

        data: [S]@Vector(S, T),

        // Values in column-major layout
        pub fn init(values: [S][S]T) Self {
            comptime {
                if (S < 2) {
                    @compileError("Matrices must be at least 2x2");
                }
            }

            var a: Self = undefined;
            var i: u32 = 0;
            while (i < S) : (i += 1) {
                var j: u32 = 0;
                while (j < S) : (j += 1) {
                    a.data[j][i] = values[i][j];
                }
            }
            return a;
        }

        pub fn zero() Self {
            return std.mem.zeroes(Self);
        }

        pub fn loadFromSlice(self: *Self, slice: []const T) !void {
            if (slice.len != S * S) {
                assert(false);
                return error.InvalidSliceLength;
            }

            const V = @Vector(S, T);
            std.mem.copy(V, @ptrCast([*]V, &self.data[0])[0..S], @ptrCast([*]V, &slice[0])[0..S]);
        }

        pub fn identity() Self {
            comptime var m: Self = undefined;

            comptime {
                var i: u32 = 0;
                while (i < S) : (i += 1) {
                    var j: u32 = 0;
                    while (j < S) : (j += 1) {
                        m.data[i][j] = if (i == j) 1 else 0;
                    }
                }
            }

            return m;
        }

        pub fn mul(a: Self, b: Self) Self {
            var new: Self = undefined;

            if (S == 4) {
                comptime var row: u32 = 0;
                inline while (row < 4) : (row += 1) {
                    const vx = @splat(4, a.data[row][0]);
                    const vy = @splat(4, a.data[row][1]);
                    const vz = @splat(4, a.data[row][2]);
                    const vw = @splat(4, a.data[row][3]);
                    new.data[row] =
                        vx * b.data[0] +
                        vy * b.data[1] +
                        vz * b.data[2] +
                        vw * b.data[3];
                }
            } else {
                var i: u32 = 0;
                while (i < S) : (i += 1) {
                    var j: u32 = 0;
                    while (j < S) : (j += 1) {
                        var sum: T = 0;
                        var k: u32 = 0;
                        while (k < S) : (k += 1) {
                            sum += a.data[i][k] * b.data[k][j];
                        }
                        new.data[i][j] = sum;
                    }
                }
            }

            return new;
        }

        pub fn translate(v: Vector(T, S - 1)) Self {
            comptime {
                if (S != 3 and S != 4) {
                    @compileError("Translate is only for 3x3 matrices (2D) and 4x4 matrices (3D).");
                }
            }

            var m: Self = comptime Self.identity();

            if (S == 4) {
                m.data[3][0] = v[0];
                m.data[3][1] = v[1];
                m.data[3][2] = v[2];
            } else if (S == 3) {
                m.data[2][0] = v[0];
                m.data[2][1] = v[1];
            }

            return m;
        }

        pub fn scale(v: Vector(T, S)) Self {
            comptime {
                if (S != 2 and S != 3 and S != 4) {
                    @compileError("Scale is only for 2x2 matrices (2D) 3x3 matrices (2D/3D) and 4x4 matrices (3D).");
                }
            }

            var m: Self = comptime Self.identity();

            if (S == 4) {
                m.data[0][0] = v[0];
                m.data[1][1] = v[1];
                m.data[2][2] = v[2];
                m.data[3][3] = v[3];
            } else if (S == 3) {
                m.data[0][0] = v[0];
                m.data[1][1] = v[1];
                m.data[2][2] = v[2];
            } else if (S == 2) {
                m.data[0][0] = v[0];
                m.data[1][1] = v[1];
            }

            return m;
        }

        // https://en.wikipedia.org/wiki/Rotation_matrix#Basic_rotations

        pub fn rotateX(angle: T) Self {
            comptime {
                if (S != 3 and S != 4) {
                    @compileError("Rotation about an axis is only for 3x3 matrices and 4x4 matrices.");
                }
            }

            var m: Self = comptime Self.identity();

            const sinTheta = std.math.sin(angle);
            const cosTheta = std.math.cos(angle);

            m.data[1][1] = cosTheta;
            m.data[1][2] = -sinTheta;
            m.data[2][2] = cosTheta;
            m.data[2][1] = sinTheta;

            return m;
        }

        pub fn rotateY(angle: T) Self {
            comptime {
                if (S != 3 and S != 4) {
                    @compileError("Rotation about an axis is only for 3x3 matrices and 4x4 matrices.");
                }
            }

            var m: Self = comptime Self.identity();

            const sinTheta = std.math.sin(angle);
            const cosTheta = std.math.cos(angle);

            m.data[0][0] = cosTheta;
            m.data[2][0] = sinTheta;
            m.data[0][2] = -sinTheta;
            m.data[2][2] = cosTheta;

            return m;
        }

        pub fn rotateZ(angle: T) Self {
            if (S != 3 and S != 4) {
                @compileError("Rotation about an axis is only for 3x3 matrices and 4x4 matrices.");
            }

            var m: Self = comptime Self.identity();

            const sinTheta = std.math.sin(angle);
            const cosTheta = std.math.cos(angle);

            m.data[0][0] = cosTheta;
            m.data[1][0] = -sinTheta;
            m.data[0][1] = sinTheta;
            m.data[1][1] = cosTheta;

            return m;
        }

        pub fn transpose(self: Self) Self {
            var new: Self = undefined;
            var i: u32 = 0;
            while (i < S) : (i += 1) {
                var j: u32 = 0;
                while (j < S) : (j += 1) {
                    new[j][i] = self.data[i][j];
                }
            }
            return new;
        }

        fn determinant_3x3(a: T, b: T, c: T, d: T, e: T, f: T, g: T, h: T, i: T) T {
            return a * e * i + b * f * g + c * d * h - c * e * g - b * d * i - a * f * h;
        }

        pub fn determinant(self: Self) T {
            if (S == 2) {
                return self.data[0][0] * self.data[1][1] - self.data[1][0] * self.data[0][1];
            } else if (S == 3) {
                const a = self.data[0][0];
                const b = self.data[1][0];
                const c = self.data[2][0];
                const d = self.data[0][1];
                const e = self.data[1][1];
                const f = self.data[2][1];
                const g = self.data[0][2];
                const h = self.data[1][2];
                const i = self.data[2][2];
                return determinant_3x3(a, b, c, d, e, f, g, h, i);
            } else if (S == 4) {
                const a = self.data[0][0];
                const b = self.data[1][0];
                const c = self.data[2][0];
                const d = self.data[3][0];
                const e = self.data[0][1];
                const f = self.data[1][1];
                const g = self.data[2][1];
                const h = self.data[3][1];
                const i = self.data[0][2];
                const j = self.data[1][2];
                const k = self.data[2][2];
                const l = self.data[3][2];
                const m = self.data[0][3];
                const n = self.data[1][3];
                const o = self.data[2][3];
                const p = self.data[3][3];

                return a * determinant_3x3(f, g, h, j, k, l, n, o, p) - b * determinant_3x3(e, g, h, i, k, l, m, o, p) + c * determinant_3x3(e, f, h, i, j, l, m, n, p) - d * determinant_3x3(e, f, g, i, j, k, m, n, o);
            } else {
                var i: u32 = 0;
                var sign: T = 1;
                var sum: T = 0;
                while (i < S) : (i += 1) {
                    sum += sign * self.data[i][0] * self.subMatDet(i, 0);

                    sign *= -1;
                }
                return sum;
            }
        }

        fn subMatDet(self: Self, i_skip: u32, j_skip: u32) T {
            var m: Matrix(T, S - 1) = undefined;

            var i: u32 = 0;
            while (i < S) : (i += 1) {
                var j: u32 = 0;
                while (j < S) : (j += 1) {
                    if (i != i_skip and j != j_skip) {
                        var i_: u32 = i;
                        var j_: u32 = j;

                        if (i_ > i_skip) {
                            i_ -= 1;
                        }

                        if (j_ > j_skip) {
                            j_ -= 1;
                        }

                        m.data[i_][j_] = self.data[i][j];
                    }
                }
            }

            return m.determinant();
        }

        pub fn inverse(self: Self) !Self {
            if (S == 2) {
                const det = self.determinant();

                if (det == 0) {
                    return error.NoInverse;
                }

                const determinant_recipr = 1.0 / det;

                const a = self.data[0][0];
                const b = self.data[1][0];
                const c = self.data[0][1];
                const d = self.data[1][1];

                return Matrix(T, 2).init([2][2]T{
                    [2]f32{ determinant_recipr * d, determinant_recipr * -b },
                    [2]f32{ determinant_recipr * -c, determinant_recipr * a },
                });
            } else if (S == 3) {
                const det = self.determinant();

                if (det == 0) {
                    return error.NoInverse;
                }

                const determinant_recipr = 1.0 / det;

                const a = self.data[0][0];
                const b = self.data[1][0];
                const c = self.data[2][0];
                const d = self.data[0][1];
                const e = self.data[1][1];
                const f = self.data[2][1];
                const g = self.data[0][2];
                const h = self.data[1][2];
                const i = self.data[2][2];

                const A_ = e * i - f * h;
                const B_ = -(d * i - f * g);
                const C_ = d * h - e * g;
                const D_ = -(b * i - c * h);
                const E_ = a * i - c * g;
                const F_ = -(a * h - b * g);
                const G_ = b * f - c * e;
                const H_ = -(a * f - c * d);
                const I_ = a * e - b * d;

                return Matrix(f32, 3).init([3][3]f32{
                    [3]f32{ determinant_recipr * A_, determinant_recipr * D_, determinant_recipr * G_ },
                    [3]f32{ determinant_recipr * B_, determinant_recipr * E_, determinant_recipr * H_ },
                    [3]f32{ determinant_recipr * C_, determinant_recipr * F_, determinant_recipr * I_ },
                });
            } else {
                const det = self.determinant();

                if (det == 0) {
                    return error.NoInverse;
                }

                const determinant_recipr = 1.0 / det;
                var result: Self = undefined;

                var i: u32 = 0;
                var sign: T = 1;
                while (i < S) : (i += 1) {
                    var j: u32 = 0;
                    while (j < S) : (j += 1) {
                        var co: T = self.subMatDet(i, j);

                        result.data[j][i] = sign * determinant_recipr * co;
                        sign *= -1;
                    }
                    sign *= -1;
                }

                return result;
            }
        }

        // aspect_ratio = window_width / window_height
        // fov is in radians
        pub fn perspectiveProjection(aspect_ratio: T, fovy: T, near_z: T, far_z: T) Matrix(T, 4) {
            var m: Matrix(T, 4) = Matrix(T, 4).zero();

            // Code borrowed from cGLM

            const f = 1.0 / std.math.tan(fovy * 0.5);
            const f2 = 1.0 / (near_z - far_z);

            m.data[0][0] = f / aspect_ratio;
            m.data[1][1] = -f;
            m.data[2][2] = far_z * f2;
            m.data[2][3] = -1.0;
            m.data[3][2] = near_z * far_z * f2;

            return m;
        }

        pub fn orthoProjection(left: f32, right: f32, bottom: f32, top: f32, nearZ: f32, farZ: f32) Matrix(T, 4) {
            var m: Matrix(T, 4) = Matrix(T, 4).zero();

            // Code borrowed from cGLM

            const rl = 1.0 / (right - left);
            const tb = 1.0 / (top - bottom);
            const fn_ = -1.0 / (farZ - nearZ);

            m.data[0][0] = 2.0 * rl;
            m.data[1][1] = 2.0 * tb;
            m.data[2][2] = fn_;
            m.data[3][0] = -(right + left) * rl;
            m.data[3][1] = -(top + bottom) * tb;
            m.data[3][2] = nearZ * fn_;
            m.data[3][3] = 1.0;

            return m;
        }

        pub fn position3D(self: Self) Vector(T, 3) {
            if (S == 4) {
                return Vector(T, 3){ self.data[3][0], self.data[3][1], self.data[3][2] };
            } else {
                @compileError("Matrix.position3D is only for 4x4 matrices.");
            }
        }

        pub fn position2D(self: Self) Vector(T, S) {
            if (S == 3) {
                return Vector(T, 2){ self.data[2][0], self.data[2][1] };
            } else {
                @compileError("Matrix.position2D is only for 3x3 matrices.");
            }
        }

        pub fn equalTo(self: Self, b: Self) bool {
            var a_slice = @ptrCast([*]const f32, &self.data[0][0])[0 .. S * S];
            var b_slice = @ptrCast([*]const f32, &b.data[0][0])[0 .. S * S];

            return std.mem.eql(f32, a_slice, b_slice);
        }
    };
}

test "Multiply matrix by identity" {
    var m: Matrix(f32, 2) = Matrix(f32, 2).identity();
    try std.testing.expectEqual(m.data[0][0], 1.0);
    try std.testing.expectEqual(m.data[1][1], 1.0);

    var m2: Matrix(f32, 2) = Matrix(f32, 2).init([2][2]f32{
        [2]f32{ 1, 2 },
        [2]f32{ 5, 6 },
    });

    var m3: Matrix(f32, 2) = m2.mul(m);

    try std.testing.expectApproxEqRel(m3.data[0][0], 1.0, testing_tolerance);
    try std.testing.expectApproxEqRel(m3.data[0][1], 5.0, testing_tolerance);
    try std.testing.expectApproxEqRel(m3.data[1][0], 2.0, testing_tolerance);
    try std.testing.expectApproxEqRel(m3.data[1][1], 6.0, testing_tolerance);
}

test "Multiply vec2 by mat2" {
    var m: Matrix(f32, 2) = Matrix(f32, 2).identity();
    var v1: Vector(f32, 2) = Vector(f32, 2){ 1.0, 2.0 };
    var v2: Vector(f32, 2) = vector.mulMat(v1, m);

    try std.testing.expectApproxEqRel(v2[0], 1.0, testing_tolerance);
    try std.testing.expectApproxEqRel(v2[1], 2.0, testing_tolerance);
}

test "Multiply vec4 by mat4" {
    var m: Matrix(f32, 4) = Matrix(f32, 4).identity();
    m.data[3][0] = 1.0;
    m.data[3][1] = 2.0;
    m.data[3][2] = 3.0;

    var v1: Vector(f32, 4) = Vector(f32, 4){ 0.0, 0.0, 0.0, 1.0 };
    var v2: Vector(f32, 4) = vector.mulMat(v1, m);

    try std.testing.expectApproxEqRel(v2[0], 1.0, testing_tolerance);
    try std.testing.expectApproxEqRel(v2[1], 2.0, testing_tolerance);
    try std.testing.expectApproxEqRel(v2[2], 3.0, testing_tolerance);
    try std.testing.expectApproxEqRel(v2[3], 1.0, testing_tolerance);
}

test "Translate vec2 by mat3" {
    var m: Matrix(f32, 3) = Matrix(f32, 3).identity();
    var v1: Vector(f32, 2) = Vector(f32, 2){ 1.0, 2.0 };
    m = m.mul(Matrix(f32, 3).translate(v1));

    var v2: Vector(f32, 3) = Vector(f32, 3){ 0.0, 0.0, 1.0 };
    v2 = vector.mulMat(v2, m);

    try std.testing.expectApproxEqRel(v2[0], 1.0, testing_tolerance);
    try std.testing.expectApproxEqRel(v2[1], 2.0, testing_tolerance);
}

test "Rotate vec4 about x axis" {
    // Rotate 45 degrees
    var m: Matrix(f32, 4) = Matrix(f32, 4).identity().mul(Matrix(f32, 4).rotateX(0.7853981625));

    var v1: Vector(f32, 4) = Vector(f32, 4){ 0.0, 1.0, 0.0, 1.0 };
    v1 = vector.mulMat(v1, m);

    try std.testing.expectApproxEqRel(v1[0], 0.0, testing_tolerance);
    try std.testing.expectApproxEqRel(v1[1], 0.7071067811865475, testing_tolerance);
    try std.testing.expectApproxEqRel(v1[2], -0.7071067811865475, testing_tolerance);
    try std.testing.expectApproxEqRel(v1[3], 1.0, testing_tolerance);
}

test "Rotate vec3 about y axis" {
    // Rotate 45 degrees
    var m: Matrix(f32, 3) = Matrix(f32, 3).identity().mul(Matrix(f32, 3).rotateY(0.7853981625));

    var v1: Vector(f32, 3) = Vector(f32, 3){ 1.0, 0.0, 0.0 };
    v1 = vector.mulMat(v1, m);

    try std.testing.expectApproxEqRel(v1[0], 0.7071067811865475, testing_tolerance);
    try std.testing.expectApproxEqRel(v1[1], 0.0, testing_tolerance);
    try std.testing.expectApproxEqRel(v1[2], -0.7071067811865475, testing_tolerance);
}

test "Rotate vec3 about z axis" {
    // Rotate 45 degrees
    var m: Matrix(f32, 3) = Matrix(f32, 3).identity().mul(Matrix(f32, 3).rotateZ(0.7853981625));

    var v1: Vector(f32, 3) = Vector(f32, 3){ 1.0, 0.0, 0.0 };
    v1 = vector.mulMat(v1, m);

    try std.testing.expectApproxEqRel(v1[0], 0.7071067811865475, testing_tolerance);
    try std.testing.expectApproxEqRel(v1[1], 0.7071067811865475, testing_tolerance);
    try std.testing.expectApproxEqRel(v1[2], 0.0, testing_tolerance);
}

test "Transformation matrix" {
    var m: Matrix(f32, 4) = Matrix(f32, 4).identity();
    m = m.mul(Matrix(f32, 4).scale(Vector(f32, 4){ 2.0, 2.0, 2.0, 1.0 }));
    m = m.mul(Matrix(f32, 4).rotateY(0.7853981625));
    m = m.mul(Matrix(f32, 4).translate(Vector(f32, 3){ 0.0, 5.0, 0.0 }));

    var v1: Vector(f32, 4) = Vector(f32, 4){ 0.0, 0.0, 0.0, 1.0 };
    v1 = vector.mulMat(v1, m);

    try std.testing.expectApproxEqRel(v1[0], 0.0, testing_tolerance);
    try std.testing.expectApproxEqRel(v1[1], 5.0, testing_tolerance);
    try std.testing.expectApproxEqRel(v1[2], 0.0, testing_tolerance);
    try std.testing.expectApproxEqRel(v1[3], 1.0, testing_tolerance);

    var v2: Vector(f32, 4) = Vector(f32, 4){ 1.0, 0.0, 0.0, 1.0 };
    v2 = vector.mulMat(v2, m);

    try std.testing.expectApproxEqRel(v2[0], 1.414213562373095, testing_tolerance);
    try std.testing.expectApproxEqRel(v2[1], 5.0, testing_tolerance);
    try std.testing.expectApproxEqRel(v2[2], -1.414213562373095, testing_tolerance);
    try std.testing.expectApproxEqRel(v2[3], 1.0, testing_tolerance);
}

test "Determinant" {
    var m: Matrix(f32, 2) = Matrix(f32, 2).init([2][2]f32{
        [2]f32{ 1, 2 },
        [2]f32{ 3, 4 },
    });
    try std.testing.expectApproxEqRel(m.determinant(), -2.0, testing_tolerance);

    var m2: Matrix(f32, 3) = Matrix(f32, 3).init([3][3]f32{
        [3]f32{ 2, 8, 5 },
        [3]f32{ 8, 6, 4 },
        [3]f32{ 5, 3, 6 },
    });
    try std.testing.expectApproxEqRel(m2.determinant(), -206.0, testing_tolerance);

    var m3: Matrix(f32, 4) = Matrix(f32, 4).init([4][4]f32{
        [4]f32{ 9, 5, 9, 7 },
        [4]f32{ 9, 8, 3, 6 },
        [4]f32{ 4, 8, 5, 2 },
        [4]f32{ 4, 3, 8, 8 },
    });
    try std.testing.expectApproxEqRel(m3.determinant(), 1623.0, testing_tolerance);

    var m4: Matrix(f32, 5) = Matrix(f32, 5).init([5][5]f32{
        [5]f32{ 0, 6, -2, -1, 5 },
        [5]f32{ 0, 0, 0, -9, -7 },
        [5]f32{ 0, 0, 15, 35, 0 },
        [5]f32{ 0, 0, -1, -11, -2 },
        [5]f32{ 1, -2, -2, 3, -2 },
    });
    try std.testing.expectApproxEqRel(m4.determinant(), 3840.0, testing_tolerance);
}

test "Inverse" {
    var m: Matrix(f32, 2) = Matrix(f32, 2).init([2][2]f32{
        [2]f32{ 1, 2 },
        [2]f32{ 3, 4 },
    });
    const m_ = try m.inverse();
    try std.testing.expectApproxEqRel(m_.data[0][0], -2.0, testing_tolerance);
    try std.testing.expectApproxEqRel(m_.data[0][1], 3.0 / 2.0, testing_tolerance);
    try std.testing.expectApproxEqRel(m_.data[1][0], 1.0, testing_tolerance);
    try std.testing.expectApproxEqRel(m_.data[1][1], -1.0 / 2.0, testing_tolerance);

    const m__ = try m_.inverse();
    try std.testing.expect(m__.equalTo(m));

    var m2: Matrix(f32, 3) = Matrix(f32, 3).init([3][3]f32{
        [3]f32{ 2, 8, 5 },
        [3]f32{ 8, 6, 4 },
        [3]f32{ 5, 3, 6 },
    });
    const m2_ = try m2.inverse();
    try std.testing.expectApproxEqRel(m2_.data[0][0], -12.0 / 103.0, testing_tolerance);
    try std.testing.expectApproxEqRel(m2_.data[0][1], 14.0 / 103.0, testing_tolerance);
    try std.testing.expectApproxEqRel(m2_.data[0][2], 3.0 / 103.0, testing_tolerance);
    try std.testing.expectApproxEqRel(m2_.data[1][0], 33.0 / 206.0, testing_tolerance);
    try std.testing.expectApproxEqRel(m2_.data[1][1], 13.0 / 206.0, testing_tolerance);
    try std.testing.expectApproxEqRel(m2_.data[1][2], -17.0 / 103.0, testing_tolerance);
    try std.testing.expectApproxEqRel(m2_.data[2][0], -1.0 / 103.0, testing_tolerance);
    try std.testing.expectApproxEqRel(m2_.data[2][1], -16.0 / 103.0, testing_tolerance);
    try std.testing.expectApproxEqRel(m2_.data[2][2], 26.0 / 103.0, testing_tolerance);

    var m3: Matrix(f32, 4) = Matrix(f32, 4).init([4][4]f32{
        [4]f32{ 1, 2, 1, 1 },
        [4]f32{ 2, 2, 1, 1 },
        [4]f32{ 1, 2, 2, 1 },
        [4]f32{ 1, 1, 1, 2 },
    });

    const m3_ = try m3.inverse();
    try std.testing.expectApproxEqRel(m3_.data[0][0], -1.0, testing_tolerance);
    try std.testing.expectApproxEqRel(m3_.data[0][1], 4.0 / 3.0, testing_tolerance);
    try std.testing.expectApproxEqRel(m3_.data[0][2], -1.0, testing_tolerance);
    try std.testing.expectApproxEqRel(m3_.data[0][3], 1.0 / 3.0, testing_tolerance);
    try std.testing.expectApproxEqRel(m3_.data[1][0], 1.0, testing_tolerance);
    try std.testing.expectApproxEqRel(m3_.data[1][1], -1.0 / 3.0, testing_tolerance);
    try std.testing.expectApproxEqRel(m3_.data[1][2], 0.0, testing_tolerance);
    try std.testing.expectApproxEqRel(m3_.data[1][3], -1.0 / 3.0, testing_tolerance);
    try std.testing.expectApproxEqRel(m3_.data[2][0], 0.0, testing_tolerance);
    try std.testing.expectApproxEqRel(m3_.data[2][1], -1.0 / 3.0, testing_tolerance);
    try std.testing.expectApproxEqRel(m3_.data[2][2], 1.0, testing_tolerance);
    try std.testing.expectApproxEqRel(m3_.data[2][3], -1.0 / 3.0, testing_tolerance);
    try std.testing.expectApproxEqRel(m3_.data[3][0], 0.0, testing_tolerance);
    try std.testing.expectApproxEqRel(m3_.data[3][1], -1.0 / 3.0, testing_tolerance);
    try std.testing.expectApproxEqRel(m3_.data[3][2], 0.0, testing_tolerance);
    try std.testing.expectApproxEqRel(m3_.data[3][3], 2.0 / 3.0, testing_tolerance);

    var m4: Matrix(f32, 4) = Matrix(f32, 4).translate(Vector(f32, 3){ 1.5, 0, 0 });
    try std.testing.expectApproxEqRel(m4.data[3][0], 1.5, testing_tolerance);

    const m4_ = try m4.inverse();

    try std.testing.expectApproxEqRel(m4_.data[3][0], -1.5, testing_tolerance);
}

const Mat4 = Matrix(f32, 4);

extern fn mat4x4f32MulAVX(n: c_uint, a: [*c]const f32, b: [*c]const f32, dst: [*c]f32) void;

/// Ideally align matrices to 64 bytes
pub fn f32Mat4OptimisedMultiplyx8(a: *align(32) const Mat4, b: *align(32) const Mat4, dst: *align(32) Mat4) void {
    if (comptime std.Target.x86.featureSetHas(@import("builtin").cpu.features, .avx)) {
        mat4x4f32MulAVX(1, &a.data[0][0], &b.data[0][0], &dst.data[0][0]);
    } else {
        dst.* = a.mul(b);
    }
}

pub fn f32Mat4OptimisedMultiplyx8Many(a: []align(32) const Mat4, b: []align(32) const Mat4, dst: []align(32) Mat4) void {
    std.debug.assert(a.len == b.len);
    std.debug.assert(b.len == dst.len);

    if (comptime std.Target.x86.featureSetHas(@import("builtin").cpu.features, .avx)) {
        mat4x4f32MulAVX(@intCast(c_uint, a.len), &a[0].data[0][0], &b[0].data[0][0], &dst[0].data[0][0]);
    } else {
        for (a) |_, i| {
            dst[i] = a[i].mul(b[i]);
        }
    }
}

test "Optimised Mat4 Multiply" {
    const a: Mat4 align(32) = Mat4.init([4][4]f32{
        [4]f32{ 5, 2, 1, 1 },
        [4]f32{ 2, 3, 12, 1 },
        [4]f32{ 1, 2, 2, 7 },
        [4]f32{ 1, 8, 1, 2 },
    });
    const b align(32) = Mat4.init([4][4]f32{
        [4]f32{ 15, 2, 1, 1 },
        [4]f32{ 2, 3, 88, -5 },
        [4]f32{ 33, 2, 2, 7 },
        [4]f32{ 1, 8, -55, 2 },
    });

    const x = a.mul(b);

    var dst: Mat4 align(32) = undefined;
    f32Mat4OptimisedMultiplyx8(&a, &b, &dst);

    for (x.data) |v, i| {
        var j: u32 = 0;
        while (j < 4) : (j += 1) {
            try std.testing.expectApproxEqRel(v[j], dst.data[i][j], testing_tolerance);
        }
    }
}
