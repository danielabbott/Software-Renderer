#include <immintrin.h>
#include <assert.h>
#include <stdint.h>

#ifdef __AVX__

// TODO benchmark the latency & throughput of this compared to the Zig SSE version

// ! Pointers must be 32-byte aligned !
void mat4x4f32MulAVX(unsigned int n, const float * restrict a, const float * restrict b, float * restrict dst)
{
    a = __builtin_assume_aligned (a, 32);
    b = __builtin_assume_aligned (b, 32);
    dst = __builtin_assume_aligned (dst, 32);

    for(; n > 0; n--, a += 16, b += 16, dst += 16) {
        const __m256 b0 = _mm256_broadcast_ps((const __m128 *) b);
        const __m256 b1 = _mm256_broadcast_ps((const __m128 *) &b[4]);
        const __m256 b2 = _mm256_broadcast_ps((const __m128 *) &b[8]);
        const __m256 b3 = _mm256_broadcast_ps((const __m128 *) &b[12]);

        #pragma unroll
        for (int row = 0; row < 4; row += 2) {
            const __m128 x = _mm_set_ps1(a[row*4+0]);
            const __m128 x2 = _mm_set_ps1(a[row*4+4]);
            const __m256 vx = _mm256_insertf128_ps(_mm256_castps128_ps256(x),x2,1);
            const __m256 m0 = _mm256_mul_ps(vx, b0);

            const __m128 y = _mm_set_ps1(a[row*4+1]);
            const __m128 y2 = _mm_set_ps1(a[row*4+5]);
            const __m256 vy = _mm256_insertf128_ps(_mm256_castps128_ps256(y),y2,1);
            const __m256 m1 = _mm256_mul_ps(vy, b1);

            const __m128 z = _mm_set_ps1(a[row*4+2]);
            const __m128 z2 = _mm_set_ps1(a[row*4+6]);
            const __m256 vz = _mm256_insertf128_ps(_mm256_castps128_ps256(z),z2,1);
            const __m256 m2 = _mm256_mul_ps(vz, b2);

            const __m128 w = _mm_set_ps1(a[row*4+3]);
            const __m128 w2 = _mm_set_ps1(a[row*4+7]);
            const __m256 vw = _mm256_insertf128_ps(_mm256_castps128_ps256(w),w2,1);
            const __m256 m3 = _mm256_mul_ps(vw, b3);

            const __m256 final = _mm256_add_ps(m0, _mm256_add_ps(m1,
                _mm256_add_ps(m2, m3)));

            _mm256_store_ps(&dst[row*4], final);
        }

    }
}

#endif
