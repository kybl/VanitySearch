/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

// 8-way (AVX2) SHA-256. Faithful 8-lane port of hash/sha256_sse.cpp.
// The round sequence is identical to the scalar/SSE implementation; only the
// data type widens from __m128i (4 lanes) to __m256i (8 lanes). Input gather
// and output scatter go through memory (negligible next to 64 rounds x 8 lanes)
// which keeps the transpose trivial and the vector code a pure round kernel.

#include "sha256.h"
#include <immintrin.h>
#include <string.h>
#include <stdint.h>

namespace _sha256avx2
{

#define Maj(b,c,d) _mm256_or_si256(_mm256_and_si256(b, c), _mm256_and_si256(d, _mm256_or_si256(b, c)))
#define Ch(b,c,d)  _mm256_xor_si256(_mm256_and_si256(b, c), _mm256_andnot_si256(b, d))
#define ROR(x,n)   _mm256_or_si256(_mm256_srli_epi32(x, n), _mm256_slli_epi32(x, 32 - n))
#define SHR(x,n)   _mm256_srli_epi32(x, n)

#define S0(x) (_mm256_xor_si256(ROR((x), 2),  _mm256_xor_si256(ROR((x), 13), ROR((x), 22))))
#define S1(x) (_mm256_xor_si256(ROR((x), 6),  _mm256_xor_si256(ROR((x), 11), ROR((x), 25))))
#define s0(x) (_mm256_xor_si256(ROR((x), 7),  _mm256_xor_si256(ROR((x), 18), SHR((x), 3))))
#define s1(x) (_mm256_xor_si256(ROR((x), 17), _mm256_xor_si256(ROR((x), 19), SHR((x), 10))))

#define add4(x0, x1, x2, x3) _mm256_add_epi32(_mm256_add_epi32(x0, x1), _mm256_add_epi32(x2, x3))
#define add3(x0, x1, x2)     _mm256_add_epi32(_mm256_add_epi32(x0, x1), x2)
#define add5(x0, x1, x2, x3, x4) _mm256_add_epi32(add3(x0, x1, x2), _mm256_add_epi32(x3, x4))

#define Round(a, b, c, d, e, f, g, h, i, w)                 \
    T1 = add5(h, S1(e), Ch(e, f, g), _mm256_set1_epi32(i), w); \
    d = _mm256_add_epi32(d, T1);                            \
    T2 = _mm256_add_epi32(S0(a), Maj(a, b, c));             \
    h = _mm256_add_epi32(T1, T2);

#define WMIX() \
  w0 = add4(s1(w14), w9, s0(w1), w0); \
  w1 = add4(s1(w15), w10, s0(w2), w1); \
  w2 = add4(s1(w0), w11, s0(w3), w2); \
  w3 = add4(s1(w1), w12, s0(w4), w3); \
  w4 = add4(s1(w2), w13, s0(w5), w4); \
  w5 = add4(s1(w3), w14, s0(w6), w5); \
  w6 = add4(s1(w4), w15, s0(w7), w6); \
  w7 = add4(s1(w5), w0, s0(w8), w7); \
  w8 = add4(s1(w6), w1, s0(w9), w8); \
  w9 = add4(s1(w7), w2, s0(w10), w9); \
  w10 = add4(s1(w8), w3, s0(w11), w10); \
  w11 = add4(s1(w9), w4, s0(w12), w11); \
  w12 = add4(s1(w10), w5, s0(w13), w12); \
  w13 = add4(s1(w11), w6, s0(w14), w13); \
  w14 = add4(s1(w12), w7, s0(w15), w14); \
  w15 = add4(s1(w13), w8, s0(w0), w15);

// Gather word i (lane j comes from buffer j)
#define LOADW(i) _mm256_setr_epi32(b0[i], b1[i], b2[i], b3[i], b4[i], b5[i], b6[i], b7[i])

  static void Initialize(__m256i *s) {
    s[0] = _mm256_set1_epi32(0x6a09e667ul);
    s[1] = _mm256_set1_epi32(0xbb67ae85ul);
    s[2] = _mm256_set1_epi32(0x3c6ef372ul);
    s[3] = _mm256_set1_epi32(0xa54ff53aul);
    s[4] = _mm256_set1_epi32(0x510e527ful);
    s[5] = _mm256_set1_epi32(0x9b05688cul);
    s[6] = _mm256_set1_epi32(0x1f83d9abul);
    s[7] = _mm256_set1_epi32(0x5be0cd19ul);
  }

  // 8 SHA-256 blocks in parallel
  static void Transform(__m256i *s,
                        uint32_t *b0, uint32_t *b1, uint32_t *b2, uint32_t *b3,
                        uint32_t *b4, uint32_t *b5, uint32_t *b6, uint32_t *b7) {

    __m256i a, b, c, d, e, f, g, h;
    __m256i w0, w1, w2, w3, w4, w5, w6, w7;
    __m256i w8, w9, w10, w11, w12, w13, w14, w15;
    __m256i T1, T2;

    a = s[0]; b = s[1]; c = s[2]; d = s[3];
    e = s[4]; f = s[5]; g = s[6]; h = s[7];

    w0 = LOADW(0);   w1 = LOADW(1);   w2 = LOADW(2);   w3 = LOADW(3);
    w4 = LOADW(4);   w5 = LOADW(5);   w6 = LOADW(6);   w7 = LOADW(7);
    w8 = LOADW(8);   w9 = LOADW(9);   w10 = LOADW(10); w11 = LOADW(11);
    w12 = LOADW(12); w13 = LOADW(13); w14 = LOADW(14); w15 = LOADW(15);

    Round(a, b, c, d, e, f, g, h, 0x428A2F98, w0);
    Round(h, a, b, c, d, e, f, g, 0x71374491, w1);
    Round(g, h, a, b, c, d, e, f, 0xB5C0FBCF, w2);
    Round(f, g, h, a, b, c, d, e, 0xE9B5DBA5, w3);
    Round(e, f, g, h, a, b, c, d, 0x3956C25B, w4);
    Round(d, e, f, g, h, a, b, c, 0x59F111F1, w5);
    Round(c, d, e, f, g, h, a, b, 0x923F82A4, w6);
    Round(b, c, d, e, f, g, h, a, 0xAB1C5ED5, w7);
    Round(a, b, c, d, e, f, g, h, 0xD807AA98, w8);
    Round(h, a, b, c, d, e, f, g, 0x12835B01, w9);
    Round(g, h, a, b, c, d, e, f, 0x243185BE, w10);
    Round(f, g, h, a, b, c, d, e, 0x550C7DC3, w11);
    Round(e, f, g, h, a, b, c, d, 0x72BE5D74, w12);
    Round(d, e, f, g, h, a, b, c, 0x80DEB1FE, w13);
    Round(c, d, e, f, g, h, a, b, 0x9BDC06A7, w14);
    Round(b, c, d, e, f, g, h, a, 0xC19BF174, w15);

    WMIX()

    Round(a, b, c, d, e, f, g, h, 0xE49B69C1, w0);
    Round(h, a, b, c, d, e, f, g, 0xEFBE4786, w1);
    Round(g, h, a, b, c, d, e, f, 0x0FC19DC6, w2);
    Round(f, g, h, a, b, c, d, e, 0x240CA1CC, w3);
    Round(e, f, g, h, a, b, c, d, 0x2DE92C6F, w4);
    Round(d, e, f, g, h, a, b, c, 0x4A7484AA, w5);
    Round(c, d, e, f, g, h, a, b, 0x5CB0A9DC, w6);
    Round(b, c, d, e, f, g, h, a, 0x76F988DA, w7);
    Round(a, b, c, d, e, f, g, h, 0x983E5152, w8);
    Round(h, a, b, c, d, e, f, g, 0xA831C66D, w9);
    Round(g, h, a, b, c, d, e, f, 0xB00327C8, w10);
    Round(f, g, h, a, b, c, d, e, 0xBF597FC7, w11);
    Round(e, f, g, h, a, b, c, d, 0xC6E00BF3, w12);
    Round(d, e, f, g, h, a, b, c, 0xD5A79147, w13);
    Round(c, d, e, f, g, h, a, b, 0x06CA6351, w14);
    Round(b, c, d, e, f, g, h, a, 0x14292967, w15);

    WMIX()

    Round(a, b, c, d, e, f, g, h, 0x27B70A85, w0);
    Round(h, a, b, c, d, e, f, g, 0x2E1B2138, w1);
    Round(g, h, a, b, c, d, e, f, 0x4D2C6DFC, w2);
    Round(f, g, h, a, b, c, d, e, 0x53380D13, w3);
    Round(e, f, g, h, a, b, c, d, 0x650A7354, w4);
    Round(d, e, f, g, h, a, b, c, 0x766A0ABB, w5);
    Round(c, d, e, f, g, h, a, b, 0x81C2C92E, w6);
    Round(b, c, d, e, f, g, h, a, 0x92722C85, w7);
    Round(a, b, c, d, e, f, g, h, 0xA2BFE8A1, w8);
    Round(h, a, b, c, d, e, f, g, 0xA81A664B, w9);
    Round(g, h, a, b, c, d, e, f, 0xC24B8B70, w10);
    Round(f, g, h, a, b, c, d, e, 0xC76C51A3, w11);
    Round(e, f, g, h, a, b, c, d, 0xD192E819, w12);
    Round(d, e, f, g, h, a, b, c, 0xD6990624, w13);
    Round(c, d, e, f, g, h, a, b, 0xF40E3585, w14);
    Round(b, c, d, e, f, g, h, a, 0x106AA070, w15);

    WMIX()

    Round(a, b, c, d, e, f, g, h, 0x19A4C116, w0);
    Round(h, a, b, c, d, e, f, g, 0x1E376C08, w1);
    Round(g, h, a, b, c, d, e, f, 0x2748774C, w2);
    Round(f, g, h, a, b, c, d, e, 0x34B0BCB5, w3);
    Round(e, f, g, h, a, b, c, d, 0x391C0CB3, w4);
    Round(d, e, f, g, h, a, b, c, 0x4ED8AA4A, w5);
    Round(c, d, e, f, g, h, a, b, 0x5B9CCA4F, w6);
    Round(b, c, d, e, f, g, h, a, 0x682E6FF3, w7);
    Round(a, b, c, d, e, f, g, h, 0x748F82EE, w8);
    Round(h, a, b, c, d, e, f, g, 0x78A5636F, w9);
    Round(g, h, a, b, c, d, e, f, 0x84C87814, w10);
    Round(f, g, h, a, b, c, d, e, 0x8CC70208, w11);
    Round(e, f, g, h, a, b, c, d, 0x90BEFFFA, w12);
    Round(d, e, f, g, h, a, b, c, 0xA4506CEB, w13);
    Round(c, d, e, f, g, h, a, b, 0xBEF9A3F7, w14);
    Round(b, c, d, e, f, g, h, a, 0xC67178F2, w15);

    s[0] = _mm256_add_epi32(a, s[0]);
    s[1] = _mm256_add_epi32(b, s[1]);
    s[2] = _mm256_add_epi32(c, s[2]);
    s[3] = _mm256_add_epi32(d, s[3]);
    s[4] = _mm256_add_epi32(e, s[4]);
    s[5] = _mm256_add_epi32(f, s[5]);
    s[6] = _mm256_add_epi32(g, s[6]);
    s[7] = _mm256_add_epi32(h, s[7]);
  }

  // Write big-endian 32-byte digests, one per lane.
  static inline void Depack(__m256i *s, unsigned char *d[8]) {
    uint32_t st[8][8];
    for (int k = 0; k < 8; k++)
      _mm256_storeu_si256((__m256i *)st[k], s[k]);
    for (int lane = 0; lane < 8; lane++) {
      uint32_t *o = (uint32_t *)d[lane];
      o[0] = __builtin_bswap32(st[0][lane]);
      o[1] = __builtin_bswap32(st[1][lane]);
      o[2] = __builtin_bswap32(st[2][lane]);
      o[3] = __builtin_bswap32(st[3][lane]);
      o[4] = __builtin_bswap32(st[4][lane]);
      o[5] = __builtin_bswap32(st[5][lane]);
      o[6] = __builtin_bswap32(st[6][lane]);
      o[7] = __builtin_bswap32(st[7][lane]);
    }
  }

} // namespace _sha256avx2

// One 64-byte block (compressed pubkey path)
void sha256avx2_8B(
  uint32_t *i0, uint32_t *i1, uint32_t *i2, uint32_t *i3,
  uint32_t *i4, uint32_t *i5, uint32_t *i6, uint32_t *i7,
  unsigned char *d0, unsigned char *d1, unsigned char *d2, unsigned char *d3,
  unsigned char *d4, unsigned char *d5, unsigned char *d6, unsigned char *d7) {

  __m256i s[8];
  _sha256avx2::Initialize(s);
  _sha256avx2::Transform(s, i0, i1, i2, i3, i4, i5, i6, i7);
  unsigned char *d[8] = { d0, d1, d2, d3, d4, d5, d6, d7 };
  _sha256avx2::Depack(s, d);
}

// Two 64-byte blocks (uncompressed pubkey path)
void sha256avx2_16B(
  uint32_t *i0, uint32_t *i1, uint32_t *i2, uint32_t *i3,
  uint32_t *i4, uint32_t *i5, uint32_t *i6, uint32_t *i7,
  unsigned char *d0, unsigned char *d1, unsigned char *d2, unsigned char *d3,
  unsigned char *d4, unsigned char *d5, unsigned char *d6, unsigned char *d7) {

  __m256i s[8];
  _sha256avx2::Initialize(s);
  _sha256avx2::Transform(s, i0, i1, i2, i3, i4, i5, i6, i7);
  _sha256avx2::Transform(s, i0 + 16, i1 + 16, i2 + 16, i3 + 16,
                            i4 + 16, i5 + 16, i6 + 16, i7 + 16);
  unsigned char *d[8] = { d0, d1, d2, d3, d4, d5, d6, d7 };
  _sha256avx2::Depack(s, d);
}
