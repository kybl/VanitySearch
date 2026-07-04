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

#ifndef SHA256_H
#define SHA256_H
#include <string>
#include <cstdint>

void sha256(uint8_t *input,int length, uint8_t *digest);
void sha256_33(uint8_t *input, uint8_t *digest);
void sha256_65(uint8_t *input, uint8_t *digest);
void sha256_checksum(uint8_t *input, int length, uint8_t *checksum);
void sha256sse_1B(uint32_t *i0, uint32_t *i1, uint32_t *i2, uint32_t *i3,
  uint8_t *d0, uint8_t *d1, uint8_t *d2, uint8_t *d3);
void sha256sse_2B(uint32_t *i0, uint32_t *i1, uint32_t *i2, uint32_t *i3,
  uint8_t *d0, uint8_t *d1, uint8_t *d2, uint8_t *d3);
void sha256sse_checksum(uint32_t *i0, uint32_t *i1, uint32_t *i2, uint32_t *i3,
  uint8_t *d0, uint8_t *d1, uint8_t *d2, uint8_t *d3);
std::string sha256_hex(unsigned char *digest);
void sha256sse_test();

// 8-way AVX2 variants (i0..i7 message buffers, d0..d7 32-byte digests)
void sha256avx2_8B(uint32_t *i0, uint32_t *i1, uint32_t *i2, uint32_t *i3,
  uint32_t *i4, uint32_t *i5, uint32_t *i6, uint32_t *i7,
  unsigned char *d0, unsigned char *d1, unsigned char *d2, unsigned char *d3,
  unsigned char *d4, unsigned char *d5, unsigned char *d6, unsigned char *d7);
void sha256avx2_16B(uint32_t *i0, uint32_t *i1, uint32_t *i2, uint32_t *i3,
  uint32_t *i4, uint32_t *i5, uint32_t *i6, uint32_t *i7,
  unsigned char *d0, unsigned char *d1, unsigned char *d2, unsigned char *d3,
  unsigned char *d4, unsigned char *d5, unsigned char *d6, unsigned char *d7);

// 16-way AVX-512 variants (b[16] message buffers, d[16] 32-byte digests)
void sha256avx512_16x8B(uint32_t *b[16], unsigned char *d[16]);
void sha256avx512_16x16B(uint32_t *b[16], unsigned char *d[16]);

#endif