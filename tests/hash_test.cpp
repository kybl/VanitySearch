/*
 * Correctness tests for the vectorized hash kernels.
 *
 * The SSE / AVX2 / AVX-512 SHA-256 and RIPEMD-160 implementations are
 * hand-written and soundness-critical: a single wrong lane silently corrupts
 * the address of every key on that lane. This harness hashes random inputs
 * through each vector width and compares against the trusted scalar reference
 * (CSHA256 / CRIPEMD160). Build & run with `make test`.
 *
 * Exit code 0 = all pass, 1 = a mismatch was found.
 */

#include "../hash/sha256.h"
#include "../hash/ripemd160.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>

static int failures = 0;

static void report(const char *name, bool ok) {
  printf("  %-28s %s\n", name, ok ? "OK" : "FAIL");
  if (!ok) failures++;
}

// Build a big-endian, single-block padded SHA-256 message from an L-byte input
// (L <= 55). Mirrors what sha256_33 / the KEYBUFF macros feed the kernels.
static void build_sha_block(const unsigned char *msg, int L, uint32_t out[16]) {
  unsigned char blk[64];
  memset(blk, 0, 64);
  memcpy(blk, msg, L);
  blk[L] = 0x80;
  uint64_t bits = (uint64_t)L * 8;
  blk[62] = (unsigned char)(bits >> 8);
  blk[63] = (unsigned char)(bits & 0xff);
  for (int w = 0; w < 16; w++)
    out[w] = (blk[w*4] << 24) | (blk[w*4+1] << 16) | (blk[w*4+2] << 8) | blk[w*4+3];
}

// Two-block version for a 65-byte (uncompressed pubkey) input.
static void build_sha_block2(const unsigned char *msg, int L, uint32_t out[32]) {
  unsigned char blk[128];
  memset(blk, 0, 128);
  memcpy(blk, msg, L);
  blk[L] = 0x80;
  uint64_t bits = (uint64_t)L * 8;
  blk[126] = (unsigned char)(bits >> 8);
  blk[127] = (unsigned char)(bits & 0xff);
  for (int w = 0; w < 32; w++)
    out[w] = (blk[w*4] << 24) | (blk[w*4+1] << 16) | (blk[w*4+2] << 8) | blk[w*4+3];
}

static unsigned int seed = 1234567u;
static unsigned char rnd() { seed = seed * 1103515245u + 12345u; return (unsigned char)(seed >> 16); }

// ---- RIPEMD-160 --------------------------------------------------------------

static void test_ripemd_sse() {
  bool ok = true;
  for (int t = 0; t < 500 && ok; t++) {
    unsigned char in[4][64], ref[4][20], out[4][20];
    for (int j = 0; j < 4; j++) { for (int k = 0; k < 32; k++) in[j][k] = rnd(); ripemd160_32(in[j], ref[j]); }
    ripemd160sse_32(in[0], in[1], in[2], in[3], out[0], out[1], out[2], out[3]);
    for (int j = 0; j < 4; j++) if (memcmp(out[j], ref[j], 20)) ok = false;
  }
  report("RIPEMD160 SSE (4-way)", ok);
}

#ifdef WITH_AVX2
static void test_ripemd_avx2() {
  bool ok = true;
  for (int t = 0; t < 500 && ok; t++) {
    unsigned char in[8][64], ref[8][20], out[8][20];
    for (int j = 0; j < 8; j++) { for (int k = 0; k < 32; k++) in[j][k] = rnd(); ripemd160_32(in[j], ref[j]); }
    ripemd160avx2_32(in[0],in[1],in[2],in[3],in[4],in[5],in[6],in[7],
                     out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]);
    for (int j = 0; j < 8; j++) if (memcmp(out[j], ref[j], 20)) ok = false;
  }
  report("RIPEMD160 AVX2 (8-way)", ok);
}
#endif

#ifdef WITH_AVX512
static void test_ripemd_avx512() {
  bool ok = true;
  for (int t = 0; t < 500 && ok; t++) {
    unsigned char in[16][64], ref[16][20], out[16][20];
    unsigned char *ip[16], *op[16];
    for (int j = 0; j < 16; j++) { for (int k = 0; k < 32; k++) in[j][k] = rnd(); ripemd160_32(in[j], ref[j]); ip[j]=in[j]; op[j]=out[j]; }
    ripemd160avx512_32(ip, op);
    for (int j = 0; j < 16; j++) if (memcmp(out[j], ref[j], 20)) ok = false;
  }
  report("RIPEMD160 AVX-512 (16-way)", ok);
}
#endif

// ---- SHA-256 (1 block, compressed pubkey path) -------------------------------

static void test_sha_sse_1b() {
  bool ok = true;
  for (int t = 0; t < 500 && ok; t++) {
    uint32_t b[4][16]; unsigned char ref[4][32], out[4][32];
    for (int j = 0; j < 4; j++) {
      unsigned char msg[33]; for (int k = 0; k < 33; k++) msg[k] = rnd();
      unsigned char tmp[64]; memcpy(tmp, msg, 33); sha256_33(tmp, ref[j]);
      build_sha_block(msg, 33, b[j]);
    }
    sha256sse_1B(b[0], b[1], b[2], b[3], out[0], out[1], out[2], out[3]);
    for (int j = 0; j < 4; j++) if (memcmp(out[j], ref[j], 32)) ok = false;
  }
  report("SHA256 SSE 1-block (4-way)", ok);
}

#ifdef WITH_AVX2
static void test_sha_avx2() {
  bool ok = true;
  for (int t = 0; t < 500 && ok; t++) {
    uint32_t b[8][16]; unsigned char ref[8][32], out[8][32];
    for (int j = 0; j < 8; j++) {
      unsigned char msg[33]; for (int k = 0; k < 33; k++) msg[k] = rnd();
      unsigned char tmp[64]; memcpy(tmp, msg, 33); sha256_33(tmp, ref[j]);
      build_sha_block(msg, 33, b[j]);
    }
    sha256avx2_8B(b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7],
                  out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]);
    for (int j = 0; j < 8; j++) if (memcmp(out[j], ref[j], 32)) ok = false;
  }
  report("SHA256 AVX2 1-block (8-way)", ok);

  // Two-block (uncompressed) path
  ok = true;
  for (int t = 0; t < 500 && ok; t++) {
    uint32_t b[8][32]; unsigned char ref[8][32], out[8][32];
    for (int j = 0; j < 8; j++) {
      unsigned char msg[65]; for (int k = 0; k < 65; k++) msg[k] = rnd();
      unsigned char tmp[128]; memcpy(tmp, msg, 65); sha256_65(tmp, ref[j]);
      build_sha_block2(msg, 65, b[j]);
    }
    sha256avx2_16B(b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7],
                   out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]);
    for (int j = 0; j < 8; j++) if (memcmp(out[j], ref[j], 32)) ok = false;
  }
  report("SHA256 AVX2 2-block (8-way)", ok);
}
#endif

#ifdef WITH_AVX512
static void test_sha_avx512() {
  bool ok = true;
  for (int t = 0; t < 500 && ok; t++) {
    uint32_t b[16][16]; unsigned char ref[16][32], out[16][32];
    uint32_t *bp[16]; unsigned char *op[16];
    for (int j = 0; j < 16; j++) {
      unsigned char msg[33]; for (int k = 0; k < 33; k++) msg[k] = rnd();
      unsigned char tmp[64]; memcpy(tmp, msg, 33); sha256_33(tmp, ref[j]);
      build_sha_block(msg, 33, b[j]); bp[j] = b[j]; op[j] = out[j];
    }
    sha256avx512_16x8B(bp, op);
    for (int j = 0; j < 16; j++) if (memcmp(out[j], ref[j], 32)) ok = false;
  }
  report("SHA256 AVX-512 1-block (16-way)", ok);

  ok = true;
  for (int t = 0; t < 500 && ok; t++) {
    uint32_t b[16][32]; unsigned char ref[16][32], out[16][32];
    uint32_t *bp[16]; unsigned char *op[16];
    for (int j = 0; j < 16; j++) {
      unsigned char msg[65]; for (int k = 0; k < 65; k++) msg[k] = rnd();
      unsigned char tmp[128]; memcpy(tmp, msg, 65); sha256_65(tmp, ref[j]);
      build_sha_block2(msg, 65, b[j]); bp[j] = b[j]; op[j] = out[j];
    }
    sha256avx512_16x16B(bp, op);
    for (int j = 0; j < 16; j++) if (memcmp(out[j], ref[j], 32)) ok = false;
  }
  report("SHA256 AVX-512 2-block (16-way)", ok);
}
#endif

// ---- checksum ----------------------------------------------------------------

static void test_checksum() {
  // sha256_checksum must equal the first 4 bytes of double-SHA256.
  bool ok = true;
  for (int t = 0; t < 500 && ok; t++) {
    unsigned char in[64]; int L = 1 + (rnd() % 55);
    for (int k = 0; k < L; k++) in[k] = rnd();
    unsigned char c[4], d1[32], d2[32];
    sha256_checksum(in, L, c);
    sha256(in, L, d1); sha256(d1, 32, d2);
    if (memcmp(c, d2, 4)) ok = false;
  }
  report("sha256_checksum == dSHA256[:4]", ok);
}

int main() {
  printf("Vectorized hash kernel tests:\n");
  test_checksum();
  test_ripemd_sse();
  test_sha_sse_1b();
#ifdef WITH_AVX2
  test_ripemd_avx2();
  test_sha_avx2();
#endif
#ifdef WITH_AVX512
  test_ripemd_avx512();
  test_sha_avx512();
#endif
  if (failures == 0) {
    printf("All hash kernel tests passed.\n");
    return 0;
  }
  printf("%d test(s) FAILED.\n", failures);
  return 1;
}
