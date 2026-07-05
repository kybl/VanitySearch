#!/usr/bin/env bash
#
# Profile-guided-optimization build for the CPU search.
#
# The effect of PGO is MICROARCHITECTURE-SPECIFIC and can go either way, so
# this script builds both the plain and the PGO binary, benchmarks them on
# THIS machine, and tells you which one won. Only deploy the PGO binary if it
# actually beats the plain build here.
#
# Three phases:
#   1. build instrumented   (-fprofile-generate)
#   2. run a short representative search to record a profile
#   3. rebuild with the profile (-fprofile-use)
#
# Usage:  ./scripts/pgo-build.sh [seconds_for_profiling]
# Output: ./VanitySearch        (plain build, restored at the end)
#         ./VanitySearch-pgo    (PGO build)
set -euo pipefail
cd "$(dirname "$0")/.."

SECS="${1:-20}"
PROF_DIR="$(pwd)/pgo-data"
rm -rf "$PROF_DIR"; mkdir -p "$PROF_DIR"

# A prefix that is hard enough never to be found during profiling, so the
# search runs its hot loop for the whole SECS. Cover compressed, uncompressed
# and "both" so every checkAddresses* path is exercised.
PROFILE() {
  timeout "$SECS" ./VanitySearch -t "$(nproc)" "$@" 1FooBarBazQuxProfile >/dev/null 2>&1 || true
}

echo "==> [1/3] building instrumented binary"
make clean >/dev/null 2>&1
make -j"$(nproc)" PGOFLAGS="-fprofile-generate=$PROF_DIR -fprofile-update=prefer-atomic" >/dev/null 2>&1

echo "==> [2/3] gathering profile (~$((SECS * 3))s across search modes)"
PROFILE
PROFILE -u
PROFILE -b

echo "==> [3/3] rebuilding with profile"
make clean >/dev/null 2>&1
# -fprofile-correction tolerates the multi-threaded, slightly-racy counters.
make -j"$(nproc)" PGOFLAGS="-fprofile-use=$PROF_DIR -fprofile-correction -Wno-missing-profile" >/dev/null 2>&1
cp VanitySearch VanitySearch-pgo

echo "==> rebuilding plain binary for an apples-to-apples comparison"
make clean >/dev/null 2>&1
make -j"$(nproc)" >/dev/null 2>&1

rate() { # average Mkey/s over the tail of a short run
  local out
  out="$(timeout "$SECS" "$1" -t "$(nproc)" 1FooBarBazQuxBench 2>/dev/null || true)"
  printf '%s' "$out" | tr '\r' '\n' | grep Mkey | tail -8 \
    | awk -F'[][]' '{print $2}' | awk '{s+=$1;n++} END{if(n)printf "%.2f",s/n}'
}

echo "==> comparing plain vs PGO on this machine (${SECS}s each)"
PLAIN=$(rate ./VanitySearch) || true
PGO=$(rate ./VanitySearch-pgo) || true
echo "    plain : ${PLAIN:-?} Mkey/s"
echo "    pgo   : ${PGO:-?} Mkey/s"
awk -v p="$PLAIN" -v g="$PGO" 'BEGIN{
  if (p>0) printf "    delta : %+.1f%%\n", (g-p)/p*100.0;
  if (g>p) print "==> PGO wins -> use ./VanitySearch-pgo";
  else     print "==> PGO does not help on this machine -> keep ./VanitySearch";
}'

rm -rf "$PROF_DIR"
