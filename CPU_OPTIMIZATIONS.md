# Analýza a návrhy zrychlení CPU části VanitySearch

Datum: 2026-07-04. Měřeno na 4jádrovém Xeonu (AVX2 + AVX-512, bez SHA-NI), GCC 13.3, Linux.

## Jak CPU část funguje dnes

Hlavní smyčka je `VanitySearch::FindKeyCPU` (Vanity.cpp):

1. Každé vlákno si drží startovní bod `startP = k*G` a v každé iteraci spočítá
   skupinu `CPU_GRP_SIZE = 1024` po sobě jdoucích bodů `startP ± i*G` pomocí
   předpočítané tabulky `Gn[]`.
2. Sčítání bodů potřebuje modulární inverzi. Ta se dělá **dávkově** pro celou
   skupinu najednou (`IntGroup::ModInv`, Montgomeryho trik) — na 1024 bodů
   připadá jediná skutečná inverze (DRS62/Porninova metoda v `Int::ModInv`).
   Zbytek jsou 3 modulární násobení na bod.
3. Z každého bodu se levně odvodí dalších 5 kandidátů: endomorfismy
   (`beta*x`, `beta2*x`) a symetrie `(x, -y)` — celkem **6 kandidátních klíčů na
   1 spočítaný bod** (proto čítač přičítá `6*CPU_GRP_SIZE`).
4. Každý kandidát se zahashuje: `hash160 = RIPEMD160(SHA256(pubkey))`, SSE
   verzí po 4 kusech (`sha256sse_1B`/`sha256sse_2B` + `ripemd160sse_32`).
5. Prvních 16 bitů hash160 se vyhledá v tabulce `prefixes[65536]`; jen při
   shodě se staví Base58/Bech32 adresa a porovnává prefix řetězcově.

## Profil (callgrind, 1 vlákno, výchozí komprimovaný režim)

| Funkce | Podíl instrukcí |
|---|---|
| `_sha256sse::Transform` | **54,4 %** |
| `ripemd160sse::Transform` | **28,2 %** |
| `Int::ModMulK1(a,b)` + `ModMulK1(a)` + `ModSquareK1` | 8,0 % |
| `GetHash160` (skládání bufferů) | 2,4 % |
| režie smyčky, kopie `Int`/`Point`, ModAdd/Sub | ~5 % |

**Hashování žere ~85 % času.** EC aritmetika je díky dávkové inverzi a
endomorfismům už velmi dobře amortizovaná. Jakékoli zrychlení tedy musí mířit
primárně na SHA-256 a RIPEMD-160.

Baseline: **14,7 Mkey/s** (4 vlákna, `-O2 -mssse3` dle Makefile).

## Návrhy zlepšení (seřazeno podle poměru přínos/úsilí)

### 1. Flagy kompilátoru — ZMĚŘENO: +31 % zdarma
`-O2 -mssse3` → `-O3 -march=native -funroll-loops`:
**14,7 → 19,3 Mkey/s**. Kompilátor přeloží stávající SSE intrinsics do
AVX (3operandové VEX instrukce, méně spillů) a lépe rozbalí smyčky.
Do Makefile přidat volitelně i `-flto` a PGO (`-fprofile-generate/-fprofile-use`,
typicky další jednotky %). Pro distribuční binárku ponechat fallback bez
`-march=native`.

### 2. AVX2 8-way (a AVX-512 16-way) SHA-256 + RIPEMD-160 — největší přínos
Současné `hash/sha256_sse.cpp` a `hash/ripemd160_sse.cpp` používají `__m128i`
(4 zprávy paralelně). Přepis na `__m256i` zpracuje 8 zpráv, AVX-512 16 zpráv
na průchod. Vzorec kola je identický (jen `_mm_` → `_mm256_`/`_mm512_`,
rotace na AVX-512 nativně přes `_mm512_rol_epi32`).

- Hash je 85 % času ⇒ 2× rychlejší hash ≈ **1,7× celkově** (~33 Mkey/s),
  AVX-512 potenciálně k ~45-50 Mkey/s.
- Vyžaduje restrukturalizaci volající vrstvy: místo `checkAddressesSSE`
  po 4 bodech nasbírat všech 6×1024 kandidátních bufferů za skupinu a hashovat
  v dlouhých bězích po 8/16 — zlepší i I-cache a predikci větví.
- Runtime dispatch podle CPUID (SSE→AVX2→AVX-512, případně SHA-NI viz níže),
  aby binárka běžela všude.

### 3. SHA-NI pro CPU, které ho mají (Zen, Ice Lake+)
Tento stroj SHA-NI nemá, ale na Ryzenech dělá `sha256rnds2` jeden blok za
~60-70 cyklů — 2 nezávislé řetězce prokládané dosáhnou ~2× výkonu 4-way SSE.
RIPEMD-160 hardwarovou podporu nemá, zůstane vektorový. Řešit v rámci
dispatch vrstvy z bodu 2.

### 4. Bitmapa místo 1MB vyhledávací tabulky prefixů
`prefixes` je `vector<PREFIX_TABLE_ITEM>` (16 B × 65536 = 1 MB) a sahá se do ní
náhodně **u každého kandidáta** — prakticky jistý cache-miss do L2/L3.
Přidat 65536bitovou bitmapu (8 KB, sedí do L1): `if (bitmap[pr>>3] & (1<<(pr&7)))`
a teprve při zásahu (pravděpodobnost ~1e-5 na kandidáta u jednoho prefixu)
jít do plné tabulky. Odhad jednotky % (Ir profil cache-missy podceňuje).

### 5. Odstranit zbytečné kopie v horké smyčce
- `checkAddresses`/`checkAddressesSSE` berou `Int key` a 4× `Point` **hodnotou**
  (`Point` = 3×40 B) — předávat referencí/ukazatelem.
- `pts[CPU_GRP_SIZE]` je pole `Point` (120 KB/vlákno) a `z` složka se nikdy
  nepoužívá — přejít na SoA (`px[]`, `py[]`), tj. 64 KB, lepší cache locality.
  Totéž pro tabulku `Gn` (načítá se sekvenčně každou iteraci, hustota
  64 B místo 120 B na bod).
- Ve smyčce `pp = startP; pn = startP` se kopíruje i nepoužívané `z`.
Dohromady odhad ~2-4 %.

### 6. Branchless ModAdd/ModSub
`ModSub` dělá `if (IsNegative()) Add(&_P)` — u náhodných dat 50% větev,
tj. častý mispredict; volá se ~5× na bod. Nahradit maskovaným přičtením
(`sbb` → maska → `adc`), jako to dělá GPU kód (`GPUMath.h` je branchless).
Odhad 1-2 %.

### 7. ModMulK1 přes MULX/ADCX/ADOX (BMI2+ADX)
Ruční 4×4 násobička používá `mulq` + jediný carry řetězec. MULX+ADCX/ADOX
umožní dva nezávislé carry řetězce (jak to dělá libsecp256k1). Pole je ale jen
8 % profilu ⇒ celkově ~2-4 %. Alternativa: převzít 5×52bit reprezentaci
libsecp256k1, to už je velký zásah za malý zisk.

### 8. Co se NEvyplatí
- **Zvětšení CPU_GRP_SIZE**: inverze je 1,3 µs na skupinu, která trvá ~1,3 ms
  — amortizace už je 0,1 %. Zdvojnásobení skupiny ušetří <0,05 % a zhorší cache.
- **Optimalizace `Int::ModInv`**: volá se jednou za 6144 kandidátů, je to šum.
- **Midstate/inkrementální SHA**: vstup je jediný 64B blok (komprimovaný klíč),
  který se mění celý — není co recyklovat.

### 9. Drobnosti / hygiena
- Oprava buildu na GCC ≥ 11: chybějící `#include <cstdint>` v `Timer.h`,
  `hash/sha256.h`, `hash/sha512.h` (opraveno v tomto commitu).
- Wildcard režim (`-hasPattern`) staví pro každý kandidát celou Base58 adresu
  včetně dvojitého SHA-256 checksumu — pokud pattern začíná pevným prefixem,
  dá se předfiltrovat přes hash160 stejně jako normální režim (řádově rychlejší).
- `-t` výchozí = počet logických jader; na strojích s HT stojí za test
  `-t <fyzická jádra>` i pinning vláken (u hash-bound zátěže HT obvykle
  pomáhá, ale změřit).

## Očekávaný souhrn

| Krok | Odhad Mkey/s (tento stroj, 4 vlákna) |
|---|---|
| Baseline (`-O2 -mssse3`) | 14,7 (změřeno) |
| + flagy `-O3 -march=native` | 19,3 (změřeno) |
| + AVX2 8-way hash + dávkové hashování | ~33 (odhad) |
| + AVX-512 16-way | ~45+ (odhad) |
| + bitmapa, SoA, branchless, MULX | +5-10 % navrch |

Doporučené pořadí implementace: **1 → 2 (AVX2) → 4 → 5 → 6 → 7**, přičemž
bod 2 je jediný s velkým dopadem a zaslouží si samostatnou větev s pečlivým
testováním správnosti (porovnání hash160 proti referenční SSE implementaci).
