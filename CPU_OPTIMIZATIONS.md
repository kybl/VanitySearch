# Zrychlení CPU části VanitySearch — analýza a provedené změny

Datum: 2026-07-04. Měřeno na 4jádrovém Xeonu (AVX2 + AVX-512F, bez SHA-NI),
GCC 13.3, Linux. Všechny změny jsou čistě v CPU části; GPU kód sloužil jen
jako inspirace (branchless aritmetika, endomorfismy).

## Výsledek

| Konfigurace (4 vlákna, komprimovaný režim) | Mkey/s | Zrychlení |
|---|---|---|
| Výchozí stav (`-O2 -mssse3`, SSE 4-way hash) | 14,7 | 1,00× |
| + optimalizační flagy (`-O3 -march=native`) | 19,5 | 1,33× |
| + AVX2 8-way hashování | ~30 | 2,0× |
| + AVX-512 16-way hashování | ~38 | 2,6× |
| + branchless field aritmetika | ~38 (v šumu) | — |
| + LTO (`-flto`) | **~2,8-3×** | **~2,8-3×** |

Absolutní čísla kolísají podle zátěže cloud VM (± ~15 %); zrychlení jednotlivých
kroků je měřeno vždy A/B ve stejné session. LTO přidalo měřeně **+9 %**.

**Celkově ~3× rychlejší** oproti původnímu stavu, plus oprava vážné chyby
ve výpočtu checksumu (viz níže) a dvě opravy buildu.

## Jak CPU část funguje

Hlavní smyčka `VanitySearch::FindKeyCPU` (Vanity.cpp):

1. Každé vlákno drží bod `startP = k*G` a v každé iteraci spočítá skupinu
   `CPU_GRP_SIZE = 1024` sousedních bodů `startP ± i*G` z předpočítané tabulky
   `Gn[]`. Sčítání bodů potřebuje modulární inverzi; ta se dělá **dávkově** pro
   celou skupinu (`IntGroup::ModInv`, Montgomeryho trik) — 1 skutečná inverze
   na 1024 bodů.
2. Z každého bodu se levně odvodí 5 dalších kandidátů: dva endomorfismy
   (`beta*x`, `beta2*x`) a křivková symetrie `(x,-y)` — dohromady **6
   kandidátních klíčů na 1 spočítaný bod** (proto čítač přičítá `6*CPU_GRP_SIZE`).
   Tento poměr 6× je algoritmicky maximální (secp256k1 má jen jeden λ).
3. Každý kandidát se zahashuje `hash160 = RIPEMD160(SHA256(pubkey))` a prvních
   16 bitů se vyhledá v tabulce `prefixes[65536]`.

## Kde se čas skutečně tráví

Klíčové zjištění: **profilování podle počtu instrukcí (callgrind) je zavádějící**
— nadhodnocuje hashování (spousta levných SIMD instrukcí) a podhodnocuje
modulární aritmetiku (málo instrukcí, ale dlouhé latenční řetězce mulx/adc).

Měřeno podle skutečného času (rdtsc + A/B s vypnutým hashováním):

| Fáze | podíl času (výchozí SSE) | podíl (po AVX-512) |
|---|---|---|
| hashování (SHA-256 + RIPEMD-160) | ~68 % | ~63 % |
| generování bodů (EC field aritmetika) | ~21 % | ~21 % |
| dávková inverze | ~4 % | ~6 % |
| endomorfismy + režie | zbytek | zbytek |

Hashování je i po zrychlení dominantní, ale je už maximálně paralelizované
(16-way AVX-512) a tento CPU nemá SHA-NI, takže dále ho na CPU zrychlit nelze.

## Provedené změny (commit po commitu)

### 1. Optimalizační flagy (`-O2 -mssse3` → `-O3 -march=native -funroll-loops`)
Změřeno: **14,7 → 19,5 Mkey/s (+33 %)** bez zásahu do kódu. GCC přeloží
stávající SSE intrinsics do 3operandových AVX (VEX) instrukcí a lépe rozbalí
smyčky. Přidán `make portable=1` pro generickou binárku (`-mssse3`) na neznámé
CPU (runtime dispatch stejně vybere jen SSE cestu).

### 2. Oprava chybného `sha256_checksum` (kritická oprava správnosti)
Skalární `sha256_checksum` používal jednoblokový pomocník `Transform2`, který
počítal **špatný** double-SHA256 (ověřeno proti referenci: `b8b4cd62` místo
správného `fa55b64a`). Každá vypsaná vanity adresa i WIF privátní klíč tak
nesly **neplatný Base58Check checksum** a `-check` selhával na všech testech.
Pod `-O3 -march=native` navíc byl pomocník nedeterministický, což vyvolávalo
falešná varování „wrong private key generated". Nahrazeno důvěryhodnou
double-hash cestou přes `CSHA256`. Mimo horkou smyčku (volá se jen při zásahu
prefixu a při formátování výstupu), takže bez dopadu na výkon. 4-way SSE
checksum byl správný a zůstal beze změny.

### 3. AVX2 8-way hashování
`hash/sha256_avx2.cpp` a `hash/ripemd160_avx2.cpp` — věrný 8-lane port
4-way SSE jader (`__m256i`), ověřeno bit-identicky proti skalární referenci.
8-wide `GetHash160`, `checkAddressesAVX2`, runtime dispatch přes
`__builtin_cpu_supports("avx2")`. **~19,5 → ~30 Mkey/s.**

### 4. AVX-512 16-way hashování
`hash/sha256_avx512.cpp` a `hash/ripemd160_avx512.cpp` — 16-lane port s
nativními rotacemi (`_mm512_ror/rol_epi32`) a ternary-logic NOT. 16-wide
`GetHash160_16`, `checkAddressesAVX512`, dispatch přes `avx512f` před AVX2
cestou. Izolovaně 1,7× nad AVX2 (i přes AVX-512 downclock). **~30 → ~38 Mkey/s.**
Ověřeno proti referenci i nezávislým Python přepočtem nalezených klíčů
(P2PKH komprimovaný/nekomprimovaný, P2SH, BECH32).

### 5. Branchless `ModAdd`/`ModSub`/`ModDouble`
Datově závislá redukce (`if negative/overflow`) nahrazena maskovaným
přičtením/odečtením P (`CondAddP`/`CondSubP`), jako v GPU kódu. Volá se
několikrát na generovaný bod. Na `-O3` GCC už větve if-convertuje, takže zisk
je malý (~2-3 %) ale konzistentní, a garantuje branchless i na nižších úrovních
optimalizace / jiných překladačích.

### 6. Oprava Makefile
Explicitní pravidla pro `*_avx2.o` byla před cílem `all`, takže se stala
výchozím cílem `make` a čistý build spadl po prvním objektu. Přesunuto za `all`.

### 7. LTO + oprava strict-aliasing UB (~+9 %)
Inspirováno sesterským projektem mc-keygen (`lto=true, codegen-units=1`).
Dvě provázané změny:
- `-fno-strict-aliasing` — **oprava správnosti**. Adresní/hashovací kód všude
  type-punuje (`*(prefix_t *)hash160`, makra `KEYBUFF*`), což je UB pod
  výchozím `-fstrict-aliasing` na `-O3`. Bez LTO to náhodou funguje, ale
  s cross-TU inliningem (LTO) se to miscompiluje (všechny adresní vektory
  v `-check` selhaly se špatnou adresou). Tento flag je správný nezávisle na LTO.
- `-flto -fno-semantic-interposition` na `-march=native` buildu — **~+9 %**
  (~44,5 → ~48,6 Mkey/s, 4 vlákna). Jen pro native build (základní ISA je už
  AVX-512, takže LTO nemůže inlinovat širší kernel do užšího volajícího);
  portable build nechává LTO vypnuté.

### 8. PGO build (volitelný, `scripts/pgo-build.sh`)
Také z mc-keygen. Skript udělá instrumentovaný build, krátký reprezentativní
běh a rebuild s profilem, pak **změří plain vs PGO na daném stroji** a řekne,
který vyhrál. Efekt je mikroarchitekturně specifický: na tomto stroji byl PGO
o ~1-2 % **pomalejší**, takže výchozí build ho nepoužívá. Skript je tu proto,
aby si ho uživatel vyzkoušel na svém CPU (mc-keygen hlásí +16 % na Ice Lake,
−8 % na Broadwellu).

## Inspirace z mc-keygen (sesterský projekt)

[mc-keygen](https://github.com/kybl/mc-keygen) je vysoce CPU-optimalizovaný
generátor Ed25519 klíčů. Sdílí většinu strategie s VanitySearch a několik
technik se přeneslo:

| Technika mc-keygen | Stav ve VanitySearch |
|---|---|
| +8B adiční řetězec místo plného scalar-mult | **už měl** (`startP ± i*G`) |
| Montgomery dávková inverze (1 inverze / skupinu) | **už měl** (`IntGroup::ModInv`) |
| runtime SIMD dispatch (AVX-512 → AVX2 → scalar) | **doplněno** pro hashování |
| LTO + codegen-units=1 | **doplněno** (+9 %) |
| PGO skript s měřením plain vs PGO | **doplněno** (`scripts/pgo-build.sh`) |
| kernel prefiltr (odmítne 99 % kandidátů) | ekvivalent: 16-bit prefix tabulka |
| **SIMD napříč klíči pro field aritmetiku** | **nepřeneseno** — viz níže |

Zásadní rozdíl: mc-keygen je Ed25519, který **nemá efektivní endomorfismus**,
takže veškerou paralelizaci získává během celé field aritmetiky přes SIMD lanes
(8 nezávislých klíčů v AVX-512). VanitySearch je secp256k1 a využívá
**endomorfismus (λ) + symetrii**, což dává 6 kandidátů z jednoho spočítaného
bodu — to Ed25519 neumí. Obě strategie jsou pro svou křivku dobré; VanitySearch
je na 4 jádrech AVX-512 srovnatelně rychlý (~38-48 Mkey/s) jako mc-keygen (~34).

## Identifikovaný další velký pákový bod (budoucí práce)

Po zrychlení hashování je běh z ~63 % hash-bound a hashování je maximálně
paralelizované. Zbylých ~37 % je scalar field aritmetika (generování bodů +
endomorfismy). Jediná cesta, jak ji výrazně zrychlit, je **SIMD napříč klíči**
jako v mc-keygen: přepsat `ModMulK1`/`ModSquareK1`/`ModSub`/`ModAdd` na 8-wide
AVX-512 verzi (radix 2^51 nebo 5×52-bit s **AVX-512 IFMA** `vpmadd52`,
který tento „Ice-Lake-class" CPU má) a generovat body ve smyčce po 8 lanech
(iterace jsou navzájem nezávislé). Realistický odhad ~1,3-1,4× navíc
(~37 % → ~10 % času). Je to ale velký a chybově náchylný kus (carry
propagace, líná redukce, secp redukce `0x1000003D1` napříč lanes) — vyžadoval
by vlastní SIMD field knihovnu s důkladnými differenciálními testy proti
skalární referenci. Nebylo provedeno v tomto kole kvůli poměru
riziko/přínos, ale je to jasně další krok.

## Co se NEvyplatilo / nebylo provedeno a proč

- **Přepis `ModMulK1`/`ModSquareK1` na MULX/ADCX/ADOX (dvě carry větve):**
  Stávající kód je už dobře optimalizovaná standardní rychlá secp256k1 redukce
  (konstanta `0x1000003D1`). Field mult tvoří ~25 % času; realistický zisk
  ~3-5 % celkově za cenu vysokého rizika subtilní carry chyby v kryptografickém
  kódu. Nevyplatí se destabilizovat ověřený kód.
- **Bitmapa místo 1MB tabulky prefixů:** vyhledání prefixu je jen ~1 % času
  (běh je hash-bound), přínos <1 %.
- **Zvětšení `CPU_GRP_SIZE`:** inverze je jen ~6 % času, amortizace je už
  dostatečná; větší skupina zhorší cache.
- **SHA-NI:** tento CPU ho nemá; na Zenu/Ice Lake+ by se vyplatilo přidat do
  dispatch vrstvy (2× SHA-256 blok).

## Poznámky k přenositelnosti

AVX2/AVX-512 soubory se překládají s `-mavx2` resp. `-mavx512f -mavx512bw`, ale
volají se pouze po runtime CPUID kontrole (`__builtin_cpu_supports`). Binárka
tedy poběží i na CPU bez těchto rozšíření (spadne na SSE, resp. AVX2 cestu).
Wildcard/pattern hledání zůstává na SSE cestě.
