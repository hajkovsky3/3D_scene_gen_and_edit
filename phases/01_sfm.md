# Fáza 01: Structure-from-Motion

`contract_version: 2`

## Účel

Z neusporiadaných snímok odhadnúť pozície a parametre kamier a riedky 3D pointcloud scény.
Výstup slúži na inicializáciu gaussiánov (02), na lifting (06) a ako kamery pre renderovanie.
Súčasťou fázy je aj **undistorcia snímok** a **zarovnanie scény s gravitáciou** (+Z hore),
pretože vzťahy v grafe scény (napr. *on*) potrebujú vedieť, kde je hore.

## Kontrakt

**Vstup**
- `00_ingest/frames.json`, `00_ingest/images/`

**Výstup** (`01_sfm/`)
- `sparse/` — rekonštrukcia v **COLMAP formáte** (`cameras.bin`, `images.bin`, `points3D.bin`;
  COLMAP 4 pridáva aj `rigs.bin`, `frames.bin`), už po undistorcii (model `PINHOLE`)
  a zarovnaní. Mnohé nástroje (gsplat, viewery) ju čítajú priamo. Názov snímky v COLMAP je
  `<frame_id>.jpg`.
- `images_undistorted/<frame_id>.jpg` — snímky zodpovedajúce `PINHOLE` kamerám.
- `cameras.json` — kamery v projektovom formáte (`docs/data-contracts.md`, sekcia 6).
  Toto je **primárne rozhranie** pre ďalšie fázy; COLMAP súbory sú pre kompatibilitu.
- `points.npz` — `xyz: float32[P,3]`, `rgb: uint8[P,3]`, `error: float32[P]`, `track_len: int32[P]`.
- `alignment.json` — `{"sim3_from_raw": 4×4 matica, "scale": s, "up_method": "...", "notes": ...}`.
- `report.json` — `num_frames`, `num_registered`, `registered_ratio`, `num_points`,
  `mean_reprojection_error_px`, `mean_track_length`, `unregistered` (zoznam `frame_id`),
  `mapper` (ktorý mapper dal výsledok), `matcher`, `models` (počet snímok v každom nájdenom
  modeli; berie sa najväčší), `warnings`; od verzie 2 aj `num_pairs` (počet párovaných
  dvojíc), `match_device` (`cuda` | `cpu`) a `timings_s` (sekundy krokov `features`, `pairs`,
  `matching`, `mapping`, `undistortion`, `export`).
- `manifest.json`.

## Postup

1. **Features**: SIFT (GPU) alebo ALIKED. Pre ťažšie scény ALIKED + LightGlue.
   Kamera: ak `single_camera: true` a všetky snímky majú rovnaký rozmer aj `exif_focal_35mm`,
   jedna spoločná kamera (model `SIMPLE_RADIAL`), inak COLMAP `AUTO`. Ak `frames.json` obsahuje `exif_focal_35mm`,
   použije sa ako počiatočný odhad ohniska: `f_px = f35 * diag_px / 43.27`
   (snímky z fázy 00 už EXIF nemajú).
2. **Matching** (výber dvojíc snímok na párovanie):
   - `exhaustive`: všetky dvojice, O(n²); spoľahlivé, ale pri 200 snímkach na CPU ~1 h.
   - `sequential`: každá snímka s nasledujúcimi `sequential_overlap` snímkami + kvadratické
     posuny (2^k). Bez loop detection; na videu s medzerami sa scéna rozpadne (*izba*: 40 %).
   - `retrieval`: okno a kvadratické posuny ako `sequential` + pre každú snímku
     `retrieval_num_neighbors` najpodobnejších snímok mimo okna (uzavretie slučiek).
     Podobnosť: bag-of-words nad SIFT deskriptormi samotnej scény (RootSIFT, k-means
     s `retrieval_num_words` slovami na `retrieval_sample` deskriptoroch, tf-idf, kosínus),
     vlastný kód v numpy, nič sa nesťahuje. Dvojice sa zapíšu do `pairs.txt` (pracovný adresár)
     a párujú cez `match_image_pairs`. Počet dvojíc rastie lineárne (~13 na snímku pri oknách 5 + 5; predtým 10 + 10, ~20 na snímku).
   - `auto`: `exhaustive` pri ≤ `auto_exhaustive_max` snímkach (default 100), inak `retrieval`.
   - `vocab_tree` a loop detection v COLMAP potrebujú stiahnuť vocabulary tree, zatiaľ nie sú zapnuté.

   Všetky matchery vyrobia zoznam dvojíc (aj `sequential`, rovnakou funkciou ako okno
   v `retrieval`) a párujú sa po dávkach (~20 volaní), aby sa dal hlásiť priebeh.

   **Párovanie na GPU** (`use_gpu: true` a PyTorch vidí CUDA GPU, napr. prostredie `sp-gs`):
   vyhľadanie najbližších deskriptorov beží v PyTorch (`torch_match.py`) podľa rovnakých
   pravidiel ako SIFT matcher COLMAPu (uhol normalizovaných deskriptorov ≤ `max_distance` 0,7,
   ratio test 0,8, vzájomne najlepšia zhoda), zhody sa zapíšu do databázy a geometrickú
   verifikáciu robí COLMAP (`verify_matches`). Dvojice jednej snímky sa počítajú naraz
   (dávkové násobenie matíc do 512 MB). Zhody sa s CPU matcherom COLMAPu zhodujú na ~99 %
   (pár zhôd navyše pri dvojiciach s < 15 zhodami, ktoré verifikácia aj tak zahodí).
   Na GTX 1650: ~5 ms na dvojicu pri 4 096 bodoch, ~25 ms pri 8 192 (CPU COLMAP ~50 ms,
   resp. ~180 ms). Bez CUDA sa páruje na CPU cez `match_image_pairs` ako predtým.

   Databáza COLMAPu a pomocné súbory sú počas behu v systémovom temp adresári (lokálny disk),
   nie v `01_sfm/_work/`: databáza sa zapisuje a číta pri každej dvojici a vo WSL je
   `/mnt/c` cez 9p rádovo pomalší (verifikácia stála na I/O).

   `sift_max_image_size` zmenší obrázok pred extrakciou SIFT (rýchlejšia extrakcia aj párovanie,
   výstupné snímky a kamery sú v pôvodnom rozlíšení).
3. **Geometrická verifikácia**: odhad fundamentálnej / esenciálnej matice s RANSAC.
4. **Mapping**: globálny mapper (GLOMAP integrovaný v COLMAP 4.x) je rýchlejší, inkrementálny
   je robustnejší. Default `global`, fallback `incremental`, ak sa zaregistruje < 80 % snímok.
5. **Bundle adjustment** (súčasť mappingu). Potom sa z najväčšieho modelu vyradia **odľahlé
   kamery**: vzdialenosť od mediánu stredov kamier > `max_camera_dist_ratio` × medián týchto
   vzdialeností (default 5; `0` vypne). Sú to zle umiestnené snímky (typicky rozmazané pri
   rýchlom otočení); inak by ovládli normalizáciu v kroku 7 aj mierku scény vo fáze 02.
   Vyradené snímky sú v `unregistered` a vo `warnings`.
6. **Undistorcia**: `image_undistorter` → `PINHOLE` kamery a nové snímky.
7. **Zarovnanie**:
   - smer hore: priemer „up“ vektorov kamier (−y osi kamier, ľudia držia telefón zhruba vzpriamene)
     ako počiatočný odhad (`camera_up`), spresnený RANSAC fitom najväčšej roviny (podlaha,
     `floor_ransac`): normála do 25° od odhadu, ≥ 80 % kamier nad rovinou (vylúči strop
     a stoly), prah vzdialenosti 1 % mediánovej vzdialenosti bodov od kamier, aspoň 5 % bodov
     ako inliery. Ak sa rovina nenájde, fallback na `camera_up` (zapíše sa do `notes`).
     `none` = bez rotácie (iba na ladenie, kontrola smeru hore sa vtedy preskočí).
     `colmap_aligner` (`model_orientation_aligner`) pycolmap nesprístupňuje, zatiaľ
     neimplementované;
   - rotácia: najmenšia rotácia, ktorá zobrazí smer hore na +Z (natočenie okolo Z sa nerieši);
   - stred: medián pozícií kamier do počiatku;
   - mierka: RMS vzdialenosť kamier od ich mediánu = 1.
   Výsledná sim(3) transformácia `x' = s R (x − c)` sa aplikuje na kamery aj body a uloží
   do `alignment.json`.
8. Export `cameras.json`, `points.npz`, `report.json`. Pomocné súbory (databáza, modely
   pred undistorciou) sú v dočasnom adresári a na konci sa zmažú (`keep_work: true` ich skopíruje do `01_sfm/_work/`).

## Implementácie

| Názov | Popis | Licencia |
|---|---|---|
| `colmap` (primárna) | COLMAP 4.x cez `pycolmap` (wheel z PyPI, iba CPU) + voliteľne párovanie v PyTorch na GPU; zatiaľ SIFT, sequential/exhaustive, global/incremental mapper | BSD; wheel pribaľuje SuiteSparse (časti GPL/LGPL) — HRANIČNÉ, pozri `docs/licenses.md` |
| `vggt` (alternatíva) | feed-forward VGGT, export do COLMAP formátu, voliteľný BA; vhodné pri málo snímkach | VGGT License; **iba checkpoint `VGGT-1B-Commercial`** (pôvodný je nekomerčný) — hraničné, pozri `docs/licenses.md` |

Iba ako **nekomerčné** alternatívy (nie primárna cesta, pozri `docs/licenses.md`):
MASt3R / DUSt3R (CC BY-NC-SA; vhodné pri málo snímkach), pôvodný checkpoint VGGT-1B
(FAIR Noncommercial), SuperPoint váhy od Magic Leap (hloc).

## Konfigurácia

```yaml
sfm:
  impl: colmap
  features: sift            # sift | aliked
  matcher: auto             # auto | sequential | exhaustive | retrieval | vocab_tree
  matcher_backend: default  # default | lightglue
  mapper: global            # global | incremental
  fallback_incremental_below: 0.8
  single_camera: true       # true pre video z jedného telefónu
  align_up: floor_ransac    # camera_up | floor_ransac | none
  use_gpu: true             # párovanie na GPU cez PyTorch, ak je CUDA; inak COLMAP na CPU
  max_num_features: 8192    # menej pre slabší HW
  sift_max_image_size: -1   # -1 = COLMAP default (3200)
  sequential_overlap: 5
  retrieval_num_neighbors: 5
  retrieval_num_words: 1024
  retrieval_sample: 100000
  auto_exhaustive_max: 100
  max_camera_dist_ratio: 5.0  # vyradiť kamery ďalej než 5 × medián vzdialenosti; 0 = vypnuté
  min_registered: 10
  keep_work: false
```

`aliked`, `lightglue` a `vocab_tree` potrebujú stiahnuť váhy alebo vocabulary tree.
Kým ich autor neschváli, implementácia pri nich skončí chybou.

## Hardvér

COLMAP beží aj na CPU (pomalšie). GPU (PyTorch) urýchli párovanie, SIFT extrakcia a verifikácia ostávajú na CPU. Pri 200 snímkach rádovo
minúty až desiatky minút. VGGT potrebuje GPU, pamäť rastie s počtom snímok.

## Validácia výstupu

- `cameras.json` má ≥ `min_registered` kamier (default 10), všetky matice 4×4, rotačná časť ortonormálna
  (det ≈ 1), intrinsika kladná, `cx, cy` vo vnútri obrázka.
- `points.npz` neprázdny, bez NaN.
- Po zarovnaní: väčšina kamier má −y os kamery s kladnou zložkou Z (kontrola smeru hore).
- Priemerná reprojekčná chyba < 2 px (inak varovanie, nie chyba).
- COLMAP `sparse/` sa dá načítať cez `pycolmap` a počty sedia s `cameras.json`.

## Riziká a otvorené otázky

- Málo textúry (biele steny), opakujúce sa vzory, zrkadlá → neregistrované snímky alebo
  rozpadnutá rekonštrukcia (viac komponentov). COLMAP 4.2 vracia viac modelov; berieme
  najväčší a ostatné zapíšeme do reportu.
- Zarovnanie podlahy zlyhá vo vonkajších scénach bez roviny → fallback na `camera_up`.
- Mierka je ľubovoľná. Ak bude treba metre (napr. pre realistickú veľkosť vkladaného objektu),
  dá sa doplniť odhad z rozmerov známych objektov — zatiaľ mimo rozsah.

## Evaluácia

Podiel registrovaných snímok, reprojekčná chyba, čas behu. Pri datasetoch so známymi pózami
(napr. Mip-NeRF 360 má COLMAP pózy) chyba rotácie/translácie po zarovnaní.

## Stav

Implementácia: `src/scenepipe/phases/p01_sfm/colmap.py` (+ `align.py`),
test: `tests/contracts/test_p01.py` (syntetická miestnosť so známymi kamerami,
`tests/contracts/synthetic_room.py`). Inštalácia: `pip install -e ".[dev,sfm]"`.

- [x] beh COLMAP (SIFT, sequential/exhaustive matching, global mapper)
- [x] fallback na inkrementálny mapper
- [x] undistorcia
- [x] zarovnanie (+Z hore, normalizácia)
- [x] export `cameras.json`, `points.npz`, `report.json`
- [x] kontraktový test (30 syntetických snímok: všetky zaregistrované, ohnisko ±3 %,
  sklon kamier voči +Z ako v skutočnosti; ~11 s na CPU)
- [x] overenie na reálnej scéne *izba* (2026-10-16, 208 snímok z fázy 00, video zvnútra izby):
  `sequential` → 83/208 (40 %), 7 modelov (medzery po rýchlych otočeniach, bez loop detection);
  `exhaustive` → 186/208 (89 %), 30 823 bodov, reprojekčná chyba 0,61 px; ~75 min na CPU
  (takmer všetko párovanie)
- [x] matcher `retrieval` (okno + bag-of-words vyhľadanie podobných snímok), `auto` ho volí
  nad 100 snímok namiesto `sequential`. *izba* s `configs/fast.yaml` (4 096 SIFT bodov,
  1 024 px): 4 156 dvojíc namiesto 21 528, 194/208 registrovaných (93 %), 0,66 px, **5 min**
  namiesto 75 min s `exhaustive`. (Beh s dopĺňaním medzier v 00 hlásil 224/242, ale bol
  degenerovaný kvôli 5 odľahlým kamerám, pozri `phases/00_ingest.md`, Stav.)
- [x] párovanie na GPU v PyTorch + verifikácia v COLMAPe, databáza v lokálnom temp adresári,
  `timings_s` v reporte (2026-10-17, beh vo WSL v `sp-gs`, GTX 1650). Presnosť voči GT pózam
  Mip-NeRF 360 (`python -m scenepipe.eval.sfm_poses`, medián chyby rotácie / polohy v % rozptylu
  kamier; všetky behy registrovali všetky snímky):

  | Scéna | Konfigurácia | Dvojice | Čas | Rotácia | Poloha |
  |---|---|---|---|---|---|
  | room (311) | pôvodne: exhaustive, 8 192, CPU | 48 205 | 3 h 10 min | 0,035° | 0,026 % |
  | room | default, GPU (okná 10 + 10) | 6 509 | 19 min | 0,031° | 0,023 % |
  | room | `fast`, GPU (10 + 10) | 6 448 | 9,2 min | 0,029° | 0,036 % |
  | room | `fast`, GPU, okná 5 + 5 | 4 068 | 6,9 min | 0,045° | 0,043 % |
  | room | `fast`, GPU, 2 048 bodov | 6 425 | 8,5 min | 0,049° | 0,046 % |
  | counter (240) | pôvodne: exhaustive, 8 192, CPU | 28 680 | 4 h 23 min | 0,012° | 0,012 % |
  | counter | default, GPU (10 + 10) | 4 663 | 26 min | 0,017° | 0,015 % |
  | counter | `fast`, GPU (10 + 10) | 4 758 | 9,5 min | 0,020° | 0,023 % |
  | counter | `fast`, GPU, okná 5 + 5 | 2 977 | 6,8 min | 0,021° | 0,024 % |
  | counter | `fast`, GPU, 2 048 bodov | 4 825 | 4,4 min | 0,023° | 0,029 % |

  Rozdiely sú v stotinách stupňa, default má preto okná 5 + 5. Menej bodov (2 048) ušetrí
  málo, lebo extrakcia a mapping trvajú rovnako. *izba* (225 snímok, `fast`): pôvodne 365 s
  (CPU, Windows) → 266 s (GPU, 10 + 10) → **171 s** (GPU, 5 + 5; párovanie + verifikácia 44 s,
  extrakcia 45 s, mapping 59 s); 02 (`fast`) PSNR 22,57 / SSIM 0,860 / LPIPS 0,402 oproti
  pôvodným 22,27 / 0,860 / 0,387. Pozor: úsek *izby* 33.–55. s umiestni každá konfigurácia
  inak (aj exhaustive s 8 192 bodmi, 23 min), globálny mapper je tam nejednoznačný; zhoda póz
  medzi behmi preto na *izbe* nie je meradlo kvality. Vo WSL je extrakcia SIFT ~1,5× pomalšia
  než natívne na Windowse (61 vs. 40 s pri `fast`).
- [ ] ALIKED / LightGlue / vocab tree (čaká na schválenie stiahnutia váh)
- [ ] (voliteľné) implementácia `vggt`

## Zdroje

- Schönberger, Frahm: Structure-from-Motion Revisited, CVPR 2016.
- Pan et al.: Global Structure-from-Motion Revisited (GLOMAP), ECCV 2024.
- COLMAP dokumentácia: https://colmap.github.io/
- Wang et al.: VGGT: Visual Geometry Grounded Transformer, CVPR 2025.
