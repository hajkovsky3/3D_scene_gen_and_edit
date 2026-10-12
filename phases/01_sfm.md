# Fáza 01: Structure-from-Motion

`contract_version: 1`

## Účel

Z neusporiadaných snímok odhadnúť pozície a parametre kamier a riedky 3D pointcloud scény.
Výstup slúži na inicializáciu gaussiánov (02), na lifting (06) a ako kamery pre renderovanie.
Súčasťou fázy je aj **undistorcia snímok** a **zarovnanie scény s gravitáciou** (+Z hore),
pretože vzťahy v grafe scény (napr. *on*) potrebujú vedieť, kde je hore.

## Kontrakt

**Vstup**
- `00_ingest/frames.json`, `00_ingest/images/`

**Výstup** (`01_sfm/`)
- `sparse/` — rekonštrukcia v **COLMAP formáte** (`cameras.bin`, `images.bin`, `points3D.bin`),
  už po undistorcii (model `PINHOLE`) a zarovnaní. Mnohé nástroje (gsplat, viewery) ju čítajú priamo.
- `images_undistorted/<frame_id>.jpg` — snímky zodpovedajúce `PINHOLE` kamerám.
- `cameras.json` — kamery v projektovom formáte (`docs/data-contracts.md`, sekcia 6).
  Toto je **primárne rozhranie** pre ďalšie fázy; COLMAP súbory sú pre kompatibilitu.
- `points.npz` — `xyz: float32[P,3]`, `rgb: uint8[P,3]`, `error: float32[P]`, `track_len: int32[P]`.
- `alignment.json` — `{"sim3_from_raw": 4×4 matica, "scale": s, "up_method": "...", "notes": ...}`.
- `report.json` — počet registrovaných snímok, priemerná reprojekčná chyba, dĺžka trackov,
  zoznam neregistrovaných `frame_id`.
- `manifest.json`.

## Postup

1. **Features**: SIFT (GPU) alebo ALIKED. Pre ťažšie scény ALIKED + LightGlue.
2. **Matching**: pri videu `sequential_matcher` (s loop detection), pri fotkách `exhaustive_matcher`
   (do ~300 snímok) alebo `vocab_tree_matcher`.
3. **Geometrická verifikácia**: odhad fundamentálnej / esenciálnej matice s RANSAC.
4. **Mapping**: globálny mapper (GLOMAP integrovaný v COLMAP 4.x) je rýchlejší, inkrementálny
   je robustnejší. Default `global`, fallback `incremental`, ak sa zaregistruje < 80 % snímok.
5. **Bundle adjustment** (súčasť mappingu).
6. **Undistorcia**: `image_undistorter` → `PINHOLE` kamery a nové snímky.
7. **Zarovnanie**:
   - smer hore: priemer „up“ vektorov kamier (−y osi kamier, ľudia držia telefón zhruba vzpriamene)
     ako počiatočný odhad, spresnený RANSAC fitom najväčšej roviny (podlaha) s normálou blízko
     tohto odhadu; alternatívne `colmap model_orientation_aligner`;
   - stred: medián pozícií kamier (alebo bodov) do počiatku;
   - mierka: rozptyl pozícií kamier ≈ 1.
   Výsledná sim(3) transformácia sa aplikuje na kamery aj body a uloží do `alignment.json`.
8. Export `cameras.json`, `points.npz`, `report.json`.

## Implementácie

| Názov | Popis | Licencia |
|---|---|---|
| `colmap` (primárna) | COLMAP 4.x cez `pycolmap` alebo CLI; SIFT/ALIKED, LightGlue, global/incremental mapper | BSD |
| `vggt` (alternatíva) | feed-forward VGGT, export do COLMAP formátu, voliteľný BA; vhodné pri málo snímkach | VGGT License; **iba checkpoint `VGGT-1B-Commercial`** (pôvodný je nekomerčný) — hraničné, pozri `docs/licenses.md` |

Neodporúčané pre tento projekt kvôli licencii: MASt3R / DUSt3R (CC BY-NC-SA),
SuperPoint váhy od Magic Leap (nekomerčné).

## Konfigurácia

```yaml
sfm:
  impl: colmap
  features: sift            # sift | aliked
  matcher: auto             # auto | sequential | exhaustive | vocab_tree
  matcher_backend: default  # default | lightglue
  mapper: global            # global | incremental
  fallback_incremental_below: 0.8
  single_camera: true       # true pre video z jedného telefónu
  align_up: floor_ransac    # camera_up | floor_ransac | colmap_aligner | none
  use_gpu: true
```

## Hardvér

COLMAP beží aj na CPU (pomalšie). GPU urýchli SIFT a matching. Pri 200 snímkach rádovo
minúty až desiatky minút. VGGT potrebuje GPU, pamäť rastie s počtom snímok.

## Validácia výstupu

- `cameras.json` má ≥ `min_registered` kamier, všetky matice 4×4, rotačná časť ortonormálna
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

- [ ] beh COLMAP (feature extraction, matching, global mapper)
- [ ] fallback na inkrementálny mapper
- [ ] undistorcia
- [ ] zarovnanie (+Z hore, normalizácia)
- [ ] export `cameras.json`, `points.npz`, `report.json`
- [ ] kontraktový test
- [ ] (voliteľné) implementácia `vggt`

## Zdroje

- Schönberger, Frahm: Structure-from-Motion Revisited, CVPR 2016.
- Pan et al.: Global Structure-from-Motion Revisited (GLOMAP), ECCV 2024.
- COLMAP dokumentácia: https://colmap.github.io/
- Wang et al.: VGGT: Visual Geometry Grounded Transformer, CVPR 2025.
