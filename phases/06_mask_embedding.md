# Fáza 06: Embeddingy masiek

`contract_version: 1`

## Účel

Ku každej maske vypočítať sémantický vektor v spoločnom priestore obrazu a textu (CLIP-like).
Vďaka nemu sa dá neskôr nájsť objekt podľa ľubovoľného textu („tá zelená rastlina pri okne“),
nielen podľa labelu zo slovníka, a porovnať objekt s obrázkovým promptom.

Oddelené od 05, aby sa dal vymeniť model embeddingov bez novej segmentácie.

## Kontrakt

**Vstup**
- `01_sfm/images_undistorted/`, `01_sfm/cameras.json`
- `05_segmentation/detections.json`, `05_segmentation/masks/`
- `04_tagging/tags.json` (slovník), iba ak `compute_label_scores: true`

**Výstup** (`06_mask_embedding/`)
- `embeddings.npy` — `float16[M, D]`, L2-normalizované.
- `index.json` — `{"contract_version": 1, "mask_ids": ["f000001_m000", …], "skipped": […]}`;
  riadok i v `.npy` patrí `mask_ids[i]`; `skipped` sú masky bez embeddingu (príliš malé).
  `mask_ids` + `skipped` sú presne všetky masky z 05.
- `meta.json` — `{"model": "hf:laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "dim": 1024,
  "normalized": true, "crop_strategy": "multi", "text_template": "a photo of a {}"}`.
  `model` určuje váhy; fáza 10 načíta pre text ten istý model.
- voliteľne `label_scores.json` — zero-shot klasifikácia ako druhý názor k labelu z detektora:
  `{"contract_version": 1, "labels": [slovník 04], "text_template": "...",
  "scores": {"<mask_id>": {"best": "sofa", "probs": {"<label>": p, …}}}}` (softmax nad celým
  slovníkom s teplotou modelu, súčet 1).
- `report.json` — počty masiek, časy, podiel masiek, kde sa zero-shot zhoduje s labelom z 05.
- `manifest.json`.

## Postup

Pre každú masku:
1. **Výrezy** (crop strategy, overené v ConceptGraphs a OpenMask3D):
   - výrez bounding boxu so zväčšeným okrajom (kontext),
   - výrez s pozadím mimo masky vyplneným neutrálnou farbou alebo rozmazaním (izolovaný objekt),
   - voliteľne viac mierok okraja.
2. Výrezy zmenšiť na vstup modelu, zakódovať obrazovým enkodérom.
3. Vážený priemer embeddingov výrezov, L2 normalizácia.
4. Masky, ktorých bbox má obe strany menšie ako `min_crop_px`, sa vynechajú (`skipped`
   v `index.json`).

Výrezy sú štvorcové okolo stredu bboxu (strana = dlhšia strana bboxu × (1 + 2 × okraj)),
časť mimo snímky je sivá. Pri `multi` sa priemerujú dva výrezy s rovnakou váhou: `bbox`
(okraj `bbox_padding`) a `masked` (okraj `masked_padding`, pozadie mimo masky sivé alebo
rozmazané podľa `mask_background`).

Voliteľne zero-shot: text embeddingy slovníka (`text_template`) → kosínusová podobnosť → softmax.

## Implementácie

| Názov | Popis | Licencia |
|---|---|---|
| `openclip` (primárna) | OpenCLIP ViT-H-14 (LAION-2B) alebo ViT-L-14; robustný štandard. Váhy z repozitárov `laion/CLIP-*` na HF, načítané cez `transformers` (bez balíka `open_clip_torch`) | MIT (kód), váhy MIT |
| `siglip` (alternatíva) | SigLIP / SigLIP 2 cez `open_clip` alebo `transformers`; často lepšie zero-shot | Apache 2.0 |

Pri voľbe modelu over licenciu **konkrétnych váh** a zapíš ju do `docs/licenses.md`.

## Konfigurácia

```yaml
mask_embedding:
  impl: openclip
  model: laion/CLIP-ViT-H-14-laion2B-s32B-b79K
  model_dir: checkpoints/huggingface
  device: auto
  fp16: false               # GTX 16xx: fp16 ~4× pomalšie (bez tensor cores)
  text_device: cpu          # textová vetva kóduje iba slovník
  crop_strategy: multi      # bbox | masked | multi
  bbox_padding: 0.15
  masked_padding: 0.05
  mask_background: gray     # gray | blur
  min_crop_px: 32
  batch_size: 16
  compute_label_scores: true
  text_template: "a photo of a {}"
```

## Hardvér

GPU odporúčaná, ViT-L/B zvládne aj CPU pomalšie. ViT-H má obrazovú vetvu ~2,5 GB (fp32)
a textovú ~1,4 GB; textová kóduje iba slovník (desiatky fráz), preto beží na CPU
(`text_device: cpu`) a na GPU ide iba obrazová. Na 4 GB GPU vo fp32 s dávkou 16 je špička
~2,6 GB; dávka 64 pri obsadenej VRAM (iné aplikácie) pretečie do systémovej pamäte a beh je
niekoľkonásobne pomalší.

**fp16 na GTX 16xx nepoužívať:** čip TU117 nemá tensor cores a maticové násobenie vo fp16
je ~4× pomalšie ako vo fp32 (GTX 1650: 0,40 vs. 1,62 TFLOPS, 2048 × 2048). Na GPU s tensor
cores (RTX) je `fp16: true` rýchlejšie a polovičné na VRAM. Výsledky sú prakticky rovnaké
(kosínus embeddingov fp16 vs. fp32 ≥ 0,9997, rovnaký zero-shot label pri 99,8 % masiek).

## Validácia výstupu

- Počet riadkov `embeddings.npy` = dĺžka `mask_ids`; každý `mask_id` existuje v 05.
- Normy vektorov ∈ [0.99, 1.01], žiadne NaN.
- `meta.json.dim` = šírka matice.

## Riziká a otvorené otázky

- CLIP je slabý na jemné atribúty a priestorové vzťahy („ľavá stolička“) — tie rieši graf (08, 10),
  nie embedding.
- Čiastočne zakryté alebo orezané objekty dávajú horšie embeddingy → v 07 vážiť podľa
  `mask_score`, `area_px` a `touches_border`.

## Evaluácia

Zero-shot presnosť labelov voči ručnej anotácii; retrieval presnosť (text → správna maska).

## Stav

Implementácia: `src/scenepipe/phases/p06_mask_embedding/` (`common.py`, `openclip.py`),
test: `tests/contracts/test_p06.py` (falošný enkodér = priemerná farba výrezu, beží na CPU
za ~1 s). Beží v `sp-gs` vo WSL (transformers už sú tam kvôli 05).

- [x] načítanie masiek a výrezy
- [x] OpenCLIP enkódovanie (transformers), dávkovanie
- [x] zero-shot label scores
- [x] kontraktový test
- [x] stiahnutie váh (ViT-H, 3,9 GB) a overenie na reálnej scéne
- [x] overenie na Mip-NeRF 360 *counter* (2026-11-06, GTX 1650 vo WSL, súčasne bežal
  Android emulátor s ~1,4 GB VRAM; vstup: 05 s profilom `fast`, 80 snímok, 2 050 masiek,
  slovník 42 tagov, 2 výrezy na masku):

  | Nastavenie | Čas embeddingov | Rýchlosť |
  |---|---|---|
  | fp16, dávka 64 (prvá verzia) | 3 617 s | 1,1 výrezu/s |
  | **fp32, textová vetva na CPU, dávka 16 (default)** | 2 304 s | 1,8 výrezu/s |

  (Samostatný benchmark bez I/O: fp32 2,6 výrezu/s, fp16 1,1, nezávisle od veľkosti dávky.)
  Kosínusová podobnosť masiek v jednom tracku 0,83, v rôznych trackoch 0,59 (pre prah
  `merge_sim` 0,75 v 07). Zero-shot sa zhoduje s labelom detektora pri 44 % masiek; časť
  nezhôd sú synonymá (*bowl / mixing bowl*, *platter / tray*, *blender / pitcher*), časť
  chyby (*apple → orange*, *orange → onion*).
- [x] ablácia výrezov na *counter* (2026-11-06, rovnaký stroj; 07 `gsplat_vote` s default
  nastavením na každom výstupe, IoU podľa `scenepipe.eval.lifting_consistency`):

  | `crop_strategy` | Čas embeddingov | Zero-shot zhoda s detektorom | IoU 07 (priemer / medián) |
  |---|---|---|---|
  | **`multi` (default)** | 2 304 s | **44 %** | 0,712 / 0,823 |
  | `bbox` | 1 021 s | 32 % | 0,690 / 0,814 |
  | `masked` (`fast`) | 1 181 s | 40 % | 0,712 / 0,823 |

  Výrez s pozadím mimo masky je pre spájanie inštancií rovnako dobrý ako oba výrezy spolu
  za polovičný čas; kontext z `bbox` pridáva hlavne zero-shot presnosť. Profil `fast`
  používa `masked`.

## Zdroje

- Radford et al.: CLIP, ICML 2021; Cherti et al.: Reproducible scaling laws (OpenCLIP), CVPR 2023.
- Zhai et al.: SigLIP, ICCV 2023.
