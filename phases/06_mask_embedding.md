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
  fp16: true
  crop_strategy: multi      # bbox | masked | multi
  bbox_padding: 0.15
  masked_padding: 0.05
  mask_background: gray     # gray | blur
  min_crop_px: 32
  batch_size: 64
  compute_label_scores: true
  text_template: "a photo of a {}"
```

## Hardvér

GPU odporúčaná (ViT-H ~ 4 GB VRAM pri inferencii), ViT-L/B zvládne aj CPU pomalšie.

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
- [ ] stiahnutie váh a overenie na reálnej scéne

## Zdroje

- Radford et al.: CLIP, ICML 2021; Cherti et al.: Reproducible scaling laws (OpenCLIP), CVPR 2023.
- Zhai et al.: SigLIP, ICCV 2023.
