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

**Výstup** (`06_mask_embedding/`)
- `embeddings.npy` — `float16[M, D]`, L2-normalizované.
- `index.json` — `{"contract_version": 1, "mask_ids": ["f000001_m000", …]}`; riadok i v `.npy`
  patrí `mask_ids[i]`.
- `meta.json` — `{"model": "open_clip:ViT-H-14/laion2b_s32b_b79k", "dim": 1024,
  "normalized": true, "crop_strategy": "multi", "text_template": "a photo of a {}"}`.
- voliteľne `label_scores.json` — pre každú masku podobnosť so všetkými pojmami zo slovníka
  (zero-shot klasifikácia ako druhý názor k labelu z detektora).
- `manifest.json`.

## Postup

Pre každú masku:
1. **Výrezy** (crop strategy, overené v ConceptGraphs a OpenMask3D):
   - výrez bounding boxu so zväčšeným okrajom (kontext),
   - výrez s pozadím mimo masky vyplneným neutrálnou farbou alebo rozmazaním (izolovaný objekt),
   - voliteľne viac mierok okraja.
2. Výrezy zmenšiť na vstup modelu, zakódovať obrazovým enkodérom.
3. Vážený priemer embeddingov výrezov, L2 normalizácia.
4. Masky menšie ako `min_crop_px` dostanú embedding s príznakom nízkej spoľahlivosti
   (alebo sa vynechajú — zapísať do `index.json` ako `skipped`).

Voliteľne zero-shot: text embeddingy slovníka (`text_template`) → kosínusová podobnosť → softmax.

## Implementácie

| Názov | Popis | Licencia |
|---|---|---|
| `openclip` (primárna) | OpenCLIP ViT-H-14 (LAION-2B) alebo ViT-L-14; robustný štandard | MIT (kód), váhy MIT |
| `siglip` (alternatíva) | SigLIP / SigLIP 2 cez `open_clip` alebo `transformers`; často lepšie zero-shot | Apache 2.0 |

Pri voľbe modelu over licenciu **konkrétnych váh** a zapíš ju do `docs/licenses.md`.

## Konfigurácia

```yaml
mask_embedding:
  impl: openclip
  model: ViT-H-14
  pretrained: laion2b_s32b_b79k
  crop_strategy: multi      # bbox | masked | multi
  bbox_padding: 0.15
  min_crop_px: 32
  batch_size: 64
  compute_label_scores: true
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

- [ ] načítanie masiek a výrezy
- [ ] OpenCLIP enkódovanie, dávkovanie
- [ ] zero-shot label scores
- [ ] kontraktový test

## Zdroje

- Radford et al.: CLIP, ICML 2021; Cherti et al.: Reproducible scaling laws (OpenCLIP), CVPR 2023.
- Zhai et al.: SigLIP, ICCV 2023.
