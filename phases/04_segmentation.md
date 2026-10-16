# Fáza 04: 2D segmentácia inštancií

`contract_version: 1`

## Účel

Pre každú snímku nájsť objekty zo slovníka (03) a vytvoriť ich presné masky. Ak to model
dokáže, priradiť rovnakému objektu naprieč snímkami rovnaký `track_id` — výrazne to
zjednoduší spájanie inštancií v 3D (06).

## Kontrakt

**Vstup**
- `01_sfm/cameras.json`, `01_sfm/images_undistorted/` — pracuje sa s undistortovanými snímkami,
  aby masky presne sedeli na kamery. Iba registrované snímky.
- `03_tagging/tags.json` (`vocabulary`)

**Výstup** (`04_segmentation/`)
- `detections.json`:
  ```json
  {
    "contract_version": 1,
    "has_tracks": true,
    "detections": [
      {"mask_id": "f000001_m000", "frame_id": "f000001",
       "label": "sofa", "label_scores": {"sofa": 0.82, "chair": 0.11},
       "det_score": 0.82, "mask_score": 0.95,
       "bbox_xyxy": [120, 340, 980, 860], "area_px": 312400,
       "track_id": "t0003", "touches_border": false}
    ]
  }
  ```
- `masks/<frame_id>.json` — zoznam `{mask_id, rle}` (COCO RLE, rozmer undistortovanej snímky).
- `previews/` — niekoľko snímok s prekreslenými maskami (kontrola očami).
- `manifest.json`.

Ak model nevie sledovať objekty, `has_tracks: false` a `track_id: null`; spájanie potom
spraví fáza 06.

## Postup

1. **Detekcia**: Grounding DINO so slovníkom (po dávkach, ak je dlhý) → bounding boxy s labelmi.
   Prahy `box_threshold`, `text_threshold`. NMS naprieč labelmi.
2. **Segmentácia**: SAM 2 s boxmi ako promptom → masky + skóre kvality.
3. **Sledovanie** (voliteľné, odporúčané): snímky zoradiť do sekvencie (pri videu časové poradie,
   pri fotkách poradie podľa covisibility z 01) a použiť **SAM 2 video prediktor** na šírenie
   masiek. Detekcia sa púšťa každých `redetect_every` snímok; nové objekty sa pridajú ako nové
   tracky, existujúce sa spárujú podľa IoU masiek.
4. **Čistenie**: zahodiť príliš malé masky (`min_area_frac`), vyplniť malé diery, odstrániť
   malé izolované komponenty; označiť masky dotýkajúce sa okraja (`touches_border`), lebo objekt
   je pravdepodobne orezaný.
5. **Prekryvy**: masky sa môžu prekrývať (vankúš na gauči). Neriešiť tu, iba uložiť; 06 to rieši
   v 3D. Voliteľne vytvoriť aj neprekryvnú „label mapu“ (menšia maska vyhráva).

## Implementácie

| Názov | Popis | Licencia |
|---|---|---|
| `gdino_sam2` (primárna) | Grounding DINO (Swin-T/B) + SAM 2 (image + video prediktor), postup ako v repozitári Grounded-SAM-2 | Apache 2.0 (kód aj váhy) |
| `sam3` (alternatíva) | SAM 3: detekcia, segmentácia a sledovanie podľa textového konceptu jedným modelom | SAM License — vlastná, nie OSI, váhy za prihlásením na HF; **hraničné**, pozri `docs/licenses.md` |
| `sam2_auto` (alternatíva) | SAM 2 automatic mask generator bez textu (všetko, čo je objekt), labely dodá až 05/07 cez CLIP | Apache 2.0 |

Pozor: Grounding DINO 1.5/1.6 a DINO-X **nie sú** free to use (iba platené API) — nepoužívať.
Pozor: Grounded-SAM negeneruje CLIP embeddingy; to je samostatná fáza 05.

## Konfigurácia

```yaml
segmentation:
  impl: gdino_sam2
  gdino_model: swint_ogc
  sam2_model: sam2.1_hiera_large   # small/tiny pre slabší HW
  box_threshold: 0.30
  text_threshold: 0.25
  vocab_batch: 20
  tracking: true
  redetect_every: 10
  min_area_frac: 0.001
  max_masks_per_frame: 100
```

## Hardvér

GPU. SAM 2 large + Grounding DINO ~ 8–10 GB VRAM; `sam2.1_hiera_tiny` výrazne menej.
Video prediktor drží stav v pamäti — pri dlhých sekvenciách spracovávať po úsekoch.

## Validácia výstupu

- Každá maska má platný `frame_id` a dekóduje sa na rozmer snímky z `cameras.json`.
- `mask_id` unikátne, `label` je zo slovníka (alebo `null` pri `sam2_auto`).
- `bbox_xyxy` a `area_px` sedia s dekódovanou maskou.
- Ak `has_tracks`, každý `track_id` sa vyskytuje najviac raz na snímku.

## Riziká a otvorené otázky

- Rovnaký objekt pod rôznymi labelmi v rôznych snímkach (gauč/kreslo) → rieši sa hlasovaním v 06/07.
- Časti vs. celky (vankúš vs. gauč) → hierarchia sa nerieši tu; 06 a 07 potrebujú pravidlo.
- Drift pri sledovaní (maska „preskočí“ na iný objekt) → redetekcia a kontrola konzistencie
  labelu v tracku.

## Evaluácia

mIoU masiek na pár ručne anotovaných snímkach; konzistencia trackov (počet prerušení).

## Stav

- [ ] Grounding DINO detekcia so slovníkom
- [ ] SAM 2 masky z boxov
- [ ] poradie snímok + SAM 2 video sledovanie
- [ ] čistenie masiek
- [ ] RLE zápis, náhľady
- [ ] kontraktový test

## Zdroje

- Liu et al.: Grounding DINO, ECCV 2024.
- Ravi et al.: SAM 2: Segment Anything in Images and Videos, ICLR 2025.
- Ren et al.: Grounded SAM, arXiv:2401.14159, 2024.
- Meta Superintelligence Labs: SAM 3: Segment Anything with Concepts, 2025 (autorov overiť pri citovaní).
