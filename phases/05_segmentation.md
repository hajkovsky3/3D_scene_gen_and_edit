# Fáza 05: 2D segmentácia inštancií

`contract_version: 1`

## Účel

Pre každú snímku nájsť objekty zo slovníka (04) a vytvoriť ich presné masky. Ak to model
dokáže, priradiť rovnakému objektu naprieč snímkami rovnaký `track_id` — výrazne to
zjednoduší spájanie inštancií v 3D (07).

## Kontrakt

**Vstup**
- `01_sfm/cameras.json`, `01_sfm/images_undistorted/` — pracuje sa s undistortovanými snímkami,
  aby masky presne sedeli na kamery. Iba registrované snímky.
- `04_tagging/tags.json` (`vocabulary`)

**Výstup** (`05_segmentation/`)
- `detections.json`:
  ```json
  {
    "contract_version": 1,
    "has_tracks": true,
    "frames": ["f000001", "f000004", "f000007"],
    "detections": [
      {"mask_id": "f000001_m000", "frame_id": "f000001",
       "label": "sofa", "label_scores": {"sofa": 0.82, "chair": 0.11},
       "det_score": 0.82, "mask_score": 0.95,
       "bbox_xyxy": [120, 340, 980, 860], "area_px": 312400,
       "track_id": "t0003", "touches_border": false}
    ]
  }
  ```
- `masks/<frame_id>.json` — zoznam `{mask_id, rle}` (COCO RLE, rozmer undistortovanej snímky)
  pre každú snímku z `frames` (aj prázdny).
- `previews/` — niekoľko snímok s prekreslenými maskami (kontrola očami).
- `report.json` — štatistiky behu: `num_frames`, `num_masks`, `masks_per_frame_mean`,
  `frames_without_masks`, `num_tracks`, `track_length_median`, `labels` (počet masiek
  podľa labelu), `segmentation_s`.
- `manifest.json`.

Polia:
- `frames`: snímky, ktoré fáza spracovala (pri `frame_stride` > 1 iba každá k-tá registrovaná).
  Snímka mimo `frames` nebola spracovaná; to neznamená, že na nej objekty nie sú.
- `bbox_xyxy`: pixely, `x1`/`y1` exkluzívne (`x1 - x0` je šírka masky).
- `label`: zo slovníka 04; `label_scores`: skóre fráz slovníka pre detekovaný box (≥
  `text_threshold`, zoradené), `det_score` = najvyššie z nich.
- `mask_score`: pravdepodobnosť prítomnosti objektu podľa SAM 2 (0–1).
- Pri sledovaní nesú masky zo šírenia `label`, `label_scores` a `det_score` poslednej
  detekcie svojho tracku (label sa medzi oknami môže zmeniť, hlasuje sa v 07).
- `mask_id` je unikátne v scéne; poradie masiek v snímke: podľa `det_score`, potom plochy.

Ak model nevie sledovať objekty, `has_tracks: false` a `track_id: null`; spájanie potom
spraví fáza 07.

## Postup

1. **Detekcia**: Grounding DINO so slovníkom (po dávkach, ak je dlhý) → bounding boxy s labelmi.
   Prahy `box_threshold`, `text_threshold`. Skóre frázy = maximum cez jej tokeny (vstavané
   post-processing v transformers skladá názvy z tokenov nad prahom, napr. „blenderr
   pitcher“, preto sa nepoužíva). NMS naprieč labelmi; potlačený box pridá svoje skóre fráz
   do `label_scores` ponechaného.
2. **Segmentácia**: SAM 2 s boxmi ako promptom → masky + skóre kvality.
3. **Sledovanie** (voliteľné, odporúčané): snímky v poradí `frame_id` (pri videu časové
   poradie, pri fotkách poradie súborov; poradie podľa covisibility z 01 zatiaľ nie je)
   a **SAM 2 video prediktor** v oknách po `redetect_every` spracovaných snímkach. Na prvej
   snímke okna sa detekcie spárujú s trackmi podľa IoU boxu (detekcia vs. bbox masky, ktorú
   predchádzajúce okno doťahalo do tejto snímky, `track_iou`); nespárovaná detekcia = nový
   track. Track bez detekcie pokračuje z boxu svojej masky najviac `carry_windows` okien za
   sebou. Detekcia beží najprv na všetkých začiatkoch okien a Grounding DINO sa uvoľní skôr,
   než sa načíta SAM 2 (oba naraz by sa do 4 GB VRAM nezmestili).
4. **Čistenie**: zahodiť príliš malé masky (`min_area_frac`), vyplniť malé diery, odstrániť
   malé izolované komponenty; označiť masky dotýkajúce sa okraja (`touches_border`), lebo objekt
   je pravdepodobne orezaný.
5. **Prekryvy**: masky sa môžu prekrývať (vankúš na gauči). Neriešiť tu, iba uložiť; 07 to rieši
   v 3D. Voliteľne vytvoriť aj neprekryvnú „label mapu“ (menšia maska vyhráva).

## Implementácie

| Názov | Popis | Licencia |
|---|---|---|
| `gdino_sam2` (primárna) | Grounding DINO Swin-T + SAM 2.1 Hiera-S (video prediktor), oba z `transformers`, postup ako demo „continuous id“ v Grounded-SAM-2 | Apache 2.0 (kód aj váhy) |
| `sam3` (alternatíva) | SAM 3: detekcia, segmentácia a sledovanie podľa textového konceptu jedným modelom | SAM License — vlastná, nie OSI, váhy za prihlásením na HF; **hraničné**, pozri `docs/licenses.md` |
| `sam2_auto` (alternatíva) | SAM 2 automatic mask generator bez textu (všetko, čo je objekt), labely dodá až 06/08 cez CLIP | Apache 2.0 |

Pozor: Grounding DINO 1.5/1.6 a DINO-X **nie sú** free to use (iba platené API) — nepoužívať.
Pozor: Grounded-SAM negeneruje CLIP embeddingy; to je samostatná fáza 06.

## Konfigurácia

```yaml
segmentation:
  impl: gdino_sam2
  gdino_model: IDEA-Research/grounding-dino-tiny
  sam2_model: facebook/sam2.1-hiera-small   # large ~900 MB (neschválené), na 4 GB VRAM na hrane
  model_dir: checkpoints/huggingface
  device: auto
  frame_stride: 1           # spracovať každú k-tu registrovanú snímku
  box_threshold: 0.30
  text_threshold: 0.25
  vocab_batch: 20
  nms_iou: 0.7              # NMS boxov naprieč labelmi
  tracking: true
  redetect_every: 10        # dĺžka okna v spracovaných snímkach
  track_iou: 0.5            # IoU boxov pri párovaní detekcií s trackmi
  carry_windows: 2          # koľko okien za sebou track prežije bez detekcie
  min_area_frac: 0.001
  hole_area_frac: 0.0005    # vyplniť diery menšie ako tento podiel snímky
  component_area_frac: 0.0005  # zahodiť časti masky menšie ako tento podiel (najväčšia ostáva)
  max_masks_per_frame: 100
  num_previews: 8
```

Modely sa načítajú z `checkpoints/huggingface` (cache Hugging Face). Bez sledovania
(`tracking: false`) sa detekuje na každej spracovanej snímke a `has_tracks` je `false`.

## Hardvér

GPU. Na GTX 1650 (4 GB, bez tensor cores): Grounding DINO tiny špička 1,9 GB VRAM,
~1,3 s na snímku a dávku slovníka. SAM 2.1 small v sledovaní ~0,35 s na objekt a snímku
(čas rastie lineárne s počtom objektov, pamäťová attention sa počíta pre každý objekt);
fp16 je na tejto karte pomalšie než fp32. Pri ~25 objektoch to je ~9 s na snímku, preto
`frame_stride` a `max_masks_per_frame`. Špička VRAM SAM 2 pri 25 objektoch ~2,9 GB.
Video prediktor drží v okne iba jeho snímky (na CPU), dlhé sekvencie preto nevadia.

## Validácia výstupu

- `frames` sú unikátne registrované snímky; ku každej existuje `masks/<frame_id>.json`
  a jeho `mask_id` sedia s `detections.json`.
- Každá maska má platný `frame_id` z `frames` a dekóduje sa na rozmer snímky z `cameras.json`.
- `mask_id` unikátne, `label` je zo slovníka (alebo `null` pri `sam2_auto`).
- `bbox_xyxy` a `area_px` sedia s dekódovanou maskou.
- Ak `has_tracks`, každý `track_id` sa vyskytuje najviac raz na snímku.

## Riziká a otvorené otázky

- Rovnaký objekt pod rôznymi labelmi v rôznych snímkach (gauč/kreslo) → rieši sa hlasovaním v 07/08.
- Časti vs. celky (vankúš vs. gauč) → hierarchia sa nerieši tu; 07 a 08 potrebujú pravidlo.
- Drift pri sledovaní (maska „preskočí“ na iný objekt) → redetekcia a kontrola konzistencie
  labelu v tracku; prenos tracku bez detekcie je obmedzený (`carry_windows`).
- Bez prenosu trackov (prvá verzia) prežil track iba jedno okno: na *counter* (240 snímok,
  okno 10) 288 trackov s mediánom dĺžky presne 10, lebo detektor na začiatku okna nenájde
  vždy tie isté objekty.
- Detektor priraďuje labely nepresne (krabica od vajec ako *juice*, cestoviny ako *cereal*);
  masky sú presnejšie než labely. Labely opravuje hlasovanie v 07 a CLIP v 06/08.

## Evaluácia

mIoU masiek na pár ručne anotovaných snímkach; konzistencia trackov (počet prerušení).

## Stav

Implementácia: `src/scenepipe/phases/p05_segmentation/` (`common.py`, `gdino_sam2.py`), COCO RLE
v `src/scenepipe/core/masks.py`; testy: `tests/contracts/test_p05.py` (falošná implementácia
so známymi maskami, pomocné funkcie bez modelu), `tests/test_masks.py`. Beží v `sp-gs` vo WSL.

- [x] Grounding DINO detekcia so slovníkom (skóre po frázach, NMS naprieč labelmi)
- [x] SAM 2 masky z boxov
- [x] poradie snímok (`frame_id`) + SAM 2 video sledovanie v oknách, prenos trackov
- [x] čistenie masiek (diery, malé časti, minimálna plocha)
- [x] RLE zápis (vlastná implementácia), náhľady, `report.json`
- [x] kontraktový test
- [x] overenie na Mip-NeRF 360 *counter* (2026-10-27, GTX 1650 vo WSL, slovník 42 tagov
  z 04, 1557 × 1037):

  | Nastavenie | Snímky | Čas | Masky / snímka | Tracky | Medián dĺžky tracku | Najdlhší |
  |---|---|---|---|---|---|---|
  | default, bez prenosu trackov (prvá verzia) | 240 | 54 min | 20,5 | 288 | 10 | — |
  | `frame_stride` 3, `redetect_every` 5, prenos trackov (`fast`) | 80 | 17,7 min | 25,6 | 169 | 9 | 65 / 80 |

  Detekcia: 16 začiatkov okien, 70 s, 372 detekcií. Masky sú presné (linka, misy, ovocie,
  kanvica, vrecká), labely menej: v rámci tracku sa striedajú príbuzné pojmy (*counter /
  kitchen island*, *bowl / mixing bowl*, *blender / pitcher / juicer*) a niektoré sú zlé
  (skrinka ako *kitchen island*, krabica od vajec ako *juice*). Krátkych trackov (1–5 snímok)
  je 64 zo 169.
- [ ] default (všetky snímky) s prenosom trackov; *room* a *kitchen*
- [ ] poradie snímok podľa covisibility pre neusporiadané fotky
- [ ] mIoU na ručne anotovaných snímkach (sekcia Evaluácia)

## Zdroje

- Liu et al.: Grounding DINO, ECCV 2024.
- Ravi et al.: SAM 2: Segment Anything in Images and Videos, ICLR 2025.
- Ren et al.: Grounded SAM, arXiv:2401.14159, 2024.
- Meta Superintelligence Labs: SAM 3: Segment Anything with Concepts, 2025 (autorov overiť pri citovaní).
