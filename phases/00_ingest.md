# Fáza 00: Ingest (príprava vstupných snímok)

`contract_version: 1`

## Účel

Zjednotiť vstup. Používateľ dodá buď sadu fotiek, alebo video (prípadne oboje). Fáza z toho
spraví čistú sadu snímok s jednotnými názvami, orientáciou a rozlíšením a vyradí nepoužiteľné
snímky (rozmazané, takmer duplicitné). Všetky ďalšie fázy potom pracujú len s `frame_id`.

Oddelenie tejto fázy od SfM je zámerné: výber snímok z videa a filtrovanie kvality sú
samostatný problém a dajú sa vylepšovať nezávisle.

## Kontrakt

**Vstup**
- `input/` — ľubovoľné kombinácie: `*.jpg|*.jpeg|*.png|*.heic` a `*.mp4|*.mov`.

**Výstup** (`00_ingest/`)
- `images/<frame_id>.jpg` — snímky v poradí (video: časové poradie; fotky: podľa EXIF času,
  inak podľa názvu), aplikovaná EXIF orientácia, sRGB, JPEG kvalita 95.
- `frames.json`:
  ```json
  {
    "contract_version": 1,
    "frames": [
      {"frame_id": "f000001", "image": "00_ingest/images/f000001.jpg",
       "width": 1920, "height": 1080,
       "source": "input/room.mp4", "source_index": 120, "timestamp_s": 4.0,
       "sharpness": 183.2, "exif_focal_mm": 4.2, "exif_focal_35mm": 26}
    ],
    "rejected": [ {"source": "input/room.mp4", "source_index": 125, "reason": "blur"} ]
  }
  ```
- `manifest.json`.

## Postup

1. **Načítanie**: fotky cez Pillow (+ `pillow-heif` pre HEIC), video cez ffmpeg / PyAV.
2. **Výber snímok z videa**: najprv vzorkovanie s pevným krokom (napr. 2–4 fps), potom
   výber podľa ostrosti v okne (z každých k kandidátov najostrejší).
3. **Ostrosť**: rozptyl Laplaciánu na zmenšenom šedotónovom obrázku. Prah relatívny
   (napr. pod 30. percentilom = rozmazané), nie absolútny.
4. **Deduplikácia**: snímky príliš podobné predchádzajúcej (malý optický tok alebo vysoká
   podobnosť zmenšených obrázkov) sa vyradia. Pomáha to SfM aj rýchlosti tréningu.
5. **Zmena rozlíšenia**: dlhšia strana max `max_side` px (default 1600). Pôvodné rozlíšenie
   sa v reťazci nepoužíva.
6. **EXIF**: zachovať ohniskovú vzdialenosť (pomáha SfM ako počiatočný odhad intrinsiky).
7. Zápis `frames.json` vrátane zoznamu vyradených snímok s dôvodom.

## Implementácie

| Názov | Popis | Licencia |
|---|---|---|
| `basic` (primárna) | OpenCV + Pillow + ffmpeg, postup vyššie | Apache 2.0 / HPND / LGPL (ffmpeg ako externý program) |

Alternatívy do budúcna: výber snímok podľa pokrytia scény (keyframe selection podľa optického
toku), automatické maskovanie pohybujúcich sa objektov a ľudí.

## Konfigurácia

```yaml
ingest:
  impl: basic
  video_fps: 3
  sharpness_window: 5
  sharpness_percentile_reject: 30
  dedup_threshold: 0.97
  max_side: 1600
  max_frames: 300
```

## Hardvér

Iba CPU. Sekundy až minúty.

## Validácia výstupu

- `frames.json` existuje, `frame_id` unikátne a súvislé, všetky `image` súbory existujú.
- Rozmery v JSON sedia so skutočnými rozmermi súborov.
- Aspoň `min_frames` (default 20) snímok, inak chyba s vysvetlením.

## Riziká a otvorené otázky

- Video z telefónu má často premenlivý framerate a rolling shutter. Pri rýchlom pohybe
  pomôže nižšie `video_fps` a prísnejší prah ostrosti.
- Rôzne fotoaparáty v jednej sade: zachovať informáciu o zdroji kvôli SfM (rôzne kamery).
- Lesklé a priehľadné povrchy, pohybujúce sa objekty: tu sa neriešia, len zaznamenať.

## Evaluácia

Počet zachovaných/vyradených snímok, vplyv prahov na úspešnosť registrácie v 01.

## Stav

- [ ] načítanie fotiek (vrátane EXIF orientácie, HEIC)
- [ ] extrakcia snímok z videa
- [ ] filter ostrosti
- [ ] deduplikácia
- [ ] zápis `frames.json` + manifest
- [ ] kontraktový test
