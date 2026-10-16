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
- `input/` — ľubovoľné kombinácie: `*.jpg|*.jpeg|*.png` a `*.mp4|*.mov|*.m4v|*.avi`
  (prípony bez ohľadu na veľkosť písmen, aj v podadresároch). Iné súbory (napr. `*.heic`)
  sa nespracujú a zapíšu sa do `rejected` s dôvodom `unsupported_format`. HEIC zatiaľ
  nepodporujeme (pozri Riziká), fotky z iPhonu treba vopred skonvertovať na JPEG.

**Výstup** (`00_ingest/`)
- `images/<frame_id>.jpg` — snímky v poradí, aplikovaná EXIF orientácia (pri videu
  rotácia z metadát), sRGB, JPEG kvalita 95. `frame_id` sú súvislé od `f000001`.
  Poradie: najprv videá (podľa cesty, vnútri časovo), potom fotky (podľa EXIF
  `DateTimeOriginal`, ak ho majú všetky fotky, inak podľa cesty).
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
  - Pri fotke je `source_index` a `timestamp_s` `null`. Pri videu je `source_index` index
    snímky vo videu (od 0) a `timestamp_s` čas v sekundách. `exif_focal_*` je `null`, ak
    údaj chýba (pri videu vždy).
  - `sharpness`: rozptyl Laplaciánu na šedotónovej verzii zmenšenej na dlhšiu stranu 512 px.
  - `reason` v `rejected` je jedno z: `unsupported_format`, `unreadable`, `blur`,
    `duplicate`, `over_max_frames`. Kandidáti z videa, ktorí prehrali výber v okne ostrosti
    (krok 2), sa do `rejected` nezapisujú (bolo by ich príliš veľa).
- `manifest.json`.

## Postup

1. **Načítanie**: fotky cez Pillow (EXIF orientácia, ICC profil → sRGB), video cez
   `cv2.VideoCapture` (ffmpeg pribalený v OpenCV, LGPL; rotácia z metadát sa aplikuje
   automaticky).
2. **Výber snímok z videa**: kandidáti sa vzorkujú s frekvenciou
   `video_fps * sharpness_window` (z každého časového slotu prvá snímka), potom sa z každých
   `sharpness_window` po sebe idúcich kandidátov vyberie najostrejší. Výsledok je približne
   `video_fps` snímok za sekundu videa.
3. **Ostrosť**: rozptyl Laplaciánu na zmenšenom šedotónovom obrázku. Prah je relatívny:
   snímka pod `sharpness_percentile_reject`-tým percentilom svojho videa je `blur`.
   Počíta sa zvlášť pre každé video (rôzne kamery majú rôznu škálu). Fotky sa podľa
   ostrosti nefiltrujú, ak nie je `blur_filter_photos: true` (fotky sú zámerné zábery
   a percentilový prah by vždy zahodil ich pevný podiel).
   **Medzery**: pri rýchlom otočení je rozmazaných veľa snímok za sebou a po ich vyradení
   by vo videu ostala diera, ktorú SfM nepremostí. Preto sa rozmazaná snímka ponechá, ak by
   ďalší kandidát toho istého videa bol od poslednej ponechanej snímky ďalej než
   `blur_max_gap_s` (greedy, default 1 s; `0` vypne). Rozmazané snímky pred prvou a po
   poslednej ostrej snímke videa sa neponechávajú, ani snímky s ostrosťou pod
   `blur_gap_min_ratio` × medián ostrosti videa (default 0,3): takto rozmazané snímky SfM
   umiestni zle a jediná vzdialená kamera pokazí normalizáciu celej scény (*izba*, pozri Stav).
4. **Deduplikácia**: šedotónová miniatúra 64×64, normalizovaná korelácia (NCC)
   s poslednou ponechanou snímkou. Ak je `> dedup_threshold`, snímka je `duplicate`.
   Pomáha to SfM aj rýchlosti tréningu.
5. **Limit počtu**: ak ostane viac ako `max_frames` snímok, rovnomerne sa vyberie
   `max_frames` z nich, ostatné sú `over_max_frames`.
6. **Zmena rozlíšenia**: dlhšia strana max `max_side` px (default 1600), nikdy sa nezväčšuje.
   Pôvodné rozlíšenie sa v reťazci nepoužíva.
7. **EXIF**: zachovať ohniskovú vzdialenosť (pomáha SfM ako počiatočný odhad intrinsiky).
8. Zápis `frames.json` vrátane zoznamu vyradených snímok s dôvodom. Snímky sa počas behu
   ukladajú do `00_ingest/_staging/` (nie všetky naraz v pamäti), na konci sa presunú do
   `images/` a `_staging/` sa zmaže.

## Implementácie

| Názov | Popis | Licencia |
|---|---|---|
| `basic` (primárna) | OpenCV + Pillow, postup vyššie | Apache 2.0 / MIT-CMU (HPND) / LGPL (ffmpeg pribalený v OpenCV) |

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
  min_frames: 20             # menej ponechaných snímok = chyba
  blur_filter_photos: false
  blur_max_gap_s: 1.0        # 0 = vypnuté
  blur_gap_min_ratio: 0.3    # dieru nevypĺňať snímkou pod 0,3 × medián ostrosti
```

## Hardvér

Iba CPU. Sekundy až minúty.

## Validácia výstupu

- `frames.json` existuje, `frame_id` unikátne a súvislé, všetky `image` súbory existujú.
- Rozmery v JSON sedia so skutočnými rozmermi súborov, dlhšia strana ≤ `max_side`.
- V `images/` nie sú iné súbory než tie z `frames.json`.
- Počet snímok je medzi `min_frames` (default 20) a `max_frames`, inak chyba s vysvetlením.
- Každý `reason` v `rejected` je z povoleného zoznamu.

## Riziká a otvorené otázky

- Video z telefónu má často premenlivý framerate a rolling shutter. Pri rýchlom pohybe
  pomôže nižšie `video_fps` a prísnejší prah ostrosti.
- Rôzne fotoaparáty v jednej sade: zachovať informáciu o zdroji kvôli SfM (rôzne kamery).
- Lesklé a priehľadné povrchy, pohybujúce sa objekty: tu sa neriešia, len zaznamenať.
- **HEIC** (fotky z iPhonu) primárna implementácia nepodporuje: `pillow-heif` vo svojich wheel
  balíčkoch pribaľuje x265 (GPL-2), preto môže byť len voliteľný doplnok (stav `NEKOMERČNÉ`
  v `docs/licenses.md`), nie súčasť primárnej cesty. To isté platí pre PyAV (pribaľuje
  x264/x265). Riešenie v primárnej ceste: konverzia na JPEG mimo reťazca.
- `cv2.VideoCapture` pri videu s premenlivým framerate môže vracať nepresné časy
  (`CAP_PROP_POS_MSEC`); na výber snímok to stačí.

## Evaluácia

Počet zachovaných/vyradených snímok, vplyv prahov na úspešnosť registrácie v 01.

## Stav

Implementácia: `src/scenepipe/phases/p00_ingest/basic.py`, test: `tests/contracts/test_p00.py`.
Inštalácia: `pip install -e ".[dev,ingest]"`.

- [x] načítanie fotiek (EXIF orientácia, ICC → sRGB; HEIC vynechané, pozri Riziká)
- [x] extrakcia snímok z videa
- [x] filter ostrosti
- [x] deduplikácia
- [x] zápis `frames.json` + manifest
- [x] kontraktový test (syntetické fotky a video)
- [x] overenie na reálnej scéne *izba* (2026-10-16): video z telefónu na výšku, 100 s, 1080×1920
  s rotáciou v metadátach → 208 snímok 900×1600, 90 vyradených pre rozmazanie, 1 duplikát; 72 s
  na CPU. Pri rýchlych otočeniach vznikajú medzery až 6 s bez snímok.
- [x] dopĺňanie medzier `blur_max_gap_s` (*izba*: 242 snímok, 56 vyradených, najdlhšia medzera
  0,94 s namiesto 6,3 s). **Pozor:** prvá verzia bez `blur_gap_min_ratio` pustila aj veľmi
  rozmazané snímky (ostrosť 37 pri mediáne ~350). 01 ich zaregistroval (224/242, 0,66 px),
  ale 5 kamier bolo 50–100× ďalej než ostatné, normalizácia scénu zmrštila a 02 divergoval
  (PSNR 9). Počet registrovaných kamier a reprojekčná chyba to neodhalia. Oprava: prah
  `blur_gap_min_ratio` tu a vyradenie odľahlých kamier v 01 (`max_camera_dist_ratio`).
  Po oprave (2026-10-16, `fast.yaml`): 225 snímok, 01 vyradil 10 odľahlých kamier → 211/225,
  02 PSNR 22.27 / SSIM 0.860. Bez dopĺňania medzier (194 kamier) bolo PSNR 23.96, ale na inej
  testovacej sade, takže prínos dopĺňania pre kvalitu zatiaľ nie je preukázaný.
- [ ] doladenie prahov podľa úspešnosti registrácie v 01
