# Fáza 04: Tagovanie (slovník objektov)

`contract_version: 1`

## Účel

Open-vocabulary detektory (Grounding DINO, SAM 3) potrebujú textový zoznam toho, čo majú hľadať.
Aby graf scény vznikol **automaticky**, bez ručného vymenúvania objektov, táto fáza z obrázkov
zostaví slovník objektov v scéne. Oddelenie od segmentácie umožňuje slovník aj ručne doplniť
alebo nahradiť (napr. pevný zoznam tried pre experiment).

## Kontrakt

**Vstup**
- `00_ingest/frames.json`, `00_ingest/images/` (stačí podmnožina snímok)
- voliteľne `input/vocabulary.txt` — ručne zadané pojmy (jeden na riadok), vždy sa pridajú

**Výstup** (`04_tagging/`)
- `tags.json`:
  ```json
  {
    "contract_version": 1,
    "per_frame": { "f000001": [{"tag": "sofa", "score": 0.91}, …] },
    "vocabulary": [
      {"tag": "sofa", "frames": 42, "max_score": 0.95, "source": "model"},
      {"tag": "lamp", "frames": 1, "max_score": 0.80, "source": "user"}
    ],
    "excluded": ["room", "living room", "floor", "wall", "indoor"]
  }
  ```
- `manifest.json`.

`vocabulary` je výsledný zoznam pre fázu 05: zjednotený, deduplikovaný, bez abstraktných pojmov.

## Postup

1. Vybrať podmnožinu snímok (každú k-tu, default ~30 snímok rovnomerne cez sekvenciu).
2. Pre každú snímku tagger → zoznam tagov so skóre.
3. **Filtrovanie**: odstrániť pojmy, ktoré nie sú objekty (scény, materiály, farby, akcie)
   podľa blacklistu v konfigurácii; voliteľne ponechať „stuff“ triedy (podlaha, stena, strop)
   v osobitnom zozname, lebo sú užitočné ako podložky pre reláciu *on*.
4. **Agregácia**: tag sa dostane do slovníka, ak sa objaví aspoň v `min_frames` snímkach
   alebo má veľmi vysoké skóre.
5. **Zjednotenie synoným**: lemmatizácia a mapa synoným (`couch` → `sofa`), voliteľne podľa
   podobnosti textových CLIP embeddingov.
6. Pridanie ručne zadaných pojmov.

## Implementácie

| Názov | Popis | Licencia |
|---|---|---|
| `ram_plus` (primárna) | Recognize Anything Plus (RAM++), ~4500 tagov, rýchly | Apache 2.0 (over aj váhy) |
| `vlm` (alternatíva) | VLM (napr. Qwen2.5-VL-7B / Florence-2) s promptom „vymenuj objekty“ — otvorenejší slovník, pomalší | Apache 2.0 / MIT |
| `fixed` | iba `input/vocabulary.txt`, bez modelu — na kontrolované experimenty | — |

## Konfigurácia

```yaml
tagging:
  impl: ram_plus
  sample_frames: 30
  min_frames: 2
  score_threshold: 0.6
  keep_stuff_classes: true
  blacklist_file: configs/tag_blacklist.txt
  synonyms_file: configs/tag_synonyms.yaml
```

## Hardvér

GPU odporúčaná (RAM++ ~ 2–4 GB VRAM), beží aj na CPU pomalšie. VLM variant 16+ GB, alebo kvantizovaný.

## Validácia výstupu

- `vocabulary` neprázdny, unikátne tagy, všetky `per_frame` kľúče sú platné `frame_id`.
- Žiadny tag z `excluded` sa nedostal do `vocabulary`.

## Riziká a otvorené otázky

- Príliš všeobecné tagy („furniture“, „object“) spôsobia duplicitné detekcie v 05 → blacklist.
- Malé objekty sa v tagoch často neobjavia → ručný slovník alebo VLM variant.
- Dlhý slovník spomaľuje Grounding DINO (dĺžka textového promptu je obmedzená) → v 05 sa
  slovník delí na dávky.

## Evaluácia

Recall slovníka voči ručne anotovanému zoznamu objektov v testovacej scéne.

## Stav

- [ ] výber snímok
- [ ] tagger RAM++
- [ ] filtrovanie, agregácia, synonymá
- [ ] ručný slovník
- [ ] kontraktový test

## Zdroje

- Zhang et al.: Recognize Anything: A Strong Image Tagging Model, 2023; Huang et al.: RAM++, 2023.
- Gu et al.: ConceptGraphs, ICRA 2024 (používa rovnaký princíp tag → detekcia).
