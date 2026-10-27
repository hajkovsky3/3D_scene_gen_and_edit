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
      {"tag": "sofa", "frames": 42, "max_score": 0.95, "source": "model", "kind": "object"},
      {"tag": "floor", "frames": 28, "max_score": 0.93, "source": "model", "kind": "stuff"},
      {"tag": "lamp", "frames": 0, "max_score": null, "source": "user", "kind": "object"}
    ],
    "excluded": ["room", "living room", "indoor"]
  }
  ```
- `manifest.json`.

`vocabulary` je výsledný zoznam pre fázu 05: zjednotený, deduplikovaný, bez abstraktných pojmov.

- Tagy sú malými písmenami, po zjednotení synoným (`couch` → `sofa`).
- `per_frame`: iba vybrané snímky (krok 1), tagy so skóre nad prahom (aj tie, ktoré sa potom
  vyradia), zoradené podľa skóre. Pri implementácii `fixed` je prázdny.
- `frames`: v koľkých snímkach z `per_frame` sa tag objavil; `max_score` je najvyššie skóre,
  `null` pri ručnom pojme, ktorý model nevidel (vtedy `frames` = 0).
- `source`: `model`, alebo `user` pre pojem z `input/vocabulary.txt` (aj keď ho videl aj model).
- `kind`: `stuff` pre plochy bez jasnej hranice objektu (`configs/tag_stuff.txt`: podlaha,
  stena, strop …), použiteľné ako podložka pre reláciu *on*; inak `object`.
- `vocabulary` je zoradený podľa `frames` zostupne, potom podľa `max_score`.
- `excluded`: tagy nad prahom, ktoré vyradil blacklist (alebo stuff pri
  `keep_stuff_classes: false`), abecedne.

## Postup

1. Vybrať podmnožinu snímok (každú k-tu, default ~30 snímok rovnomerne cez sekvenciu).
2. Pre každú snímku tagger → zoznam tagov so skóre.
3. **Zjednotenie synoným**: malé písmená a mapa synoným `configs/tag_synonyms.yaml`
   (`couch` → `sofa`, `dinning table` → `dining table`); pri zhode v jednej snímke platí
   vyššie skóre. Lemmatizácia ani podobnosť CLIP embeddingov zatiaľ nie sú (zoznam RAM++ je
   uzavretý a v jednotnom čísle, mapa stačí).
4. **Filtrovanie**: odstrániť pojmy, ktoré nie sú objekty (scény, materiály, farby, akcie,
   príliš všeobecné pojmy) podľa `configs/tag_blacklist.txt`; „stuff“ triedy
   (`configs/tag_stuff.txt`) ostanú v slovníku s `kind: stuff`, ak `keep_stuff_classes`.
5. **Agregácia**: tag sa dostane do slovníka, ak sa objaví aspoň v `min_frames` snímkach
   alebo má skóre aspoň `high_score`.
6. Pridanie ručne zadaných pojmov (vždy, aj keď sú na blackliste).

## Implementácie

| Názov | Popis | Licencia |
|---|---|---|
| `ram_plus` (primárna) | Recognize Anything Plus (RAM++, Swin-L, 384 × 384), 4585 tagov, rýchly | Apache 2.0 (kód aj váhy) |
| `vlm` (alternatíva) | VLM (napr. Qwen2.5-VL-7B / Florence-2) s promptom „vymenuj objekty“ — otvorenejší slovník, pomalší | Apache 2.0 / MIT |
| `fixed` | iba `input/vocabulary.txt`, bez modelu — na kontrolované experimenty | — |

`ram_plus` volá balík `ram` ako externú závislosť (inštalácia: `envs/README.md`). Funkcia
`generate_tag` vracia iba názvy tagov, preto `ram_plus.py` zopakuje jej dopredný prechod
nad tými istými podmodulmi a vráti skóre všetkých tagov. Balík je písaný pre
transformers 4.25; s transformers 5 ho `ram_plus.py` spustí tak, že za behu doplní
odstránené pomocné funkcie a vypne inicializáciu váh a tokenizer (všetky váhy sú
z checkpointu, z tokenizera sa používa iba veľkosť slovníka). Kód balíka sa nemení.

## Konfigurácia

```yaml
tagging:
  impl: ram_plus            # ram_plus | fixed
  sample_frames: 30         # 0 = všetky snímky
  min_frames: 2
  score_threshold: 0.6
  high_score: 0.85          # tag z jedinej snímky prejde, ak má aspoň toto skóre
  keep_stuff_classes: true
  blacklist_file: configs/tag_blacklist.txt
  stuff_file: configs/tag_stuff.txt
  synonyms_file: configs/tag_synonyms.yaml
  # ram_plus
  checkpoint: checkpoints/ram_plus/ram_plus_swin_large_14m.pth
  tokenizer_dir: checkpoints/bert-base-uncased
  device: auto              # auto | cuda | cpu
  fp16: true
  batch_size: 4
  class_thresholds: true    # kalibrované prahy RAM++ pre jednotlivé tagy (navyše k score_threshold)
```

Cesty v konfigurácii sú relatívne ku koreňu repozitára. Váhy ležia mimo workspace, preto nie
sú medzi vstupmi v manifeste (ich cesta je v `config`). Profil `fast`: `sample_frames: 15`.

## Hardvér

GPU odporúčaná: RAM++ vo fp16 má špičku ~1 GB VRAM (GTX 1650, dávka 4), RAM ~5 GB pri
načítaní checkpointu. Na CPU beží pomalšie (`device: cpu`, `fp16` sa vtedy ignoruje).
VLM variant 16+ GB, alebo kvantizovaný.

## Validácia výstupu

- `vocabulary` neprázdny, unikátne tagy, všetky `per_frame` kľúče sú platné `frame_id`.
- Žiadny tag z `excluded` sa nedostal do `vocabulary`.
- Skóre v [0, 1], `source` a `kind` z povolených hodnôt.

## Riziká a otvorené otázky

- Príliš všeobecné tagy („furniture“, „object“) spôsobia duplicitné detekcie v 05 → blacklist.
- Malé objekty sa v tagoch často neobjavia → ručný slovník alebo VLM variant.
- Dlhý slovník spomaľuje Grounding DINO (dĺžka textového promptu je obmedzená) → v 05 sa
  slovník delí na dávky.
- Zoznam RAM++ obsahuje aj slovesá, farby a typy scén (*sit, black, image, living room*).
  Pri globálnom prahu 0,5 ich bolo v slovníku väčšina (254–323 tagov na scénu), lebo sa
  objavujú takmer v každej snímke a `min_frames` ich nezastaví. Kalibrované prahy RAM++
  (`class_thresholds`) ich väčšinu odstránia, zvyšok blacklist. Blacklist je zostavený podľa
  troch interiérových scén; pri inom type scény (exteriér) ho treba skontrolovať.
- Konzistentné halucinácie: tag, ktorý model vidí v celej scéne, prejde, aj keď objekt
  neexistuje (overiť v 05: Grounding DINO preň nenájde box).
- Balík `ram` nie je udržiavaný pre nové transformers; pri ďalšej zmene ich API môže
  obchádzanie v `ram_plus.py` prestať stačiť (vtedy vlastné prostredie so starými verziami).

## Evaluácia

Recall slovníka voči ručne anotovanému zoznamu objektov v testovacej scéne.

## Stav

Implementácia: `src/scenepipe/phases/p04_tagging/` (`common.py`, `ram_plus.py`, `fixed.py`),
test: `tests/contracts/test_p04.py` (falošný tagger + `fixed`; RAM++ iba ak je nainštalovaný
balík `ram` a checkpoint). Beží v `sp-gs` vo WSL (`envs/README.md`).

- [x] výber snímok (rovnomerne cez sekvenciu)
- [x] tagger RAM++ (fp16 na GPU, skóre všetkých tagov, kalibrované prahy)
- [x] filtrovanie, agregácia, synonymá (`configs/tag_blacklist.txt`, `tag_stuff.txt`,
  `tag_synonyms.yaml`)
- [x] ručný slovník (`input/vocabulary.txt`) a implementácia `fixed`
- [x] kontraktový test
- [x] overenie na Mip-NeRF 360 *room*, *counter*, *kitchen* (2026-10-27, GTX 1650 vo WSL,
  30 z 240–311 snímok, `images_2`): ~30 s načítanie modelu z `/mnt/c`, ~30 s tagovanie,
  špička VRAM 992 MB. Vplyv prahov (počet tagov v slovníku):

  | Scéna | prah 0,5 | + kalibrované prahy, prah 0,6 | + blacklist a synonymá (default) |
  |---|---|---|---|
  | room | 288 | 42 | 31 |
  | counter | 323 | 56 | 42 |
  | kitchen | 254 | 17 | 11 |

  Výsledné slovníky (*stuff* označené hviezdičkou):
  - *room*: floor\*, carpet, sofa, armchair, chair, speaker, table, plant, bookcase, sandal,
    pillow, piano, curtain, entertainment center, doorway, stool, shoe, footstool, laptop,
    teddy bear, slipper, computer, subwoofer, television, shelf, picture frame, pot,
    dining table, mat, flower, window
  - *counter*: counter, blender, tray, pot, apple, kitchen island, juicer, bowl, pitcher,
    orange, pan, onion, rolling pin, juice, kitchen sink, mixing bowl, table, spatula, can,
    stove, plant, plate, sink, bread, food processor, butter, condiment, drawer, dishwasher,
    mixer, baking sheet, jug, pastry, chip, cereal, bag, dough, peach, package, platter,
    garlic, boiler
  - *kitchen*: table, lego, bulldozer, mat, carpet, floor\*, excavator, cloth, dining table,
    toy car, fire truck

  Stena ani strop sa v slovníkoch neobjavili (RAM++ ich nedal nad prah); ak ich 08 bude
  potrebovať ako podložky, doplniť cez `input/vocabulary.txt`.
- [ ] recall slovníka voči ručne anotovanému zoznamu objektov (sekcia Evaluácia)
- [ ] overenie na scéne z telefónu (*izba*) a na exteriéri (blacklist)

## Zdroje

- Zhang et al.: Recognize Anything: A Strong Image Tagging Model, 2023; Huang et al.: RAM++, 2023.
- Gu et al.: ConceptGraphs, ICRA 2024 (používa rovnaký princíp tag → detekcia).
