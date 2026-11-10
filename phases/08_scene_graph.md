# Fáza 08: Graf scény (scene graph)

`contract_version: 2`

## Účel

Postaviť symbolickú reprezentáciu scény: **uzly** = objekty (s geometriou, labelom, popisom
a príznakom), **hrany** = vzťahy medzi nimi (*on*, *near*, *inside*, …). Graf je to, na čo
ukazuje prompt („pohár na stole“), a to, čo úprava (12) používa na umiestnenie objektov.

Po tejto fáze runner vytvorí počiatočný stav scény `state/v000`.

## Kontrakt

**Vstup**
- `03_refinement/gaussians.ply`
- `07_lifting/gaussian_instances.npz`, `instances.json`, `instance_features.npy`
- `01_sfm/cameras.json`, `01_sfm/images_undistorted/`
- `05_segmentation/detections.json` (výber najlepších pohľadov na objekt; masky z `masks/`
  číta iba implementácia s VLM na výrezy)
- `04_tagging/tags.json` (`kind: stuff` určuje štruktúrne uzly)

`images_undistorted/` a `masks/` potrebuje iba implementácia s VLM, `geometric_only` ich nečíta.

**Výstup** (`08_scene_graph/`)
- `scene_graph.json`:
  ```json
  {
    "contract_version": 2,
    "up_axis": "+z",
    "embedding_model": "hf:laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "scene_scale": 1.37,
    "nodes": [
      {"node_id": "n0000", "instance_id": 0, "label": "sofa", "label_original": null,
       "label_votes": {"sofa": 0.78, "chair": 0.22},
       "caption": "brown fabric three-seat sofa with two cushions",
       "attributes": {"color": "brown", "material": "fabric"},
       "obb": {"center": [0.1, 0.4, 0.35], "extent": [2.1, 0.9, 0.8],
               "rotation": [[…],[…],[…]]},
       "aabb": {"min": […], "max": […]},
       "num_gaussians": 41233, "feature_row": 0,
       "best_views": ["f000014", "f000052", "f000090"],
       "is_structure": false}
    ],
    "edges": [
      {"source": "n0004", "target": "n0000", "relation": "on",
       "confidence": 0.92, "origin": "geometric"},
      {"source": "n0007", "target": "n0000", "relation": "next_to",
       "confidence": 0.71, "origin": "semantic"}
    ]
  }
  ```
- `node_features.npy` — `float16[K, D]` (kópia/odvodenie z 07, riadok podľa `feature_row`).
- `graph.html` — vizualizácia grafu (kontrola očami): pohľad zhora na pôdorysy OBB s hranami
  a tabuľky uzlov a hrán, samostatná stránka bez závislostí.
- `report.json` — počty uzlov, štruktúrnych a neoznačených uzlov, hrán podľa vzťahu, uzly bez
  opory (`unsupported_nodes`: nie sú štruktúra a nemajú `on` ani `inside`), použité prahy
  `on_eps`, `near_delta` a časy krokov. Pri `geometric_vlm` navyše `vlm`, počet popísaných uzlov,
  nespracovateľné odpovede (`vlm_failed`), opravené labely (`relabeled`: `node_id`, `from`, `to`)
  a čas na uzol.
- `manifest.json`.

Polia:
- `node_id` = `n` + `instance_id` na 4 cifry; uzly v poradí inštancií 07, jeden na inštanciu.
- `label`: label z 07 (`null`, ak ho 07 nemá). Ak ho VLM opraví, pôvodný je v `label_original`,
  inak `null`. `label_votes`: kópia z 07 (pre grounding v 10).
- `caption`, `attributes`: z VLM; v `geometric_only` je `caption` = label a `attributes` = `{}`.
  `attributes` má kľúče `color` a `material` (malé písmená), ak ich VLM uviedol; uzly bez
  popisu (štruktúra, nespracovateľná odpoveď) majú `caption` = label a `attributes` = `{}`.
  Ak VLM pomenuje uzol bez labelu, `label` je nový a `label_original` ostáva `null`
  (zmena je v `report.json`, `relabeled`).
- `obb`: bod sveta = `center + rotation @ lokálny`, lokálny v `[-extent/2, extent/2]`; stĺpce
  `rotation` sú osi kvádra vo svete, `extent` sú plné dĺžky strán. Pri `gravity_aligned` je
  tretí stĺpec +Z a prvý hlavný vodorovný smer. `aabb` je z rovnakých orezaných bodov.
- `scene_scale`: mierka scény (`core.cameras.scene_scale`, ~polomer kamier); prahy vzťahov
  sú jej násobky.
- `best_views`: `frame_id` zoradené od najlepšieho.
- Povolené vzťahy: geometrické `on`, `inside`, `near` (`origin: geometric`) a sémantické
  `part_of`, `supports`, `belongs_to_set`, `next_to` (`origin: semantic`). `near` a `next_to`
  sú symetrické a uložené raz (`source` < `target`). `confidence` je 0–1.
- Kontrolu formátu robí `src/scenepipe/core/scene_graph.py` (použije ju aj 12, keď graf mení).

`is_structure: true` pre podlahu, steny, strop (ak boli v slovníku) — slúžia ako podložky,
ale nie sú bežnými cieľmi úprav. Štruktúra = label s `kind: stuff` v `04_tagging/tags.json`
alebo v `structure_labels` (predvolene `floor`, `wall`, `ceiling`).

**Stav `v000`**: po úspešnom behu 08 ho zapíše CLI (`core/state.py`, `write_initial_state`),
nie fáza samotná. Ukazuje na výstupy 03, 07 a 08 a `state/current` (textový súbor) nastaví na
`v000`. Opätovný build `v000` prepíše; ak už existujú novšie stavy, `current` sa nemení
a do logu ide varovanie (tie stavy vznikli zo starého buildu).

## Postup

1. **Geometria uzlov** z gaussiánov inštancie (iba gaussiány s dostatočnou opacity):
   - **OBB** s osou Z zarovnanou s gravitáciou (nábytok stojí vzpriamene): PCA v rovine XY
     pre natočenie, rozsah po odstránení 1–2 % extrémov. Plné 3D OBB len ako voliteľné.
   - AABB pre rýchle testy.
2. **Najlepšie pohľady**: snímky, kde je objekt najviac viditeľný (veľká plocha masky, nedotýka
   sa okraja, vysoká kvalita) → `best_views`.
3. **Caption a atribúty**: výrez najlepšieho pohľadu (alebo 2–3 pohľadov) → VLM s promptom
   na krátky popis + JSON atribúty (farba, materiál). Výsledný label: konsenzus label z 07,
   VLM ho môže opraviť, ak je veľmi istý (zaznamenať pôvodný aj nový).
4. **Geometrické vzťahy** (analyticky, deterministicky):
   - `on(A, B)`: spodná podstava A blízko hornej plochy B (|z_min(A) − z_max(B)| < ε), prekryv
     priemetov do XY nad prahom, A je menší.
   - `inside(A, B)`: OBB A z väčšej časti vo vnútri OBB B.
   - `near(A, B)`: vzdialenosť povrchov (aproximácia cez OBB) < δ relatívne k veľkosti scény.
   - `above/below`, `left_of/right_of` sa **neukladajú** — závisia od pohľadu; rieši ich 10
     pri groundingu vzhľadom na kameru.
   - prahy ε, δ relatívne k mierke scény (normalizovaná v 01).
5. **Sémantické vzťahy** (voliteľné, LLM): pre páry blízkych objektov poslať LLM labely,
   captions a geometrické fakty → vzťahy ako `part_of`, `supports`, `belongs_to_set`.
   Iba ako doplnok, geometrické vzťahy sú zdroj pravdy pre umiestňovanie.
6. Uložiť JSON (+ NetworkX sa dá postaviť pri načítaní; formát kontroluje `core/scene_graph.py`).

### Presne (implementované)

Popisy (krok 3, `geometric_vlm`): každý uzol okrem štruktúry dostane `views_per_caption`
výrezov zo svojich najlepších pohľadov (najväčšia maska inštancie v danej snímke):
štvorec okolo bboxu so stranou `crop_context` × dlhšia strana (okolie pomáha rozpoznať
objekt, mimo snímky sivá), zmenšený na `crop_size`, maska obtiahnutá červenou. Prompt
povie label zo 07 a požiada o jeden riadok JSON (`name`, `label_correct`, `caption`, `color`,
`material`); upozorní, že vyznačená oblasť môže byť časť väčšieho objektu alebo objekt
ležiaci na inom (inak model pomenuje gauč namiesto vankúša na ňom). Label sa zmení, iba ak
model povie `label_correct: false`, jeho `name` je iné a pri `label_correction: vocabulary`
je v slovníku 04 alebo medzi hlasmi 07 (aj v množnom čísle), aby graf ostal v slovníku, ktorý
10 vie groundovať. Navrhnutú opravu ešte potvrdí `relabel_check`: pre každý pohľad zvlášť
otázka „Which word better names the object outlined in red: „starý“ or „nový“?“; label sa zmení,
iba ak všetky pohľady zvolia nový (vo voľnom JSON model často pomenuje väčší objekt alebo
niečo vedľa, pri priamej voľbe sa vráti k správnemu; na *room* takto odpadlo 7 zo 14 návrhov,
medzi nimi všetky zjavne zlé: *teddy bear → pillow*, *armchair → teddy bear*,
*carpet → slipper*). Odpoveď sa číta ako prvý `{…}` v texte; ak sa nedá prečítať, uzol ostane
bez popisu.

Geometria (krok 1): body = stredy gaussiánov inštancie s opacity ≥ `min_opacity` (ak ich
ostane menej ako 20 alebo 10 %, všetky). OBB aj AABB po orezaní `obb_trim_percent` na oboch
koncoch každej osi, strana najmenej `min_extent_rel` × mierka. Spodok objektu pre `on` je
`bottom_percentile` (0,5) výšok bodov, nie AABB: tenké nohy stola majú málo bodov a orezanie
1,5 % by spodok zdvihlo nad podlahu. Na vzťahy sa berie náhodná podmnožina `max_points` bodov.

Najlepšie pohľady (krok 2): masky inštancie zo 07 (`source_masks`), skóre = podiel plochy
snímky × `mask_score` × `border_penalty` pri dotyku s okrajom; najlepšia maska v snímke, snímky
aspoň `best_view_min_gap` spracovaných snímok od seba (ak ich je málo, odstup sa uvoľní).

Vzťahy (krok 4), ε = `on_eps_rel` × mierka, δ = `near_delta_rel` × mierka:
- **opora** A na B (podmienka `on`): A nie je štruktúra; A má menší pôdorys **alebo** jeho
  spodok je o viac ako ε vyššie než spodok B (rastlina je širšia než kvetináč); pôdorysy sa
  prekrývajú aspoň `on_overlap` menšieho z nich; a |spodok A − výška opory B| < ε. Výška opory
  = 95. percentil výšok bodov B pod pôdorysom A (zväčšeným `support_scale` ×, lebo povrch
  priamo pod A je zakrytý) a pod stredom výšky A (operadlo gauča nie je sedák, na ktorom leží
  vankúš). **Vodorovná štruktúra** (podlaha) je rovina: rozhoduje iba výška, lebo inštancia
  podlahy často pokrýva len časť izby (na *room* gauč inak nemal oporu).
- **inside**(A, B): aspoň `inside_share` bodov A v OBB B, A má menší objem, B nie je štruktúra
  a B A **uzatvára**: má body viac ako ε nad vrchom A nad aspoň 1/4 pôdorysu A (skrinka,
  polica), alebo body v hornej polovici výšky A (a viac ako ε nad spodkom A) na všetkých
  štyroch stranách A (miska, drez, zásuvka). Samotné ležanie v OBB nestačí: vankúš na gauči
  leží v jeho kvádri a riedke inštancie v pozadí majú obrovské kvádre (na *counter* bolo
  takto 43 hrán `inside`, zväčša „všetko na linke je v kuchynskom ostrove“). Ponechá sa
  najmenší kontajner.
- **on**(A, B): najvyššia opora A (pohár na podnose na stole je na podnose). A v kontajneri
  stojí iba na kontajneri samotnom alebo na uzloch v ňom (krabica v skrinke nie je na podlahe).
  Každý uzol má najviac jednu hranu `on`; cyklus (možný iba cez pravidlo spodku) sa rozbije
  pri najslabšej hrane.
- **near**(A, B): ani jeden nie je štruktúra, medzi nimi nie je `on`/`inside`, najmenšia
  vzdialenosť bodov (podmnožina `near_points`) < δ; `confidence` = 1 − d/δ.
- `confidence` hrany `on` = prekryv pôdorysov × (1 − ½ |Δz|/ε), pri podlahe 1 − ½ |Δz|/ε;
  `inside` = podiel bodov A v OBB B.

## Implementácie

| Názov | Popis | Licencia |
|---|---|---|
| `geometric_vlm` (primárna) | kroky 1–4 vlastný kód, caption cez VLM (`geometric_vlm.py`; predvolene Qwen3-VL-4B-Instruct 4-bit cez `transformers` + `bitsandbytes`), krok 5 (LLM) zatiaľ nie | vlastný kód; Qwen3-VL-4B-Instruct (Apache 2.0), na silnejšej GPU Qwen2.5-VL-7B / Qwen3-VL-8B (Apache 2.0) |
| `geometric_only` (baseline) | bez VLM/LLM; caption = label, iba geometrické vzťahy (`phases/p08_scene_graph/geometric_only.py`, spoločná časť `common.py`) | vlastný kód |
| `florence` (ľahká alternatíva) | popisy cez Florence-2-large (`florence.py`, úloha `<CAPTION>` na výreze s vymaskovaným pozadím, jeden pohľad); neodpovedá na otázky, takže labely neopravuje a atribúty sú prvé slovo farby a materiálu z popisu (bez „on a gray background“) | MIT |

ConceptGraphs (MIT) slúži ako referencia pre postup (captions cez VLM, vzťahy cez LLM),
kód sa dá čiastočne prevziať pri dodržaní licencie (MIT — zachovať copyright notice).

Pri LLM/VLM over licenciu **konkrétnej veľkosti** modelu (napr. Qwen2.5-VL-3B a -72B majú
iné licencie ako 7B). Multimodálne Llama modely nepoužívať (licencia ich v EÚ neumožňuje, `ZAKÁZANÉ`).
Lokálny beh: `transformers`, `vLLM` (Apache 2.0) alebo `llama.cpp` / `Ollama` (MIT).

## Konfigurácia

```yaml
scene_graph:
  impl: geometric_vlm
  vlm: Qwen/Qwen3-VL-4B-Instruct   # Apache 2.0; 4-bit ~2,9 GB VRAM
  vlm_quantization: 4bit      # none | 8bit | 4bit (bitsandbytes, iba CUDA)
  vlm_compute_dtype: auto     # auto | float16 | float32 (auto: float32 na GTX 16xx)
  device: auto                # auto | cuda | cpu
  views_per_caption: 2
  crop_size: 448              # strana štvorcového výrezu pre VLM, px
  crop_context: 1.6           # strana výrezu = dlhšia strana bboxu × crop_context
  vlm_batch_size: 1           # uzly na jedno volanie generate()
  max_new_tokens: 80
  label_correction: vocabulary   # off | vocabulary | any
  relabel_check: true         # potvrdenie opravy výberom starý/nový label v každom pohľade
  describe_structure: false   # popisy aj pre podlahu, steny
  # iba impl florence (vlastné kľúče, vlm*/crop* patria geometric_vlm):
  # florence_model: florence-community/Florence-2-large
  # florence_task: <CAPTION>        # <CAPTION> | <DETAILED_CAPTION>
  # florence_crop_size: 768
  # florence_crop_context: 1.2
  # florence_batch_size: 1
  obb_mode: gravity_aligned   # gravity_aligned | full
  obb_trim_percent: 1.5
  min_opacity: 0.1            # geometria iba z gaussiánov s aspoň takouto opacity
  min_extent_rel: 0.002       # najmenšia strana OBB (ploché objekty), × mierka scény
  bottom_percentile: 0.5      # spodok objektu pre `on` (percentil výšok bodov)
  max_points: 5000            # podmnožina bodov uzla na vzťahy
  num_best_views: 3
  best_view_min_gap: 5        # odstup najlepších pohľadov v spracovaných snímkach
  border_penalty: 0.5         # skóre pohľadu, ak sa maska dotýka okraja snímky
  structure_labels: [floor, wall, ceiling]   # + tagy `stuff` z 04
  on_eps_rel: 0.02
  on_overlap: 0.3             # prekryv pôdorysov (podiel menšieho)
  support_scale: 1.5          # zväčšenie pôdorysu A pri hľadaní opory pod ním
  inside_share: 0.6           # podiel bodov A v OBB B
  near_delta_rel: 0.05
  near_points: 1000           # podmnožina bodov na vzdialenosť pri `near`
  semantic_relations: false
```

## Hardvér

Geometria: CPU. Qwen2.5-VL-7B: ~16 GB VRAM v bf16, ~6–8 GB pri 4-bit kvantizácii.
Qwen3-VL-4B 4-bit (NF4, dvojitá kvantizácia): 2,8 GB po načítaní, špička 3,0 GB pri dvoch
výrezoch 448 px (~410 tokenov) — zmestí sa na GTX 1650 (4 GB). Na GTX 16xx treba počítať
4-bit vrstvy vo float32 (`vlm_compute_dtype: auto`): fp16 matmul tam nemá tensor cores
a prefill trval 14 s namiesto 6 s. Dávky (`vlm_batch_size` > 1) s bitsandbytes 4-bit na
GTX 1650 nepomohli (25,9 vs. 23,7 s na uzol, výsledky zhodné). Florence-2-large: fp32
špička 3,4 GB, 1,9 s na výrez (fp16: 1,8 GB, ale 6,2 s); dávka 8 výrezov v fp16 potrebuje
4,5 GB a na 4 GB GPU vrátila prázdne odpovede, preto `florence_batch_size: 1`.

## Validácia výstupu

- Každý uzol má unikátne `node_id`, platné `instance_id` z 07, `obb.rotation` ortonormálna,
  `extent` > 0.
- Každá hrana odkazuje na existujúce uzly, `relation` je z povoleného zoznamu, žiadne
  slučky (A, A), `on` nemá cykly.
- Počet uzlov = počet inštancií z 07 (alebo zdokumentovaný dôvod vynechania).

## Riziká a otvorené otázky

- Neúplné objekty (videné z jednej strany) → podhodnotené OBB → chybné `on`.
- Halucinácie VLM v captions (najmä farby pri zlom osvetlení) → caption len ako doplnok,
  label z hlasovania má prednosť.
- Hierarchia (izba → objekty) nie je potrebná pri jednej izbe; HOV-SG ostáva ako možné rozšírenie.

## Evaluácia

Presnosť labelov uzlov; presnosť a úplnosť hrán `on`/`near` voči ručnej anotácii testovacej
scény; ľudské hodnotenie captions (stačí malá vzorka).

## Stav

Implementácia: `src/scenepipe/phases/p08_scene_graph/` (`common.py`, `geometric_only.py`,
`geometric_vlm.py`, `florence.py`),
formát grafu `src/scenepipe/core/scene_graph.py`, stav `src/scenepipe/core/state.py`;
test: `tests/contracts/test_p08.py` (syntetická izba z kvádrov: podlaha, otočený stôl
s tenkými nohami, pohár, stolička, gauč s operadlom a vankúšom, skrinka s krabicou, miska
s jablkom, rastlina širšia než kvetináč; všetky očakávané `on`/`inside` presne, ~3 s na CPU;
`geometric_vlm` s falošným VLM: výrezy, oprava labelu v slovníku aj v množnom čísle,
odmietnutá oprava mimo slovníka a overením, nespracovateľná odpoveď, pomenovanie uzla bez
labelu). `geometric_only` beží v ľubovoľnom prostredí s numpy (bez torch a GPU),
`geometric_vlm` v `sp-gs` s `bitsandbytes` (pozri `envs/README.md`).

- [x] OBB/AABB z gaussiánov (`gravity_aligned`, voliteľne `full`)
- [x] výber najlepších pohľadov
- [x] geometrické vzťahy + testy na syntetických krabiciach
- [x] captions, atribúty a overené opravy labelov cez VLM (`geometric_vlm`, Qwen3-VL-4B 4-bit)
- [ ] (voliteľné) sémantické vzťahy cez LLM
- [x] vizualizácia (`graph.html`), zápis, vytvorenie `state/v000`
- [x] kontraktový test
- [x] beh `geometric_only` na Mip-NeRF 360 (2026-11-08, Windows, CPU; vstup 07 s profilom
  `fast`), celá fáza 2–6 s:

  | Scéna | Uzly | Štruktúra | `on` | `inside` | `near` | Bez opory | ε / δ |
  |---|---|---|---|---|---|---|---|
  | *counter* | 70 | 0 | 39 | 4 | 182 | 27 | 0,027 / 0,067 |
  | *room* | 52 | 1 (podlaha) | 32 | 3 | 74 | 16 | 0,027 / 0,069 |
  | *kitchen* | 8 | 0 | 6 | 0 | 10 | 2 | 0,027 / 0,067 |

  Prvá verzia (`inside` = väčšina bodov v OBB, `on` iba pre menší pôdorys, podlaha ako
  obdĺžnik) dala na *counter* 43 hrán `inside` („všetko na linke je v kuchynskom ostrove“,
  riedka inštancia s 1 909 gaussiánmi a kvádrom 2 × 2 m), na *room* vankúše a plyšového
  medveďa „v“ gauči a kresle a gauč bez opory, na *kitchen* rohož „v“ stole. Po úprave
  pravidiel (Postup, Presne): *counter*: jablká a pomaranče v miske alebo na nej, cibuľa
  v mise, rastlina na kvetináči, potraviny a náradie na linke; *room*: gauč, stolík, rastlina,
  reproduktory, police a záclony na podlahe, hrniec na stolíku, televízor a reproduktory
  na polici, papuča na koberci. Zvyšné chyby sú najmä zo 07: mylné labely (*curtain* na gauči,
  *doorway* v polici), duplicitné a čiastočné inštancie (*table* na *table*, *stove* na
  *stove*), riedke inštancie v pozadí bez opory. Kreslo na *room* nemá oporu: jeho spodok je
  5 cm (2 ε) nad podlahou. Presnosť a úplnosť hrán voči ručnej anotácii zatiaľ nie je zmeraná.
- [x] beh `geometric_vlm` na Mip-NeRF 360 (2026-11-08, WSL `sp-gs`, GTX 1650, Qwen3-VL-4B
  4-bit, `default.yaml`; vstup 07 s profilom `fast`). Geometria a hrany sú rovnaké ako
  v `geometric_only` (opravy labelov štruktúru nezmenili):

  | Scéna | Popísané uzly | Nečitateľné odpovede | Opravy prijaté / odmietnuté overením | Načítanie VLM | s na uzol | Popisy spolu |
  |---|---|---|---|---|---|---|
  | *counter* | 70 / 70 | 0 | 15 / 10 | 59 s | 17,7 | 21,6 min |
  | *room* | 51 / 52 (podlaha nie) | 0 | 7 / 7 | 59 s | 17,2 | 15,6 min |
  | *kitchen* | 8 / 8 | 0 | 2 / 2 | 57 s | 15,7 | 3,0 min |

  Popisy a atribúty sú väčšinou vecné („black grand piano with sheet music“, „white
  cylindrical speaker“, „two apples in wooden bowl“, „jar of organic coconut oil“) a odhaľujú
  aj chyby 07, ktoré slovník 04 opraviť nedovolí (*spatula* je nôž, *juicer* chladnička,
  *spatula* chňapka). Kontrola výrezov prijatých opráv na *counter*: ~10 z 15 zjavne lepších
  (pomaranč v sieťke → *onion*, *spatula* → *bag* so zipsom, *juicer*/*blender* → *pitcher*
  pri sklenenej karafe, *rolling pin* → *onion*), 4 neutrálne (*bag* → *package*,
  *pitcher* → *jug* pri kanvici), 1 sporná (*apple* → *peach*). Z 10 odmietnutých bolo
  5 správne odmietnutých (drevená miska → *apple*, červená cibuľa → *apple*, plechovka →
  *plant*, dvierka → *counter*), 3 by boli lepšie (pomaranč → *onion*, časť kuchynského
  robota → *food processor*). Na *room* prijaté: *doorway* → *entertainment center* (TV
  stolík), *curtain* → *speaker* (biely valcový reproduktor), *curtain* → *pillow* (deka na
  gauči), *pillow* → *sofa* (maska je bočnica gauča). Prvá verzia promptu (dlhší JSON bez
  upozornenia na časti objektov, bez overenia) na *room* zmenila 16 labelov, z toho 3× vankúš
  na *sofa* a *teddy bear* na *pillow*.

  Rýchlosť (GTX 1650): fp16 výpočet 4-bit vrstiev 27 s na uzol (dávka 4), float32 15 s;
  krátky jednoriadkový JSON čas takmer nezmenil (prefill je ~6 s, dekódovanie ~0,17 s na token).
  Profil `fast` (jeden výrez): *kitchen* 13,0 namiesto 15,5 s na uzol. Ľudské hodnotenie
  popisov na vzorke zatiaľ chýba.
- [x] beh `florence` na *room* (porovnanie, výstup sa nepoužil): 1,9 s na uzol (2,0 min
  popisy, načítanie 21 s) oproti 17,2 s pri Qwen3-VL-4B, ale popisy sú slabšie: opisuje sivé
  pozadie výrezu („a black arrow on a gray background“ pri kresle, „A 3D rendering of a piece
  of wood“ pri vankúši), mýli si objekty (reproduktor ako „camera“ či „television“) a nevie
  opraviť label. Ako primárna ostáva `geometric_vlm`; `florence` sa hodí na rýchly náhľad.

## Zdroje

- Gu et al.: ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning, ICRA 2024.
- Werby et al.: Hierarchical Open-Vocabulary 3D Scene Graphs (HOV-SG), RSS 2024.
- Bai et al.: Qwen2.5-VL Technical Report, 2025.
- Qwen Team: Qwen3-VL Technical Report, 2025.
