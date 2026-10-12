# Fáza 07: Graf scény (scene graph)

`contract_version: 1`

## Účel

Postaviť symbolickú reprezentáciu scény: **uzly** = objekty (s geometriou, labelom, popisom
a príznakom), **hrany** = vzťahy medzi nimi (*on*, *near*, *inside*, …). Graf je to, na čo
ukazuje prompt („pohár na stole“), a to, čo úprava (11) používa na umiestnenie objektov.

Po tejto fáze runner vytvorí počiatočný stav scény `state/v000`.

## Kontrakt

**Vstup**
- `02_reconstruction/gaussians.ply`
- `06_lifting/gaussian_instances.npz`, `instances.json`, `instance_features.npy`
- `01_sfm/cameras.json`, `01_sfm/images_undistorted/`
- `04_segmentation/detections.json`, `masks/` (výber najlepších pohľadov na objekt)

**Výstup** (`07_scene_graph/`)
- `scene_graph.json`:
  ```json
  {
    "contract_version": 1,
    "up_axis": "+z",
    "embedding_model": "open_clip:ViT-H-14/laion2b_s32b_b79k",
    "nodes": [
      {"node_id": "n0000", "instance_id": 0, "label": "sofa",
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
- `node_features.npy` — `float16[K, D]` (kópia/odvodenie z 06, riadok podľa `feature_row`).
- `graph.html` alebo `graph.png` — vizualizácia grafu (kontrola očami).
- `manifest.json`.

`is_structure: true` pre podlahu, steny, strop (ak boli v slovníku) — slúžia ako podložky,
ale nie sú bežnými cieľmi úprav.

## Postup

1. **Geometria uzlov** z gaussiánov inštancie (iba gaussiány s dostatočnou opacity):
   - **OBB** s osou Z zarovnanou s gravitáciou (nábytok stojí vzpriamene): PCA v rovine XY
     pre natočenie, rozsah po odstránení 1–2 % extrémov. Plné 3D OBB len ako voliteľné.
   - AABB pre rýchle testy.
2. **Najlepšie pohľady**: snímky, kde je objekt najviac viditeľný (veľká plocha masky, nedotýka
   sa okraja, vysoká kvalita) → `best_views`.
3. **Caption a atribúty**: výrez najlepšieho pohľadu (alebo 2–3 pohľadov) → VLM s promptom
   na krátky popis + JSON atribúty (farba, materiál). Výsledný label: konsenzus label z 06,
   VLM ho môže opraviť, ak je veľmi istý (zaznamenať pôvodný aj nový).
4. **Geometrické vzťahy** (analyticky, deterministicky):
   - `on(A, B)`: spodná podstava A blízko hornej plochy B (|z_min(A) − z_max(B)| < ε), prekryv
     priemetov do XY nad prahom, A je menší.
   - `inside(A, B)`: OBB A z väčšej časti vo vnútri OBB B.
   - `near(A, B)`: vzdialenosť povrchov (aproximácia cez OBB) < δ relatívne k veľkosti scény.
   - `above/below`, `left_of/right_of` sa **neukladajú** — závisia od pohľadu; rieši ich 09
     pri groundingu vzhľadom na kameru.
   - prahy ε, δ relatívne k mierke scény (normalizovaná v 01).
5. **Sémantické vzťahy** (voliteľné, LLM): pre páry blízkych objektov poslať LLM labely,
   captions a geometrické fakty → vzťahy ako `part_of`, `supports`, `belongs_to_set`.
   Iba ako doplnok, geometrické vzťahy sú zdroj pravdy pre umiestňovanie.
6. Uložiť JSON (+ NetworkX sa dá postaviť pri načítaní cez `core/graph.py`).

## Implementácie

| Názov | Popis | Licencia |
|---|---|---|
| `geometric_vlm` (primárna) | kroky 1–4 vlastný kód, caption cez VLM, krok 5 voliteľný cez LLM | vlastný kód; Qwen2.5-VL-7B-Instruct (Apache 2.0) |
| `geometric_only` (baseline) | bez VLM/LLM; caption = label, iba geometrické vzťahy | vlastný kód |
| `florence` (ľahká alternatíva) | captions cez Florence-2 (malý model, rýchly) | MIT |

ConceptGraphs (MIT) slúži ako referencia pre postup (captions cez VLM, vzťahy cez LLM),
kód sa dá čiastočne prevziať pri dodržaní licencie (MIT — zachovať copyright notice).

Pri LLM/VLM over licenciu **konkrétnej veľkosti** modelu (napr. Qwen2.5-VL-3B a -72B majú
iné licencie ako 7B). Llama modely nepoužívať (vlastná licencia, obmedzenia v EÚ pri multimodálnych).
Lokálny beh: `transformers`, `vLLM` (Apache 2.0) alebo `llama.cpp` / `Ollama` (MIT).

## Konfigurácia

```yaml
scene_graph:
  impl: geometric_vlm
  vlm: Qwen/Qwen2.5-VL-7B-Instruct
  vlm_quantization: 4bit      # none | 8bit | 4bit
  views_per_caption: 2
  obb_mode: gravity_aligned   # gravity_aligned | full
  obb_trim_percent: 1.5
  on_eps_rel: 0.02
  near_delta_rel: 0.05
  semantic_relations: false
```

## Hardvér

Geometria: CPU. Qwen2.5-VL-7B: ~16 GB VRAM v bf16, ~6–8 GB pri 4-bit kvantizácii.
Florence-2: < 2 GB.

## Validácia výstupu

- Každý uzol má unikátne `node_id`, platné `instance_id` z 06, `obb.rotation` ortonormálna,
  `extent` > 0.
- Každá hrana odkazuje na existujúce uzly, `relation` je z povoleného zoznamu, žiadne
  slučky (A, A), `on` nemá cykly.
- Počet uzlov = počet inštancií z 06 (alebo zdokumentovaný dôvod vynechania).

## Riziká a otvorené otázky

- Neúplné objekty (videné z jednej strany) → podhodnotené OBB → chybné `on`.
- Halucinácie VLM v captions (najmä farby pri zlom osvetlení) → caption len ako doplnok,
  label z hlasovania má prednosť.
- Hierarchia (izba → objekty) nie je potrebná pri jednej izbe; HOV-SG ostáva ako možné rozšírenie.

## Evaluácia

Presnosť labelov uzlov; presnosť a úplnosť hrán `on`/`near` voči ručnej anotácii testovacej
scény; ľudské hodnotenie captions (stačí malá vzorka).

## Stav

- [ ] OBB/AABB z gaussiánov
- [ ] výber najlepších pohľadov
- [ ] geometrické vzťahy + testy na syntetických krabiciach
- [ ] captions a atribúty cez VLM
- [ ] (voliteľné) sémantické vzťahy cez LLM
- [ ] vizualizácia, zápis, vytvorenie `state/v000`
- [ ] kontraktový test

## Zdroje

- Gu et al.: ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning, ICRA 2024.
- Werby et al.: Hierarchical Open-Vocabulary 3D Scene Graphs (HOV-SG), RSS 2024.
- Bai et al.: Qwen2.5-VL Technical Report, 2025.
