# Fáza 09: Grounding (nájdenie cieľa v grafe)

`contract_version: 1`

## Účel

Pre každú operáciu z plánu (08) nájsť konkrétny **uzol grafu** (alebo miesto pre `insert`),
na ktorý sa vzťahuje. Kombinuje sémantickú podobnosť (CLIP), atribúty z grafu a vzťahy
(„pohár **na stole**“, „**ľavá** stolička“).

## Kontrakt

**Vstup**
- `edits/eNNN/08_prompt_parsing/edit_plan.json`
- `state/<base_state>/state.json` → `scene_graph.json`, `node_features.npy`, `embedding_model`
- `01_sfm/cameras.json` (pre vzťahy závislé od pohľadu, napr. „ľavý“)
- voliteľne obrázky z promptu s rolou `target`

**Výstup** (`edits/eNNN/09_grounding/`)
- `grounded_plan.json` — kópia plánu, kde každá operácia má navyše:
  ```json
  "grounding": {
    "node_id": "n0000",
    "score": 0.81,
    "margin": 0.23,
    "candidates": [{"node_id": "n0000", "score": 0.81}, {"node_id": "n0005", "score": 0.58}],
    "reference_view": "f000052",
    "status": "ok"
  }
  ```
  `status`: `ok` | `ambiguous` (malý rozdiel medzi 1. a 2. kandidátom) | `not_found`.
  Pre `insert` a `translate` s podložkou: `placement: {"support_node_id": "n0002", "anchor_node_id": null}`.
- `debug/` — render referenčného pohľadu so zvýrazneným vybraným uzlom a kandidátmi.
- `manifest.json`.

## Postup

1. **Text embedding** `target.text` modelom z `embedding_model` (rovnaký ako 05; prompt template
   z `05_mask_embedding/meta.json`).
2. **Sémantické skóre** každého uzla: kosínusová podobnosť s `node_features`; kombinácia
   s prekryvom labelu (`target.label` vs. `node.label`, synonymá) a atribútov
   (`target.attributes` vs. `node.attributes` z captions).
   `s(n) = a·cos + b·label_match + c·attr_match`, váhy v konfigurácii.
3. **Obrázkový cieľ**: ak je obrázok s rolou `target`, jeho CLIP obrazový embedding
   ako ďalší člen skóre.
4. **Vzťahy**: ak `target.relation` (napr. `on` + anchor „table“) → najprv grounding anchoru
   (rekurzívne), potom ponechať len kandidátov, ktorí majú v grafe danú hranu s anchorom.
5. **Ordinály a smer pohľadu** („ľavá“, „najbližšia“, „najväčšia“): zvoliť referenčný pohľad
   (kamera, z ktorej sú kandidáti najlepšie viditeľní, alebo zadaná používateľom) a usporiadať
   kandidátov podľa projekcie stredu OBB / vzdialenosti / objemu.
6. **Rozhodnutie**: najvyššie skóre; ak `margin < ambiguity_margin` → `ambiguous`
   (pri `interactive: true` sa opýtať používateľa, inak vziať najlepšieho a zapísať varovanie).
7. **Podložka pre `insert`/`translate place_on`**: grounding `place_on` popisu; ak chýba,
   najbližšia podlaha (`is_structure`) v referenčnom pohľade.

## Implementácie

| Názov | Popis | Licencia |
|---|---|---|
| `clip_graph` (primárna) | postup vyššie | vlastný kód + OpenCLIP (MIT) |
| `label_only` (baseline) | len zhoda labelu, prvý nájdený | vlastný kód |
| `llm_select` (alternatíva) | LLM dostane zoznam uzlov (label, caption, vzťahy) a vyberie ID; skóre z CLIP ako pomocná informácia | Apache 2.0 (model) |

## Konfigurácia

```yaml
grounding:
  impl: clip_graph
  w_clip: 1.0
  w_label: 0.5
  w_attr: 0.3
  w_image: 1.0
  ambiguity_margin: 0.05
  reference_view: auto        # auto | <frame_id>
```

## Hardvér

Text enkodér CLIP — CPU stačí. Rýchle (sekundy).

## Validácia výstupu

- Každá operácia okrem `unsupported` má `grounding.status`; pri `ok`/`ambiguous` existuje `node_id`
  v grafe stavu `base_state`.
- `embedding_model` použitý tu = `embedding_model` v stave (inak chyba, nie varovanie).

## Riziká a otvorené otázky

- CLIP mieša podobné objekty (kreslo vs. gauč) → label a atribúty ako protiváha.
- „Ľavý/pravý“ závisí od pohľadu → bez referenčného pohľadu nejednoznačné; dokumentovať voľbu.
- Viacnásobné ciele („všetky stoličky“) — rozšírenie: `target.quantifier: "all"` → zoznam uzlov.

## Evaluácia

Presnosť top-1 groundingu na testovacej sade promptov s ručne určeným cieľovým uzlom;
podiel `ambiguous`; porovnanie s baseline.

## Stav

- [ ] sémantické skóre (CLIP + label + atribúty)
- [ ] vzťahy cez hrany grafu
- [ ] ordinály a referenčný pohľad
- [ ] placement (podložka)
- [ ] debug render
- [ ] kontraktový test
