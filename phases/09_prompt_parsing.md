# Fáza 09: Parsovanie promptu

`contract_version: 1`

## Účel

Preložiť voľný textový (a obrázkový) prompt na **štruktúrovaný plán úprav**: zoznam operácií,
pri každej typ operácie, slovný popis cieľa a parametre. Táto fáza **nerozhoduje, ktorý uzol**
je cieľ — to robí 10. Oddelenie umožňuje vymeniť LLM bez zmeny groundingu a testovať parsovanie
bez 3D scény.

## Kontrakt

**Vstup**
- `edits/eNNN/prompt.json` (formát v `docs/data-contracts.md`, sekcia 10)
- `state/<base_state>/state.json` → `scene_graph.json` — LLM dostane **zhrnutie** grafu
  (zoznam labelov a captions, bez geometrie), aby vedel, aké objekty existujú.

**Výstup** (`edits/eNNN/09_prompt_parsing/`)
- `edit_plan.json`:
  ```json
  {
    "contract_version": 1,
    "source_text": "vymeň hnedý gauč za modrý stôl a zhasni lampu",
    "language": "sk",
    "ops": [
      {"op_id": "op01", "op": "replace",
       "target": {"text": "brown sofa", "label": "sofa",
                  "attributes": {"color": "brown"}, "relation": null},
       "params": {"asset_prompt": "blue table", "asset_image": null,
                  "keep_footprint": true}},
      {"op_id": "op02", "op": "unsupported",
       "reason": "lighting changes are not supported", "raw": "zhasni lampu"}
    ],
    "image_roles": {"edits/e001/ref.jpg": "asset"}
  }
  ```
- `llm_raw.json` — surová odpoveď LLM a použitý prompt (pre ladenie a do práce).
- `manifest.json`.

### Povolené operácie a parametre

| `op` | Cieľ | `params` |
|---|---|---|
| `recolor` | objekt | `color` (názov alebo RGB), `strength` ∈ (0, 1] |
| `delete` | objekt | — |
| `translate` | objekt | `direction` (`toward`/`away`/`up`/…), `reference` (iný objekt, voliteľné), `distance_rel`, alebo `place_on` (cieľová podložka) |
| `rotate` | objekt | `angle_deg` okolo osi Z |
| `scale` | objekt | `factor` |
| `replace` | objekt | `asset_prompt` a/alebo `asset_image`, `keep_footprint` |
| `insert` | — (miesto) | `asset_prompt`/`asset_image`, `place_on` (popis podložky), `near` (voliteľné) |
| `style` | celá scéna alebo objekt | `style_prompt` |
| `unsupported` | — | `reason`, `raw` |

Popis cieľa (`target`) je **textový**: `text` (anglická fráza pre CLIP), `label`, `attributes`,
`relation` (napr. `{"type": "on", "anchor": {"text": "table"}}`) a voliteľne `ordinal`
(„ľavá“, „najväčšia“) — tie rieši 10.

## Postup

1. Zostaviť systémový prompt: povolené operácie (tabuľka vyššie), JSON schéma, zoznam objektov
   v scéne (label + caption), niekoľko príkladov (few-shot) v slovenčine aj angličtine.
2. Zavolať LLM s **obmedzeným výstupom na JSON schému** (grammar/structured output vo vLLM,
   llama.cpp, alebo knižnica `outlines`).
3. Validovať voči schéme; pri chybe jeden pokus o opravu s chybovou hláškou, potom `unsupported`.
4. Normalizácia: `target.text` vždy po anglicky (CLIP je anglický), farby na RGB cez tabuľku.
5. Obrázky s `role: auto`: rozhodnúť podľa textu („takýto stôl“ → `asset`, „tento objekt“ → `target`).

## Implementácie

| Názov | Popis | Licencia |
|---|---|---|
| `llm_local` (primárna) | lokálny LLM so štruktúrovaným výstupom, napr. Qwen3-8B / Qwen2.5-7B-Instruct | Apache 2.0 |
| `rules` (baseline) | regulárne výrazy a slovník pre jednoduché vety („zmeň farbu X na Y“, „odstráň X“) | vlastný kód |

Baseline je dôležitý: ukáže, čo pridáva LLM, a umožní testovať fázy 10–12 bez GPU.

## Konfigurácia

```yaml
prompt_parsing:
  impl: llm_local
  model: Qwen/Qwen3-8B
  backend: transformers       # transformers | vllm | llamacpp
  quantization: 4bit
  temperature: 0.0
  max_retries: 1
  include_scene_summary: true
```

## Hardvér

7–8B model: ~16 GB VRAM v bf16, ~5–6 GB pri 4-bit; na CPU cez llama.cpp pomaly, ale ide.

## Validácia výstupu

- `edit_plan.json` prechádza JSON schémou (`src/scenepipe/core/schemas/edit_plan.json`).
- `op_id` unikátne, `op` z povoleného zoznamu, povinné parametre prítomné.

## Riziká a otvorené otázky

- Slovenské prompty: menšie modely robia chyby → few-shot príklady v slovenčine, prípadne
  najprv preklad do angličtiny tým istým LLM.
- Viacero operácií v jednom prompte: poradie vykonania = poradie v pláne; závislosti
  („presuň stoličku k novému stolu“) zatiaľ nepodporované → `unsupported`.
- Nejednoznačný prompt („zmeň to“) → `unsupported` s dôvodom, alebo otázka používateľovi
  pri `interactive: true`.

## Evaluácia

Testovacia sada ~50 promptov (sk/en) s ručne zostaveným očakávaným plánom: presnosť typu
operácie, cieľa a parametrov; porovnanie `llm_local` vs. `rules`.

## Stav

- [ ] JSON schéma plánu
- [ ] baseline `rules`
- [ ] LLM so štruktúrovaným výstupom
- [ ] testovacia sada promptov + vyhodnotenie
- [ ] kontraktový test
