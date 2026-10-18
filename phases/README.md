# Fázy reťazca: prehľad

Reťazec je rozdelený na 14 fáz. Oproti pôvodnému návrhu (6 fáz v ročníkovom projekte) sú
zmeny tieto:

- **Rekonštrukcia 3DGS je presunutá dopredu** (fáza 02). Úprava aj lifting pracujú nad
  gaussiánmi, takže gaussiány musia existovať skôr. Na konci ostáva len renderovanie (13).
- **Vyčistenie gaussiánov je samostatná fáza** (03 refinement): odstránenie floaterov a artefaktov
  po tréningu sa dá vymeniť alebo vypnúť bez zásahu do rekonštrukcie.
- **2D segmentácia je rozdelená na tri fázy** (04 tagovanie, 05 segmentácia, 06 embeddingy),
  aby sa tagger, segmentátor a príznakový model dali vymeniť nezávisle.
- **Lifting ide priamo na gaussiány** (07), nie na riedky SfM pointcloud.
- **Úprava promptom je rozdelená na štyri fázy**: parsovanie promptu (09), grounding cieľa
  v grafe (10), generovanie nového objektu (11) a samotná úprava (12).

| # | Fáza | Časť | Vstup (hlavný) | Výstup (hlavný) | Primárna implementácia |
|---|---|---|---|---|---|
| 00 | [Ingest](00_ingest.md) | build | fotky / video | normalizované snímky + `frames.json` | vlastný kód (OpenCV, ffmpeg) |
| 01 | [SfM](01_sfm.md) | build | snímky | kamery + riedky pointcloud (COLMAP + `cameras.json`) | COLMAP 4.x |
| 02 | [Rekonštrukcia](02_reconstruction.md) | build | snímky + kamery + pointcloud | `gaussians.ply` | gsplat |
| 03 | [Refinement](03_refinement.md) | build | gaussiány + kamery + snímky | vyčistený `gaussians.ply` | vlastný kód (gsplat) |
| 04 | [Tagovanie](04_tagging.md) | build | snímky | slovník objektov `tags.json` | RAM++ |
| 05 | [Segmentácia](05_segmentation.md) | build | snímky + slovník | masky inštancií + `detections.json` | Grounding DINO + SAM 2 |
| 06 | [Embeddingy masiek](06_mask_embedding.md) | build | snímky + masky | CLIP vektor pre každú masku | OpenCLIP |
| 07 | [Lifting](07_lifting.md) | build | gaussiány + kamery + masky + embeddingy | inštancia pre každý gaussián | vlastný kód (gsplat) |
| 08 | [Scene graph](08_scene_graph.md) | build | gaussiány + inštancie + snímky | `scene_graph.json` | vlastný kód + Qwen2.5-VL |
| 09 | [Parsovanie promptu](09_prompt_parsing.md) | edit | prompt + graf | `edit_plan.json` (nevyriešené ciele) | Qwen (LLM) |
| 10 | [Grounding](10_grounding.md) | edit | plán + graf + embeddingy | plán s ID uzlov | vlastný kód (CLIP + graf) |
| 11 | [Generovanie objektu](11_asset_generation.md) | edit | text / obrázok | `asset.ply` v kanonickom tvare | FLUX.1-schnell + TRELLIS |
| 12 | [Úprava](12_editing.md) | edit | stav `vN` + plán + assety | stav `vN+1` | vlastné operátory + LaMa + gsplat |
| 13 | [Renderovanie](13_rendering.md) | obe | stav + kamery | snímky / video / viewer | gsplat + viser |

## DAG závislostí

```mermaid
flowchart LR
    IN[fotky / video] --> P00[00 Ingest]
    P00 --> P01[01 SfM]
    P00 --> P04[04 Tagovanie]
    P01 --> P02[02 Rekonštrukcia 3DGS]
    P00 --> P02
    P02 --> P03[03 Refinement]
    P01 --> P03
    P04 --> P05[05 Segmentácia]
    P00 --> P05
    P01 -.poradie snímok.-> P05
    P05 --> P06[06 Embeddingy]
    P03 --> P07[07 Lifting]
    P01 --> P07
    P05 --> P07
    P06 --> P07
    P07 --> P08[08 Scene graph]
    P03 --> P08
    P08 --> S0[(state/v000)]

    PR[prompt] --> P09[09 Parsovanie]
    S0 --> P09
    P09 --> P10[10 Grounding]
    S0 --> P10
    P10 --> P11[11 Generovanie objektu]
    P10 --> P12[12 Úprava]
    P11 --> P12
    S0 --> P12
    P12 --> S1[(state/v001)]
    S1 --> P13[13 Renderovanie]
```

Fázy 04 → 05 → 06 nezávisia od 02 a 03, dajú sa teda spustiť paralelne s tréningom gaussiánov.

## Stav scény

Výstup buildu aj každej úpravy je **stav scény** (`state/vNNN/state.json`), ktorý ukazuje na
štyri súbory: gaussiány, priradenie gaussiánov k inštanciám, graf scény a príznaky uzlov.
Fázy 09–13 čítajú a zapisujú výhradne stavy. Podrobne v `docs/data-contracts.md`.

## Poradie implementácie (odporúčané)

1. **00 → 01 → 02 → 03 → 13**: najprv funkčná fotorealistická scéna, ktorú vieš pozrieť vo vieweri.
   Všetko „out of the box“, overí to prostredie a hardvér.
2. **04 → 05 → 06**: 2D sémantika, dá sa ladiť na jednotlivých obrázkoch.
3. **07 → 08**: jadro sémantickej vrstvy, vlastný kód.
4. **09 → 10 → 12** (operátory recolor, delete, transform): prvé úpravy bez generatívnych modelov.
5. **11 → 12** (operátory replace, insert, style): generatívna časť, najnáročnejšia na hardvér.

## Šablóna dokumentu fázy

Každý `XX_*.md` má rovnaké sekcie: Účel, Kontrakt (vstupy, výstupy, `contract_version`),
Postup, Implementácie (primárna a alternatívy s licenciami), Konfigurácia, Hardvér,
Validácia výstupu, Riziká a otvorené otázky, Evaluácia, Stav, Zdroje.
