# Fáza 06: Lifting 2D → 3D (inštancie na gaussiánoch)

`contract_version: 1`

## Účel

Preniesť 2D masky a ich embeddingy na **gaussiány**: každému gaussiánu priradiť 3D inštanciu
(objekt) a každej inštancii label a sémantický vektor. Je to most medzi geometrickou vrstvou
(02) a symbolickou vrstvou (07).

V pôvodnom návrhu sa liftovalo na riedky SfM pointcloud cez ConceptGraphs. To má dva problémy:
riedke body nie sú to, čo neskôr upravujeme (gaussiány), a ConceptGraphs očakáva RGB-D senzor.
Táto fáza liftuje priamo na gaussiány, **training-free**, pomocou renderovacích váh z gsplat.
Je to vlastný modul a jeden z hlavných vlastných prínosov práce.

## Kontrakt

**Vstup**
- `02_reconstruction/gaussians.ply`
- `01_sfm/cameras.json`
- `04_segmentation/detections.json`, `04_segmentation/masks/`
- `05_mask_embedding/embeddings.npy`, `index.json`, `meta.json`

**Výstup** (`06_lifting/`)
- `gaussian_instances.npz`:
  - `instance_id: int32[N]` — inštancia pre každý gaussián (index ako v PLY), `-1` = pozadie / neisté,
  - `confidence: float16[N]` — podiel hlasov víťaznej inštancie,
  - `visibility: float32[N]` — celková váha, ktorou gaussián prispel do všetkých pohľadov
    (gaussiány s nulovou viditeľnosťou sú nepozorované).
- `instances.json`:
  ```json
  {
    "contract_version": 1,
    "embedding_model": "open_clip:ViT-H-14/laion2b_s32b_b79k",
    "instances": [
      {"instance_id": 0, "label": "sofa",
       "label_votes": {"sofa": 0.78, "chair": 0.22},
       "num_gaussians": 41233, "num_views": 57,
       "source_masks": ["f000001_m000", "f000002_m001", …],
       "track_ids": ["t0003"],
       "feature_row": 0}
    ]
  }
  ```
- `instance_features.npy` — `float16[K, D]`, L2-normalizované, riadok podľa `feature_row`.
- `debug/` — rendery „instance ID“ farbami z niekoľkých pohľadov vedľa 2D masiek.
- `manifest.json`.

## Postup

### 1. Váhy príspevku gaussiánov k maskám (jadro)

Pri alpha-blendingu prispieva gaussián *i* do pixelu *p* váhou
`w_i(p) = α_i(p) · T_i(p)` (opacity × transmitancia). Pre masku *m* chceme
`W_i(m) = Σ_{p ∈ m} w_i(p)` — koľko „hmoty“ gaussiánu *i* je viditeľné v maske *m*.

Efektívny výpočet cez gsplat (bez úpravy CUDA kódu), **gradientový trik**:
1. Pre snímku s K maskami vytvoriť per-gaussián príznak `f ∈ R^{N×K}`, inicializovaný nulami,
   `requires_grad=True`.
2. Vyrenderovať `f` (gsplat podporuje N-rozmerné príznaky namiesto farby) → obraz `F ∈ R^{H×W×K}`.
3. `L = Σ_p Σ_k M_k(p) · F_k(p)`, kde `M_k` je k-ta maska.
4. `L.backward()` → `∂L/∂f_{i,k} = W_i(m_k)` presne.
5. Rovnako (jeden kanál, celá snímka) sa získa celková viditeľnosť gaussiánu v snímke.

Pamäť: `N×K` floatov na snímku (napr. 2 M × 100 × 4 B = 0,8 GB) → spracovať masky po dávkach
a gaussiány filtrovať na tie vo výhľade kamery (frustum culling). Ukladať riedko (top hodnoty).

Alternatíva bez gradientu: vlastný jednoduchý CUDA/Triton kernel alebo použiť výstup
`gsplat` s per-pixel ID gaussiánov, ak ho daná verzia poskytuje.

### 2. Od masiek ku 3D inštanciám

- **S trackmi** (`has_tracks: true`): masky s rovnakým `track_id` sú jedna kandidátska inštancia.
- **Bez trackov**: pre každú masku množina gaussiánov s `W_i(m)` nad prahom; masky sa spájajú
  **greedy** (ako v ConceptGraphs), ak:
  - vážené IoU / prekryv ich gaussiánových množín > `merge_overlap`, **a**
  - kosínusová podobnosť embeddingov > `merge_sim`.
- Aj pri trackoch spraviť druhé kolo spájania (track sa mohol prerušiť → dva tracky, jeden objekt).

### 3. Hlasovanie o inštancii pre každý gaussián

`score_i(c) = Σ_{m ∈ c} W_i(m) · q(m)`, kde `q(m)` je kvalita masky (`mask_score`, penalizácia
`touches_border`, malá plocha). Gaussián dostane inštanciu s najvyšším skóre, ak:
- `confidence = score_max / Σ score > min_confidence` a
- `score_max / visibility_i > min_coverage` (gaussián bol v maskách často, nielen náhodou),
inak `-1`.

Prekryvné masky (vankúš na gauči): preferovať menšiu inštanciu pri hlasovaní (konfigurovateľné),
inak by veľká inštancia pohltila malé.

### 4. Vyčistenie v 3D

- Odstránenie odľahlých gaussiánov inštancie (štatistický outlier filter / DBSCAN na stredoch,
  ponechať najväčší komponent + blízke).
- Zahodiť inštancie s < `min_gaussians` alebo videné v < `min_views` pohľadoch.

### 5. Label a príznak inštancie

- `label_votes`: vážené hlasovanie labelov zdrojových masiek (+ zero-shot skóre z 05).
- `instance_features`: vážený priemer embeddingov zdrojových masiek (váha = `Σ_i W_i(m)` × kvalita),
  s vynechaním outlierov (embeddingy ďaleko od mediánu), L2 normalizácia.

## Implementácie

| Názov | Popis | Licencia |
|---|---|---|
| `gsplat_vote` (primárna) | postup vyššie, vlastný kód nad gsplat | vlastný kód + gsplat (Apache 2.0) |
| `pointcloud_project` (baseline) | pôvodný návrh: premietnuť **stredy** gaussiánov do snímok, zistiť, či padnú do masky (s kontrolou hĺbky z 02 `depth/`) — jednoduchšie, ignoruje okluziu a rozmer gaussiánov | vlastný kód |

Baseline je užitočný na porovnanie v práci (ukáže prínos váh namiesto bodovej projekcie).
Princíp training-free prenosu masiek na gaussiány je publikovaný napr. vo FlashSplat (ECCV 2024);
kód takých prác **nekopírovať**, ak stojí na INRIA rasterizéri — implementovať vlastný.

## Konfigurácia

```yaml
lifting:
  impl: gsplat_vote
  mask_batch: 32
  min_weight: 0.01
  merge_overlap: 0.3
  merge_sim: 0.75
  min_confidence: 0.5
  min_coverage: 0.2
  prefer_smaller_on_overlap: true
  min_gaussians: 200
  min_views: 3
  outlier: dbscan          # dbscan | statistical | none
```

## Hardvér

GPU (gsplat). Pamäť riadená `mask_batch`; s cullingom a riedkym ukladaním zvládnuteľné v 6–8 GB.

## Validácia výstupu

- `instance_id.shape == (N,)` kde N = počet gaussiánov v PLY; hodnoty ∈ {−1} ∪ [0, K).
- Každá inštancia v `instances.json` má aspoň `min_gaussians` gaussiánov (overiť počítaním).
- `instance_features.npy` má K riadkov, normalizované.
- `embedding_model` = `meta.json.model` z fázy 05.

## Riziká a otvorené otázky

- Veľké gaussiány pozadia zasahujú do viacerých masiek → nízka confidence; zvážiť regularizáciu v 02.
- Objekty videné len z jednej strany → neúplné inštancie (iba viditeľná časť má gaussiány).
- Rozhodnutie časť vs. celok (hierarchia) nie je riešené; aktuálne pravidlo „menšia vyhráva“.
- Kalibrácia prahov — robiť na jednej scéne s ručnou 3D anotáciou a zafixovať.

## Evaluácia

- mIoU renderovaných masiek inštancií voči 2D ground truth na testovacích pohľadoch
  (napr. dataset LERF-Mask alebo 3D-OVS, prípadne vlastné anotácie).
- Porovnanie `gsplat_vote` vs. `pointcloud_project`.
- Počet inštancií vs. ručný počet objektov (nadsegmentácia / podsegmentácia).

## Stav

- [ ] načítanie vstupov, frustum culling
- [ ] gradientový trik pre `W_i(m)` + test na syntetickej scéne (známe výsledky)
- [ ] spájanie masiek do inštancií (s trackmi aj bez)
- [ ] hlasovanie, prahy, vyčistenie
- [ ] labely a príznaky inštancií
- [ ] debug rendery
- [ ] baseline `pointcloud_project`
- [ ] kontraktový test

## Zdroje

- Gu et al.: ConceptGraphs, ICRA 2024 (spájanie inštancií, agregácia príznakov).
- Shen et al.: FlashSplat: 2D to 3D Gaussian Splatting Segmentation Solved Optimally, ECCV 2024.
- Ye et al.: gsplat, 2024.
