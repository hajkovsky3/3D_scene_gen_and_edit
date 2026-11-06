# Fáza 07: Lifting 2D → 3D (inštancie na gaussiánoch)

`contract_version: 2`

## Účel

Preniesť 2D masky a ich embeddingy na **gaussiány**: každému gaussiánu priradiť 3D inštanciu
(objekt) a každej inštancii label a sémantický vektor. Je to most medzi geometrickou vrstvou
(02, vyčistená v 03) a symbolickou vrstvou (08).

V pôvodnom návrhu sa liftovalo na riedky SfM pointcloud cez ConceptGraphs. To má dva problémy:
riedke body nie sú to, čo neskôr upravujeme (gaussiány), a ConceptGraphs očakáva RGB-D senzor.
Táto fáza liftuje priamo na gaussiány, **training-free**, pomocou renderovacích váh z gsplat.
Je to vlastný modul a jeden z hlavných vlastných prínosov práce.

## Kontrakt

**Vstup**
- `03_refinement/gaussians.ply`
- `01_sfm/cameras.json`
- `05_segmentation/detections.json`, `05_segmentation/masks/`
- `06_mask_embedding/embeddings.npy`, `index.json`, `meta.json`

**Výstup** (`07_lifting/`)
- `gaussian_instances.npz`:
  - `instance_id: int32[N]` — inštancia pre každý gaussián (index ako v PLY), `-1` = pozadie / neisté,
  - `confidence: float16[N]` — podiel hlasov víťaznej inštancie,
  - `visibility: float32[N]` — celková váha, ktorou gaussián prispel do všetkých pohľadov
    (gaussiány s nulovou viditeľnosťou sú nepozorované).
- `instances.json`:
  ```json
  {
    "contract_version": 2,
    "embedding_model": "hf:laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
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
  Inštancie sú očíslované 0..K−1 od najväčšej, `feature_row` = `instance_id`.
- `debug/` — rendery „instance ID“ farbami z niekoľkých pohľadov vedľa 2D masiek
  (snímka | masky z 05 | render inštancií, rovnaká farba inštancie vo všetkých pohľadoch).
- `report.json` — počty kandidátov, inštancií, podiel priradených a nepozorovaných gaussiánov.
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

Implementačné poznámky: frustum culling netreba, gsplat gaussiány mimo výhľadu vyradí sám
a `f` má pri 400 k gaussiánoch a 33 kanáloch len ~50 MB. Prvá dávka nesie navyše kanál
samých jednotiek (viditeľnosť v snímke). Režim rasterizácie (antialiased/classic) sa preberá
z manifestu 02. `render_scale` < 1 renderuje v nižšom rozlíšení (masky sa zmenšia
s priemerovaním plochy); podiely `frac` sa tým nemenia. Uchováva sa iba
`frac_i(m) = W_i(m) / vis_i(snímka) ≥ min_weight` (podiel viditeľnej hmoty gaussiánu
v snímke, ktorý padne do masky), takže uloženie je riedke.

### 2. Od masiek ku 3D inštanciám

- **S trackmi** (`has_tracks: true`): masky s rovnakým `track_id` sú jedna kandidátska inštancia.
- **Bez trackov**: pre každú masku množina gaussiánov s `W_i(m)` nad prahom; masky sa spájajú
  **greedy** (ako v ConceptGraphs), ak:
  - vážené IoU / prekryv ich gaussiánových množín > `merge_overlap`, **a**
  - kosínusová podobnosť embeddingov > `merge_sim`.
- Aj pri trackoch spraviť druhé kolo spájania (track sa mohol prerušiť → dva tracky, jeden objekt).

Presne (implementované): *prekryv* kandidátov a, b = min(podiel hlasov a na nosiči b, podiel
hlasov b na nosiči a), kde nosič sú gaussiány s `frac ≥ support_fraction` aspoň v jednej
maske (vankúš na gauči má malý prekryv, lebo gauč má väčšinu hlasov mimo vankúša). Kandidáti,
ktorí sú v niektorej snímke ako dve rôzne masky, sa nespájajú, okrem prekryvu nad
`duplicate_overlap` (0,7): sledovanie SAM 2 niekedy založí druhý track na objekt, ktorý už
sleduje, a takéto duplikáty ležia na tých istých gaussiánoch (na *counter* sa valček rozpadol
na 7 trackov, z toho 3 dvojice súčasných duplikátov). Spájanie ide v kolách
(`merge_rounds`), v kole union-find cez všetky vyhovujúce dvojice od najväčšieho prekryvu.

### 3. Hlasovanie o inštancii pre každý gaussián

`score_i(c) = Σ_{m ∈ c} W_i(m) · q(m)`, kde `q(m)` je kvalita masky (`mask_score`, penalizácia
`touches_border`, malá plocha). Gaussián dostane inštanciu s najvyšším skóre, ak:
- `confidence = score_max / Σ score > min_confidence` a
- `score_max / visibility_i > min_coverage` (gaussián bol v maskách často, nielen náhodou),
inak `-1`.

Prekryvné masky (vankúš na gauči): preferovať menšiu inštanciu pri hlasovaní (konfigurovateľné),
inak by veľká inštancia pohltila malé.

Presne (implementované): pokrytie `coverage_i(c) = Σ_{m∈c} W_i(m) / Σ_f vis_i(f)` cez snímky,
kde má c masku (ako konzistentne leží gaussián v maskách c, keď je c pozorovaná). Gaussián je
*kandidátom* inštancie pri `coverage ≥ min_coverage`. Víťaz: najmenšia kandidátska inštancia
(`prefer_smaller_on_overlap`, veľkosť = počet gaussiánov na nosiči), inak tá s najvyšším
pokrytím. Väčšie kandidátske inštancie sú vtedy „kontajnery“, nie súperi; `confidence` =
skóre víťaza / súčet skóre súperov vrátane víťaza a musí byť ≥ `min_confidence`.
Bez preferencie menšej pri rovnakom pokrytí vyhrá prvá inštancia v poradí, takže gauč
vankúš pohltí (overené testom).

**Fragmenty:** inštancie videné v menej ako `min_views` snímkach nehlasujú. Ich pokrytie
sa počíta z 1–2 snímok, je triviálne vysoké, a ako „menšie“ by vyhrali nad skutočným
objektom; potom ich vyradí filter a gaussiány ostanú nepriradené (na *room* takto jedna
maska zobrala 23 k gaussiánov podlahy, spolu sa stratilo 15 % gaussiánov). Po hlasovaní sa
masky fragmentu pripoja k inštancii, ktorá má aspoň `fragment_attach_share` jeho hlasov
(label, príznak, `source_masks`), inak ostanú bez inštancie.

### 4. Vyčistenie v 3D

- Odstránenie odľahlých gaussiánov inštancie (`outlier: components`): súvislé komponenty
  stredov spojených do vzdialenosti `outlier_k` × kvantil `outlier_quantile` vzdialeností
  k najbližšiemu susedovi (DBSCAN s `min_samples` = 1). Komponenty menšie ako
  `outlier_min_share` inštancie sa zahodia, najväčší ostáva. Kvantil namiesto mediánu, lebo
  hustota gaussiánov sa v rámci objektu veľmi mení a medián roztrhal riedke časti (test).
- Zahodiť inštancie s < `min_gaussians` (videné v < `min_views` pohľadoch sú vyradené už
  pred hlasovaním, pozri Fragmenty).

### 5. Label a príznak inštancie

- `label_votes`: vážené hlasovanie labelov zdrojových masiek (+ zero-shot skóre z 06 s váhou
  `zero_shot_weight`); váha masky = `Σ_i W_i(m)` × kvalita, normalizované, top 5.
- `instance_features`: vážený priemer embeddingov zdrojových masiek (váha = `Σ_i W_i(m)` × kvalita),
  v druhom prechode bez embeddingov s kosínusovou podobnosťou k prvému priemeru pod
  `feature_min_sim`, L2 normalizácia. Masky vynechané v 06 (`skipped`) sa nepočítajú.

Kvalita masky `q(m)` = `mask_score` × (`border_penalty`, ak sa dotýka okraja snímky).

## Implementácie

| Názov | Popis | Licencia |
|---|---|---|
| `gsplat_vote` (primárna) | postup vyššie, vlastný kód nad gsplat | vlastný kód + gsplat (Apache 2.0) |
| `pointcloud_project` (baseline) | pôvodný návrh: premietnuť **stredy** gaussiánov do snímok, zistiť, či padnú do masky (s kontrolou hĺbky z 02 `depth/`) — jednoduchšie, ignoruje okluziu a rozmer gaussiánov | vlastný kód |

Baseline je užitočný na porovnanie v práci (ukáže prínos váh namiesto bodovej projekcie).
Princíp training-free prenosu masiek na gaussiány je publikovaný napr. vo FlashSplat (ECCV 2024);
kód takých prác **nekopírovať**, ak stojí na INRIA rasterizéri — implementovať vlastný.
Hotové metódy nad INRIA kódom (Gaussian Grouping, LangSplat, Feature 3DGS …) sa smú spustiť
ako **nekomerčný** baseline na porovnanie (samostatné prostredie, pozri `docs/licenses.md`).

## Konfigurácia

```yaml
lifting:
  impl: gsplat_vote
  mask_batch: 32
  render_scale: 1.0
  min_weight: 0.05          # uchovať W_i(m) iba pri frac_i(m) ≥ tomuto
  support_fraction: 0.5
  border_penalty: 0.5
  merge_overlap: 0.3
  merge_sim: 0.75
  merge_rounds: 5
  duplicate_overlap: 0.7
  min_confidence: 0.5
  min_coverage: 0.2
  prefer_smaller_on_overlap: true
  outlier: components       # components | none
  outlier_k: 3.0
  outlier_quantile: 0.9
  outlier_min_share: 0.05
  min_gaussians: 200
  min_views: 3
  fragment_attach_share: 0.5
  zero_shot_weight: 0.5
  feature_min_sim: 0.6
  num_debug: 6
```

## Hardvér

GPU (gsplat). Pamäť riadená `mask_batch`; s cullingom a riedkym ukladaním zvládnuteľné v 6–8 GB.

## Validácia výstupu

- `instance_id.shape == (N,)` kde N = počet gaussiánov v PLY; hodnoty ∈ {−1} ∪ [0, K).
- Každá inštancia v `instances.json` má aspoň `min_gaussians` gaussiánov (overiť počítaním).
- `instance_features.npy` má K riadkov, normalizované.
- `embedding_model` = `meta.json.model` z fázy 06.

## Riziká a otvorené otázky

- Veľké gaussiány pozadia zasahujú do viacerých masiek → nízka confidence; zvážiť regularizáciu v 02 alebo strop mierky v 03.
- Objekty videné len z jednej strany → neúplné inštancie (iba viditeľná časť má gaussiány).
- Rozhodnutie časť vs. celok (hierarchia) nie je riešené; aktuálne pravidlo „menšia vyhráva“.
  Masky „kontajnerov“ (*floor* s kobercami, *entertainment center* s televízorom a policami)
  preto v metrike konzistencie vychádzajú zle (*room*: IoU 0,24 a 0,21), hoci je to zámer.
- Lesklé a odrazivé plochy (obrazovka, sklo, zrkadlo): 3DGS modeluje odlesky gaussiánmi za
  povrchom, ktoré sú v iných pohľadoch inde. Na *room* ležia gaussiány z masiek televízora
  v ostatných pohľadoch mimo nich (medián pokrytia 0,10) a rozptýlia sa do 11 iných inštancií
  alebo ostanú nepriradené (IoU televízora 0,21). Lifting to neopraví; pomôcť môže geometria
  z 02 (hĺbkový prior) alebo samostatné spracovanie odrazivých plôch.
- Kalibrácia prahov — robiť na jednej scéne s ručnou 3D anotáciou a zafixovať.

## Evaluácia

- mIoU renderovaných masiek inštancií voči 2D ground truth na testovacích pohľadoch
  (napr. dataset LERF-Mask alebo 3D-OVS, prípadne vlastné anotácie).
- Porovnanie `gsplat_vote` vs. `pointcloud_project`.
- Počet inštancií vs. ručný počet objektov (nadsegmentácia / podsegmentácia).

## Stav

Implementácia: `src/scenepipe/phases/p07_lifting/` (`common.py`, `gsplat_vote.py`),
test: `tests/contracts/test_p07.py` (falošné váhy: gauč s vankúšom, vankúš v dvoch
prerušených trackoch, pozadie, šum; gradientový trik na 3 gaussiánoch iba s CUDA, ~1 s).

- [x] načítanie vstupov (culling netreba, pozri Postup)
- [x] gradientový trik pre `W_i(m)` + test (súčet váh = alfa renderu v maske, zakrytie)
- [x] spájanie masiek do inštancií (s trackmi aj bez)
- [x] hlasovanie, prahy, vyčistenie
- [x] labely a príznaky inštancií
- [x] debug rendery (kód hotový, neoverené na reálnej scéne)
- [x] overenie na Mip-NeRF 360 *counter* (2026-11-06, GTX 1650 vo WSL, `fast`: 80 snímok,
  2 050 masiek, 169 trackov, 883 860 gaussiánov z 03, `render_scale` 0,5): váhy 61 s,
  celá fáza 85 s, špička VRAM 0,85 GB. Metrika `python -m scenepipe.eval.lifting_consistency`
  (inštancie vyrenderované späť do snímok, IoU s 2D maskami z 05, ktoré boli vstupom, teda
  vernosť liftingu a konzistencia masiek, nie kvalita voči ground truth):

  | Variant | Inštancie | Priradené gaussiány | Priemer IoU | Medián IoU | IoU > 0,5 |
  |---|---|---|---|---|---|
  | `gsplat_vote`, bez spájania duplikátov (prvá verzia) | 81 | 75 % | 0,545 | 0,715 | 59 % |
  | `pointcloud_project`, bez spájania duplikátov | 82 | 79 % | 0,541 | 0,681 | 59 % |
  | **`gsplat_vote` (default)** | 69 | 82 % | **0,708** | **0,822** | **80 %** |
  | `pointcloud_project` | 72 | 82 % | 0,663 | 0,761 | 78 % |

  Prvá verzia nechávala duplicitné a prerušené tracky ako samostatné kandidáty; tie prehrali
  hlasovanie, vypadli pre `min_gaussians` a ich masky ostali bez inštancie (valček IoU 0,11 →
  0,67 po oprave). Najhoršie labely po oprave: *juicer* 0,31, *can* 0,45, *bread* 0,52,
  *kitchen island* 0,53 (časť/celok s *counter*: doska ostrova patrí inštancii *counter*,
  pravidlo „menšia vyhráva“). Debug rendery: objekty konzistentne oddelené naprieč pohľadmi
  (kanvica, misa s ovocím, rastlina, valček, chňapka, plech).
- [x] *room* a *kitchen* a oprava fragmentov (2026-11-06, rovnaký stroj a profil; *room*
  100 snímok, 1 586 masiek, 405 k gaussiánov; *kitchen* 621 masiek, 8 inštancií, scéna
  zameraná na objekt). Priemer / medián IoU s 2D maskami, aktuálny kód (spájanie duplikátov
  aj fragmenty):

  | Scéna | `gsplat_vote` | `pointcloud_project` | Inštancie | Priradené gaussiány |
  |---|---|---|---|---|
  | *counter* | **0,712** / **0,823** | 0,664 / 0,761 | 70 | 82 % |
  | *room* | **0,587** / **0,700** | 0,567 / 0,658 | 52 | 80 % (pred opravou fragmentov 66 %, IoU 0,521 / 0,590) |
  | *kitchen* | **0,601** / **0,825** | 0,583 / 0,790 | 8 | 97 % |

  Najhoršie masky sú „kontajnery“ (*floor*, *table*, *entertainment center*) a televízor
  (pozri Riziká). Fáza 07 trvá 37–85 s, fáza 06 13–39 min podľa počtu masiek.
- [x] baseline `pointcloud_project` (`pointcloud_project.py`: projekcia stredov, hĺbka renderovaná z 03; spájanie a hlasovanie spoločné)
- [x] kontraktový test

## Zdroje

- Gu et al.: ConceptGraphs, ICRA 2024 (spájanie inštancií, agregácia príznakov).
- Shen et al.: FlashSplat: 2D to 3D Gaussian Splatting Segmentation Solved Optimally, ECCV 2024.
- Ye et al.: gsplat, 2024.
