# Fáza 12: Úprava scény

`contract_version: 1`

## Účel

Vykonať plán úprav nad stavom scény `vN` a vytvoriť nový stav `vN+1`: upravené gaussiány,
aktualizované priradenie gaussiánov k inštanciám a aktualizovaný graf. Toto je úzke hrdlo
celého reťazca a hlavný priestor na vlastný prínos práce.

Namiesto jedného univerzálneho editora (GaussianEditor) je fáza postavená ako **router
operácií**: každá operácia má vlastný operátor s najjednoduchším mechanizmom, ktorý stačí.
Operátory zdieľajú tri služby (inpainting, dotréning, umiestnenie), ktoré sú tiež vymeniteľné.

## Kontrakt

**Vstup**
- `state/<base_state>/state.json` (gaussiány, inštancie, graf, príznaky uzlov, kamery)
- `edits/eNNN/10_grounding/grounded_plan.json`
- `edits/eNNN/11_asset_generation/<op_id>/asset.ply` + `asset_meta.json` (pre `replace`/`insert`)
- `01_sfm/images_undistorted/` (pre dotréning a inpainting)

**Výstup** (`edits/eNNN/12_editing/`)
- `gaussians.ply` — nový stav gaussiánov.
- `gaussian_instances.npz` — rovnaké polia ako v 07, **zarovnané s novým poradím gaussiánov**.
- `scene_graph.json`, `node_features.npy` — aktualizovaný graf.
- `edit_report.json`:
  ```json
  {"contract_version": 1,
   "ops": [{"op_id": "op01", "op": "replace", "status": "done",
            "removed_gaussians": 41233, "added_gaussians": 118200,
            "nodes_removed": ["n0000"], "nodes_added": ["n0012"],
            "refine_iters": 1500, "duration_s": 312.4, "warnings": []}],
   "new_state": "v001"}
  ```
- `before_after/` — rendery pred a po z niekoľkých pohľadov.
- `views/` — (ak sa robil dotréning) upravené cieľové snímky použité ako „pseudo ground truth“.
- `state/v<N+1>/state.json` — nový stav (jediný zápis mimo vlastného adresára).
- `manifest.json`.

Operácie sa vykonávajú postupne v poradí plánu; každá dostane výsledok predchádzajúcej.
Operácia so `status != ok` z groundingu sa preskočí (`status: "skipped"` s dôvodom).

## Operátory

Každý operátor je trieda s rozhraním `apply(state, op, services) -> (state, report)`.
Pridanie novej operácie = nový operátor v registri, bez zmeny zvyšku.

### `recolor` — zmena farby
1. Vybrať gaussiány uzla (`instance_id`).
2. Previesť SH DC na RGB, posunúť **odtieň a sýtosť** k cieľovej farbe v priestore HSV/Lab,
   zachovať jas (tiene a textúra ostanú). `strength` určuje mieru.
3. Vyššie SH koeficienty (pohľadovo závislé odlesky) škálovať alebo vynulovať pre konzistenciu.
4. Žiadny difúzny model, okamžité. Voliteľne krátky dotréning nie je potrebný.

### `delete` — odstránenie
1. Odstrániť gaussiány uzla (a uzol + jeho hrany z grafu; objekty, ktoré boli `on` mazanom
   objekte, označiť varovaním — „visia vo vzduchu“).
2. **Inpainting diery**: pre tréningové pohľady, kde bol objekt viditeľný, vyrenderovať scénu
   bez objektu + masku diery (render inštancie z pôvodného stavu, mierne dilatovaná) →
   2D inpainting → upravené cieľové snímky.
3. **Dotréning**: pridať gaussiány do oblasti diery (inicializácia z okolitých gaussiánov alebo
   z hĺbky), trénovať iba ich (ostatné zmrazené) na upravených snímkach s loss len v maske.
4. Nové gaussiány dostanú `instance_id` podložky / pozadia (napr. podlahy), nie nový objekt.

### `translate`, `rotate`, `scale` — transformácia
1. Vypočítať transformáciu: z `params` a pri `place_on` z grafu — objekt položiť na hornú
   plochu podložky (z_min objektu = z_max podložky), zachovať pozíciu v XY, ak je v rámci
   podložky, inak stred podložky; kontrola kolízie OBB s inými objektmi.
2. Aplikovať rigidnú transformáciu na gaussiány uzla: stredy, rotácie (kvaternióny),
   mierky (log-scale + log(factor)); **SH koeficienty vyšších stupňov otočiť** (pri rotácii
   okolo Z je rotácia SH jednoduchá; alternatívne vynulovať f_rest a zaznamenať).
3. Objekty, ktoré sú `on` presúvanom objekte, presunúť spolu s ním (rekurzívne cez graf).
4. Inpainting pôvodného miesta ako pri `delete` (iba oblasť, ktorú objekt už nezakrýva).
5. Aktualizovať OBB uzla a prepočítať geometrické hrany dotknutých uzlov (logika ako v 08,
   zdieľaná cez `core/graph.py`).

### `replace` — výmena
1. Uložiť stopu (footprint) pôvodného objektu: OBB, podložka, natočenie.
2. `delete` pôvodného objektu (vrátane inpaintingu).
3. Asset z 11 umiestniť: natočenie podľa hlavnej osi pôvodného OBB, mierka tak, aby sa zmestil
   do pôvodnej stopy v XY (pri `keep_footprint`) s pomerom strán assetu, z_min na podložku.
4. Pridať gaussiány assetu s novým `instance_id`, nový uzol (label z `asset_prompt`,
   príznak = CLIP embedding `condition.png`), prepočítať hrany.
5. Voliteľne **harmonizácia**: krátky dotréning farieb (len SH DC) nových gaussiánov
   na niekoľkých pohľadoch so zmrazenou geometriou — otvorený problém, experimentovať.

### `insert` — vloženie
Ako `replace` bez mazania: miesto = podložka z groundingu, poloha v XY = voľné miesto
na podložke (najväčší obdĺžnik bez kolízie, alebo `near` objektu), mierka = typická veľkosť
kategórie relatívne k podložke (heuristika v konfigurácii, napr. pohár ≈ 0,1 × výška stola).

### `style` — štýl celej scény alebo objektu
1. Iteratívna úprava datasetu (princíp Instruct-NeRF2NeRF): vyrenderovať pohľad → 2D
   instruction-based editor upraví obrázok → nahradí cieľovú snímku → dotréning gaussiánov
   (pri objekte len jeho gaussiány a loss v maske) → opakovať.
2. Najdrahšia a najmenej stabilná operácia; v práci ako voliteľné rozšírenie.

## Zdieľané služby

| Služba | Rozhranie | Primárna implementácia | Licencia |
|---|---|---|---|
| `inpainter` | `(image, mask) -> image` | LaMa (big-lama) | Apache 2.0 |
| `refiner` | `(gaussians, trainable_mask, target_views, loss_masks, iters) -> gaussians` | gsplat, rovnaká loss ako 02 | Apache 2.0 |
| `placer` | `(asset_aabb, support_node, footprint, graph) -> transform` | vlastný kód | — |
| `image_editor` (len `style`) | `(image, instruction) -> image` | otvorený instruction-based model — **vybrať podľa licencie** (InstructPix2Pix váhy sú OpenRAIL-M = hraničné) | — |

Alternatívy inpainteru na neskôr: difúzny inpainting (lepší pre veľké diery, horšia konzistencia
medzi pohľadmi). **Multi-view konzistencia inpaintingu** je známy otvorený problém — LaMa
na každý pohľad zvlášť vytvára nekonzistentné výplne; dotréning ich priemeruje (rozmazanie).
Možné zlepšenie: inpaintovať len jeden referenčný pohľad a do ostatných ho preniesť cez
hĺbku (reprojekcia), zvyšok dotrénovať — dobrý kandidát na vlastný experiment.

## Postup (router)

1. Načítať stav, plán, assety.
2. Pre každú operáciu: operátor z registra → `apply` → priebežný stav v pamäti.
3. Po poslednej operácii: preindexovať `instance_id` k novému poradiu gaussiánov,
   validovať, uložiť súbory, before/after rendery, nový `state.json`.

## Implementácie

| Názov | Popis | Licencia |
|---|---|---|
| `router` (primárna) | operátory a služby vyššie | vlastný kód + gsplat, LaMa |

GaussianEditor (S-Lab, nekomerčná) a Instruct-NeRF2NeRF sa v práci citujú ako inšpirácia.
GaussianEditor sa smie spustiť ako **nekomerčný** baseline na porovnanie (samostatné prostredie,
kód sa nekopíruje ani nevolá z routera), pozri `docs/licenses.md`.

## Konfigurácia

```yaml
editing:
  impl: router
  inpainter: lama
  hole_dilate_px: 15
  inpaint_views: auto          # auto = pohľady, kde je diera > min_hole_frac
  min_hole_frac: 0.002
  refine_iters: 1500
  refine_lr_scale: 0.5
  recolor_keep_luminance: true
  rotate_sh: true
  harmonize_asset: false
  insert_scale_heuristics: configs/category_sizes.yaml
  max_ops: 5
```

## Hardvér

recolor, transform bez inpaintingu: sekundy, málo VRAM. delete/replace s dotréningom:
minúty, podobne ako 02 (≥ 6 GB). style: desiatky minút, + VRAM 2D editora.

## Validácia výstupu

- Nový PLY a `gaussian_instances.npz` majú rovnaké N; ID existujúcich objektov sa nezmenili
  (okrem zmazaných); nové ID > všetky staré.
- Graf: odstránené uzly nemajú hrany, nové uzly majú OBB, hrany odkazujú na existujúce uzly.
- `state.json` nového stavu ukazuje na existujúce súbory, `parent` = `base_state`.
- Pre `recolor`/`transform`: gaussiány mimo cieľa sú **bitovo nezmenené** (test izolácie úpravy).

## Riziká a otvorené otázky

- Inpainting: nekonzistencia medzi pohľadmi, tiene objektu ostanú na podlahe po zmazaní.
- Tiene a odrazy nie sú súčasťou inštancie → po presune ostanú na pôvodnom mieste.
- Harmonizácia vloženého objektu (osvetlenie, farebná teplota).
- Neúplné objekty (videné z jednej strany) → po otočení vidno „prázdnu“ zadnú stranu.
- Mierka pri `insert` bez metrickej scény je heuristická.

## Evaluácia

- **Izolácia úpravy**: PSNR/LPIPS mimo zasiahnutej oblasti pred vs. po (má byť ~nezmenené).
- **Splnenie promptu**: CLIP directional similarity (zmena obrazu vs. zmena textu).
- **Kvalita inpaintingu**: na scénach, kde existujú zábery aj bez objektu (vlastné nafotenie
  „pred a po“ fyzickom odstránení objektu) — PSNR/LPIPS voči skutočnosti. Silný experiment do práce.
- Čas na operáciu.
- Kvalitatívne porovnanie s GaussianEditorom na ich ukážkových scénach (len výsledky z článku).

## Stav

- [ ] router, register operátorov, zápis stavu
- [ ] `recolor`
- [ ] `delete` bez inpaintingu → s LaMa + dotréning
- [ ] `translate`/`rotate`/`scale` + presun závislých objektov + rotácia SH
- [ ] `replace` a `insert` s `primitive` assetom, potom s TRELLIS
- [ ] (voliteľné) harmonizácia
- [ ] (voliteľné) `style`
- [ ] testy izolácie a kontraktový test

## Zdroje

- Chen et al.: GaussianEditor: Swift and Controllable 3D Editing with Gaussian Splatting, CVPR 2024.
- Haque et al.: Instruct-NeRF2NeRF, ICCV 2023.
- Suvorov et al.: Resolution-robust Large Mask Inpainting with Fourier Convolutions (LaMa), WACV 2022.
