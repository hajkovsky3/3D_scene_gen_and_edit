# Fáza 10: Generovanie nového objektu (asset)

`contract_version: 1`

## Účel

Pre operácie `replace` a `insert` vytvoriť nový 3D objekt z textového alebo obrázkového
promptu, ako množinu gaussiánov v **kanonickom tvare** (`docs/data-contracts.md`, sekcia 4).
Umiestnenie do scény (poloha, mierka, natočenie) už robí fáza 11 — vďaka tomu je generovanie
nezávislé od scény a dá sa testovať samostatne.

Fáza sa spúšťa len vtedy, ak plán obsahuje `replace` alebo `insert`.

## Kontrakt

**Vstup**
- `edits/eNNN/09_grounding/grounded_plan.json` — operácie `replace`/`insert` s `asset_prompt`
  a/alebo `asset_image`
- voliteľne `state/<base_state>/…` + `01_sfm/images_undistorted/` — pri `replace` výrez
  pôvodného objektu ako kontext (napr. aby nový stôl mal podobný uhol pohľadu / osvetlenie)

**Výstup** (`edits/eNNN/10_asset_generation/<op_id>/`)
- `asset.ply` — gaussiány objektu vo formáte zo sekcie 5, v kanonickom tvare.
- `asset_meta.json`:
  ```json
  {"contract_version": 1, "op_id": "op01",
   "source": "text", "asset_prompt": "blue table",
   "condition_image": "condition.png",
   "canonical": {"up": "+z", "front": "-y", "origin": "bottom_center", "max_extent": 1.0},
   "aabb": {"min": [-0.5, -0.3, 0.0], "max": [0.5, 0.3, 0.62]},
   "num_gaussians": 120000, "generator": "trellis", "seed": 0}
  ```
- `condition.png` — obrázok, z ktorého vznikol 3D objekt (vygenerovaný alebo dodaný, po odstránení pozadia).
- `preview.mp4` alebo `preview_*.png` — otočný náhľad.
- `manifest.json` (jeden pre celú fázu, zoznam `op_id`).

## Postup

1. **Podmieňujúci obrázok**
   - ak je `asset_image` → použiť ho;
   - inak text → obrázok: prompt rozšíriť na „single object, centered, plain white background,
     full view, studio lighting“ (+ pri `replace` voliteľne štýl pôvodnej scény).
2. **Odstránenie pozadia** (segmentácia popredia) → RGBA.
3. **Obrázok → 3D**:
   - TRELLIS (v1) vie dekódovať priamo **3D gaussiány** → najjednoduchšia cesta;
   - TRELLIS.2 dáva mesh s PBR materiálmi → potrebný prevod mesh → gaussiány
     (vyrenderovať mesh z ~100 pohľadov a krátko nafitovať gaussiány cez gsplat, alebo
     vzorkovať gaussiány na povrchu).
4. **Kanonizácia**: preorientovať osi generátora na +Z hore, posunúť stred spodnej podstavy
   do počiatku, normalizovať najdlhší rozmer na 1.
5. **Prevod do formátu sekcie 5** (konvencie SH, log-scale, logit-opacity, wxyz).
6. Náhľad.

## Implementácie

| Názov | Popis | Licencia |
|---|---|---|
| `flux_trellis` (primárna) | FLUX.1-schnell (text → obrázok) + BiRefNet (pozadie) + TRELLIS image-large (→ gaussiány) | Apache 2.0 / MIT / MIT |
| `trellis2_mesh` (alternatíva) | ako vyššie, ale TRELLIS.2-4B (lepšia geometria a materiály) + mesh → gaussiány | MIT |
| `image_only` | bez text → obrázok, iba dodaný obrázok (testovanie, slabší HW) | MIT |
| `primitive` (baseline) | parametrická krabica / valec požadovanej farby ako gaussiány — overí celý reťazec úprav bez generatívnych modelov | vlastný kód |

**Licenčné pozor**:
- Hunyuan3D nepoužívať (licencia vylučuje EÚ, `ZAKÁZANÉ`).
- FLUX.1-dev, SDXL-Turbo, SD 3.x (nekomerčné / vlastné licencie) iba ako **nekomerčná**
  alternatíva text → obrázok, nie v primárnej implementácii.
- TRELLIS ťahá závislosti s vlastnými licenciami (napr. `nvdiffrast` — NVIDIA, nekomerčná;
  niektoré verzie gaussian rasterizérov odvodené od INRIA). Pred použitím prejsť `requirements`
  a zistiť, ktoré sú nutné len na renderovanie náhľadov — tie nahradiť gsplat. Ak nekomerčná
  závislosť ostane nutná, `flux_trellis` / `trellis2_mesh` sa označia ako **nekomerčné**
  alternatívy a primárnou implementáciou v `configs/default.yaml` sa stane `primitive`.
  Zapísať do `docs/licenses.md`.

## Konfigurácia

```yaml
asset_generation:
  impl: flux_trellis
  t2i_model: black-forest-labs/FLUX.1-schnell
  t2i_steps: 4
  t2i_offload: true            # CPU offload pri malej VRAM
  bg_removal: birefnet
  i23d_model: microsoft/TRELLIS-image-large
  seed: 0
  num_candidates: 1            # >1 → vygenerovať viac a vybrať podľa CLIP zhody s promptom
```

## Hardvér

Najnáročnejšia fáza. FLUX.1-schnell (12B) ~ 24 GB VRAM, s offloadom/kvantizáciou menej, ale pomalé.
TRELLIS image-large ~ 16 GB, TRELLIS.2 ~ 24 GB. Na slabom HW používať `image_only` alebo `primitive`
a generovanie spúšťať na výkonnejšom stroji (fáza je izolovaná, stačí prenos súborov).

## Validácia výstupu

- `asset.ply` podľa sekcie 5, N > 0, bez NaN.
- Kanonický tvar: `aabb.min.z ≈ 0`, stred podstavy ≈ (0, 0), max rozmer ≈ 1 (tolerancia 1 %).
- `asset_meta.op_id` zodpovedá operácii v pláne.

## Riziká a otvorené otázky

- **Osvetlenie a štýl**: vygenerovaný objekt má „štúdiové“ svetlo, scéna reálne → objekt pôsobí
  vlepene. Harmonizácia (úprava farieb podľa okolia, krátky dotréning v 11) je otvorený problém.
- Orientácia „prednej strany“ sa z generátora nedá spoľahlivo určiť → pri `replace` zarovnať
  podľa hlavnej osi OBB pôvodného objektu.
- Generatívne modely sú nedeterministické — fixovať `seed`, ukladať `condition.png`.

## Evaluácia

CLIP skóre zhody náhľadov s promptom; vizuálna kontrola; čas a VRAM.

## Stav

- [ ] baseline `primitive`
- [ ] `image_only` (pozadie + TRELLIS → gaussiány)
- [ ] text → obrázok (FLUX.1-schnell)
- [ ] kanonizácia + prevod formátu
- [ ] licenčný audit závislostí TRELLIS
- [ ] kontraktový test

## Zdroje

- Xiang et al.: Structured 3D Latents for Scalable and Versatile 3D Generation (TRELLIS), CVPR 2025.
- Microsoft: TRELLIS.2 — Native and Compact Structured Latents for 3D Generation, 2025.
- Black Forest Labs: FLUX.1 [schnell], 2024.
- Zheng et al.: Bilateral Reference for High-Resolution Dichotomous Image Segmentation (BiRefNet), 2024.
