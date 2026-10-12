# Fáza 12: Renderovanie a prehliadanie

`contract_version: 1`

## Účel

Zobraziť ľubovoľný stav scény: rendery z daných kamier, video po trajektórii, porovnanie
pred/po úprave a interaktívny viewer v prehliadači. Fáza iba číta, nič nemení, takže sa dá
spustiť nad ktorýmkoľvek stavom kedykoľvek (aj po fáze 02, ešte pred existenciou grafu).

## Kontrakt

**Vstup**
- `state/vNNN/state.json` alebo priamo `gaussians.ply` (+ voliteľne `gaussian_instances.npz`
  a `scene_graph.json` pre sémantické vrstvy)
- kamery: `01_sfm/cameras.json` (tréningové/testovacie pohľady) alebo `camera_path.json`
  (vlastná trajektória: zoznam `world_to_camera` + intrinsika)

**Výstup** (`renders/<názov_behu>/`)
- `frames/<idx>.png` — RGB rendery.
- voliteľne `depth/<idx>.npy`, `instances/<idx>.png` (farebná mapa inštancií),
  `overlay/<idx>.png` (RGB + OBB a labely uzlov).
- `video.mp4` — pri trajektórii.
- `compare/<idx>.png` — vedľa seba dva stavy (pri `compare_with`).
- `manifest.json`.

Viewer (`impl: viser`) nevytvára súbory, spúšťa lokálny server.

## Postup

1. Načítať gaussiány (formát sekcie 5) a kamery.
2. Render cez gsplat `rasterization` (RGB, voliteľne `RGB+ED` hĺbka); pre mapu inštancií
   renderovať one-hot / farbu podľa `instance_id` ako príznak.
3. Trajektória: interpolácia medzi kľúčovými kamerami (SLERP rotácií, spline polôh) alebo
   automatický orbit okolo stredu scény.
4. Video cez ffmpeg.
5. **Viewer**: viser server s gsplat renderom podľa aktuálnej kamery; panel so zoznamom uzlov
   grafu (klik = zvýrazniť gaussiány uzla, zobraziť OBB a caption), prepínač stavov `vNNN`,
   textové pole na prompt (spustí edit loop) — neskoršie rozšírenie.
6. **Export** pre externé viewery: PLY je už kompatibilný (SuperSplat v prehliadači);
   voliteľne kompresia (`.spz`, `.splat`) na zdieľanie.

## Implementácie

| Názov | Popis | Licencia |
|---|---|---|
| `gsplat_offline` (primárna) | rendery a video | Apache 2.0 |
| `viser` | interaktívny webový viewer (nerfstudio ho tiež používa) | Apache 2.0 |
| externé | SuperSplat (prehliadač, editor) na rýchlu kontrolu PLY | MIT |

## Konfigurácia

```yaml
rendering:
  impl: gsplat_offline
  cameras: test                # train | test | all | path:<súbor> | orbit
  orbit_frames: 120
  resolution_scale: 1.0
  outputs: [rgb]               # rgb, depth, instances, overlay
  compare_with: null           # napr. v000
  fps: 30
  background: [0, 0, 0]
```

## Hardvér

GPU pre gsplat. Render jedného snímku ~ milisekundy, viewer v reálnom čase na bežnej GPU.

## Validácia výstupu

- Počet snímok = počet kamier; rozmery = rozmery kamier × `resolution_scale`.
- Žiadne úplne prázdne (čierne) snímky pri tréningových kamerách.
- Konzistencia: render tréningového pohľadu zo stavu `v000` dáva PSNR ako v reporte fázy 02.

## Riziká a otvorené otázky

- Rozdiel medzi renderom v našom kóde a v externom vieweri (konvencie SH, gamma, pozadie) →
  porovnať na jednom pohľade pri prvej implementácii.
- Viewer na vzdialenom GPU stroji → port forwarding (SSH tunel).

## Evaluácia

Táto fáza poskytuje rendery pre metriky ostatných fáz (PSNR/SSIM/LPIPS v 02 a 11,
mIoU inštancií v 06). Spoločný kód metrík patrí do `src/scenepipe/core/metrics.py`.

## Stav

- [ ] render z `cameras.json`
- [ ] orbit trajektória + video
- [ ] mapa inštancií, overlay OBB
- [ ] porovnanie dvoch stavov
- [ ] viser viewer (základ), panel grafu
- [ ] kontraktový test
