# Fáza 02: Rekonštrukcia (tréning 3D Gaussian Splatting)

`contract_version: 1`

## Účel

Natrénovať fotorealistickú reprezentáciu scény ako množinu 3D gaussiánov. Je to **geometrická
vrstva**, nad ktorou pracuje lifting (06), graf (07), úprava (11) aj renderovanie (12).
V pôvodnom návrhu to bola posledná fáza; je presunutá sem, lebo všetko ostatné ju potrebuje.

## Kontrakt

**Vstup**
- `01_sfm/cameras.json`, `01_sfm/images_undistorted/`, `01_sfm/points.npz`
  (alebo priamo `01_sfm/sparse/` pre implementácie, ktoré čítajú COLMAP)

**Výstup** (`02_reconstruction/`)
- `gaussians.ply` — formát podľa `docs/data-contracts.md`, sekcia 5, vo svetových súradniciach z 01.
- `report.json` — počet gaussiánov, počet iterácií, čas, PSNR/SSIM/LPIPS na testovacích
  pohľadoch, `sh_degree`, zoznam testovacích `frame_id`.
- `test_renders/` — niekoľko renderov testovacích pohľadov vedľa referencií (na kontrolu očami).
- `manifest.json`.

Voliteľné (ak `export_depth: true`):
- `depth/<frame_id>.npy` — `float16[H,W]`, očakávaná hĺbka (z-depth v kamerovom rámci)
  pre každý tréningový pohľad. Môže sa hodiť fázam 06, 07 a 11 (napr. ako náhrada mono-depth).

## Postup

1. **Inicializácia**: gaussián pre každý bod z `points.npz` (poloha, farba z RGB → SH DC),
   izotropná mierka podľa vzdialenosti k najbližším susedom, opacity ≈ 0.1.
2. **Tréning**: pre náhodný tréningový pohľad vyrenderovať obraz (gsplat `rasterization`),
   loss = (1−λ)·L1 + λ·D-SSIM (λ = 0.2), Adam, ~30k iterácií.
3. **Adaptívna hustota** (densification): klonovanie a delenie gaussiánov s veľkým gradientom,
   prerezávanie priehľadných a príliš veľkých, periodický reset opacity. gsplat má na to
   pripravené stratégie (`DefaultStrategy`, `MCMCStrategy`).
4. **Holdout**: každý k-ty pohľad (default 8) ako testovací — nepoužíva sa v tréningu.
5. **Export** do PLY v projektovom formáte (log-scale, logit-opacity, wxyz kvaternión).

Odporúčania pre neskoršie fázy:
- **MCMC stratégia** s pevným maximom gaussiánov dáva predvídateľnú pamäť — dobré pre slabší HW.
- Regularizácia proti „floaterom“ (opacity, scale) zlepší lifting a úpravy, aj keď mierne
  zníži PSNR. Je to vedomý kompromis — zaznamenať v práci.
- `antialiased` režim gsplat znižuje artefakty pri renderoch z iných vzdialeností.

## Implementácie

| Názov | Popis | Licencia |
|---|---|---|
| `gsplat` (primárna) | vlastný tréningový skript nad knižnicou gsplat (alebo upravený `examples/simple_trainer.py`) | Apache 2.0 |
| `splatfacto` (alternatíva) | nerfstudio metóda nad gsplat, viac hotovej infraštruktúry, ťažšia závislosť | Apache 2.0 |

**Nepoužívať**: oficiálna INRIA implementácia a `diff-gaussian-rasterization` (nekomerčná
licencia), ani metódy postavené na nich (LangSplat, Feature 3DGS, Gaussian Grouping …),
pokiaľ nie sú prenesené na gsplat s kompatibilnou licenciou.

## Konfigurácia

```yaml
reconstruction:
  impl: gsplat
  iterations: 30000
  sh_degree: 3
  strategy: mcmc           # default | mcmc
  cap_max: 1000000         # max gaussiánov pri mcmc
  resolution_scale: 1.0    # 0.5 pre slabší HW
  test_every: 8
  antialiased: true
  export_depth: false
  seed: 0
```

## Hardvér

Vyžaduje CUDA GPU. Pri 1–3 mil. gaussiánov a rozlíšení ~1600 px ~ 8–12 GB VRAM, s MCMC
a `resolution_scale: 0.5` sa dá zmestiť do ~4–6 GB. Čas: desiatky minút.

## Validácia výstupu

- PLY sa dá načítať, obsahuje presne vlastnosti zo sekcie 5 v správnom poradí, N > 0, bez NaN/Inf.
- `rot` nenulové kvaternióny, `scale` v rozumnom rozsahu (napr. exp(scale) < veľkosť scény).
- Render testovacieho pohľadu z uloženého PLY (cez `core`) dáva PSNR v tolerancii voči
  hodnote v reporte — overí, že export a import sú konzistentné (konvencie kvaterniónu, SH).

## Riziká a otvorené otázky

- Floatery a „plávajúce“ gaussiány pri okrajoch scény → zhoršia lifting aj bounding boxy v 07.
  Možné riešenie: orezanie gaussiánov mimo objemu pokrytého kamerami.
- Veľké gaussiány pozadia pretínajú viac objektov → nejednoznačné priradenie v 06.
  Zvážiť obmedzenie maximálnej mierky.
- Kompatibilita formátu SH (poradie `f_rest`) s externými viewermi — otestovať v SuperSplat.

## Evaluácia

PSNR, SSIM, LPIPS na testovacích pohľadoch; počet gaussiánov; čas tréningu; špičková VRAM.
Porovnanie na Mip-NeRF 360 scénach (room, counter, kitchen) s publikovanými číslami 3DGS.

## Stav

- [ ] načítanie kamier a bodov z kontraktu 01
- [ ] tréningová slučka s gsplat
- [ ] densification (default + mcmc)
- [ ] export/import PLY + test konzistencie
- [ ] report s metrikami, testovacie rendery
- [ ] voliteľný export hĺbky
- [ ] kontraktový test

## Zdroje

- Kerbl et al.: 3D Gaussian Splatting for Real-Time Radiance Field Rendering, SIGGRAPH 2023
  (citovať ako metódu, kód nepoužívať).
- Ye et al.: gsplat: An Open-Source Library for Gaussian Splatting, 2024. https://github.com/nerfstudio-project/gsplat
- Kheradmand et al.: 3D Gaussian Splatting as Markov Chain Monte Carlo, NeurIPS 2024.
