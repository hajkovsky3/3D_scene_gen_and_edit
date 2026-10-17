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
- `report.json` — `num_gaussians`, `iterations`, `train_time_s`, `psnr`, `ssim`, `lpips`
  (priemer cez testovacie pohľady, `null` ak testovacie pohľady nie sú alebo je LPIPS vypnuté),
  `lpips_net`, `sh_degree`, `strategy`, `test_frame_ids`, `num_train_views`,
  `eval_resolution` (`[W, H]` po `resolution_scale`, metriky sa rátajú v tomto rozlíšení),
  `scene_scale`, `peak_vram_mb`, `gpu`, `per_view` (`frame_id`, `psnr`, `ssim`, `lpips`
  pre každý testovací pohľad), `warnings`.
- `test_renders/<frame_id>.jpg` — najviac `num_test_renders` testovacích pohľadov,
  vľavo render, vpravo referencia (na kontrolu očami).
- `manifest.json`.

Voliteľné (ak `export_depth: true`):
- `depth/<frame_id>.npy` — `float16[H,W]`, očakávaná hĺbka (z-depth v kamerovom rámci)
  pre každý tréningový pohľad, v **plnom rozlíšení snímky z `cameras.json`** (nezávisle od
  `resolution_scale`), 0 tam, kde sa nič nevyrenderovalo. Môže sa hodiť fázam 06, 07 a 11
  (napr. ako náhrada mono-depth).

## Postup

1. **Inicializácia**: gaussián pre každý bod z `points.npz` (poloha, farba z RGB → SH DC),
   izotropná mierka podľa vzdialenosti k najbližším susedom, opacity ≈ 0.1.
   Pri `mcmc` podľa odporúčania gsplat opacity 0.5 a mierka × 0.1; ak je bodov viac
   ako `cap_max`, náhodne sa vyberie `cap_max` bodov.
2. **Tréning**: pre náhodný tréningový pohľad vyrenderovať obraz (gsplat `rasterization`),
   loss = (1−λ)·L1 + λ·D-SSIM (λ = 0.2), Adam, ~30k iterácií.
3. **Adaptívna hustota** (densification): klonovanie a delenie gaussiánov s veľkým gradientom,
   prerezávanie priehľadných a príliš veľkých, periodický reset opacity. gsplat má na to
   pripravené stratégie (`DefaultStrategy`, `MCMCStrategy`).
   Plán densifikácie (začiatok, koniec, interval, reset opacity) sa škáluje podľa
   `iterations` / 30000, aby kratšie behy (testy, slabý HW) fungovali rozumne.
   Pri `mcmc` sa pridáva regularizácia 0.01 · mean(opacity) + 0.01 · mean(scale) (MCMC článok).
4. **Holdout**: pohľady s indexom 0, k, 2k, … (k = `test_every`, default 8) sú testovacie,
   nepoužívajú sa v tréningu. `test_every: 0` vypne holdout (všetky pohľady tréningové).
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

Oficiálna INRIA implementácia a `diff-gaussian-rasterization` (nekomerčná licencia) sa smú
použiť iba na **porovnanie** v práci (nekomerčné, samostatné prostredie, kód sa nekopíruje),
nie ako implementácia fázy. Primárna cesta rasterizuje výhradne cez gsplat.

## Konfigurácia

Rýchly profil `configs/fast.yaml` (vývoj ďalších fáz): `iterations: 7000`, `cap_max: 500000`.
Plán rozvrhu (densifikácia, learning rate) sa škáluje s počtom krokov.

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
  lpips: true              # LPIPS(alex) v reporte; stiahne váhy AlexNet z torchvision (~233 MB)
  num_test_renders: 8      # koľko testovacích renderov uložiť do test_renders/
  seed: 0
  compile_ssim: true       # SSIM v loss cez torch.compile (rovnaká hodnota, ~4× rýchlejšie)
  resolution_schedule: [[0.25, 0.25], [0.5, 0.5]]  # coarse-to-fine; [] = vypnuté
```

**Rýchlosť tréningu.** Na GTX 1650 (500k gaussiánov, 894×1590) zaberalo SSIM v loss
(2D konvolúcia 11×11, dopredu aj spätne) väčšinu kroku, rasterizácia gsplat len ~10 ms.
Preto:
- `compile_ssim`: SSIM sa počíta ako vážený súčet posunutých kópií (separabilné gaussovské
  okno, nulový padding, presne rovnaké ako konvolúcia) a kompiluje sa cez `torch.compile`
  (Triton je súčasťou PyTorch, nič sa nepridáva). Prvý krok trvá o ~10 s dlhšie (kompilácia).
  Ak kompilácia zlyhá, použije sa pôvodná funkcia. Metriky v reporte sa vždy počítajú
  pôvodnou funkciou `core.metrics.ssim`.
- Adam beží ako `fused` kernel (numericky rovnaký).
- Skúšaný `SelectiveAdam` z gsplat (aktualizuje iba viditeľné gaussiány, ~8 ms/krok menej)
  s MCMC na *izbe* vyrobil degenerovaný gaussián (mierka 481 pri scéne 134), validácia výstupu
  beh zastavila; preto sa nepoužíva.
- `resolution_schedule` (default `[[0.25, 0.25], [0.5, 0.5]]`): `[[podiel_krokov, mierka], ...]`, kroky pred
  `podiel × iterations` trénujú pri danej mierke rozlíšenia (referencia sa zmenší
  priemerovaním, intrinsika sa preškáluje), potom v plnom rozlíšení.

## Hardvér

Vyžaduje CUDA GPU. Pri 1–3 mil. gaussiánov a rozlíšení ~1600 px ~ 8–12 GB VRAM, s MCMC
a `resolution_scale: 0.5` sa dá zmestiť do ~4–6 GB. Čas: desiatky minút.

Prostredie `sp-gs` (pozri `envs/README.md`). Namerané na GTX 1650 (4 GB, WSL2), syntetická
miestnosť 160×120, 21 tréningových pohľadov, 5000 iterácií:

| Stratégia | Gaussiány | Čas | Špičková VRAM | PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|---|
| `mcmc` (`cap_max` 100k) | 100 000 | 107 s | 146 MB | 27,37 | 0,903 | 0,033 |
| `default` | 297 869 | 173 s | 462 MB | 25,73 | 0,865 | 0,044 |

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

- [x] načítanie kamier a bodov z kontraktu 01 (`core/cameras.py`)
- [x] tréningová slučka s gsplat (`phases/p02_reconstruction/gsplat.py`)
- [x] densification (default + mcmc)
- [x] export/import PLY + test konzistencie (`core/gaussians.py`, re-render v `validate_outputs`)
- [x] report s metrikami, testovacie rendery (`core/metrics.py`)
- [x] voliteľný export hĺbky
- [x] kontraktový test (`tests/contracts/test_p02.py`, syntetická miestnosť, preskočí sa bez CUDA)
- [x] overenie na reálnej scéne Mip-NeRF 360 *room* (2026-10-15, GTX 1650): `mcmc`, 1 M gaussiánov,
  30k krokov, 1556×1037 (`images_2`), 272 train / 39 test (každý 8.) → PSNR 32.94, SSIM 0.948,
  LPIPS(alex) 0.116; 7 020 s tréning, špička VRAM 1,6 GB
  (3DGS v článku: 30.63 / 0.914 / 0.220, ale LPIPS s VGG)
- [x] overenie na vlastnej scéne *izba* (2026-10-16, video z telefónu, 186 kamier z 01 s `exhaustive`):
  `mcmc`, 1 M gaussiánov, 30k krokov, 894×1590, 162 train / 24 test → PSNR 26.82, SSIM 0.895,
  LPIPS(alex) 0.256; 6 257 s tréning, špička VRAM 1,6 GB. Najhorší testovací pohľad f000097
  (PSNR 12.8) je rozmazaný záber zblízka.
- [x] rýchly profil `configs/fast.yaml` na *izba* (194 kamier z 01 `retrieval`, iná testovacia
  sada než vyššie): 7k krokov, 500k gaussiánov → PSNR 23.96, SSIM 0.870, LPIPS(alex) 0.336;
  1 055 s tréning (6× rýchlejšie), špička VRAM 0,85 GB
- [x] zrýchlenie tréningu (2026-10-17, GTX 1650 vo WSL, `fast`: 7k krokov, 500k gaussiánov).
  Profil kroku pred úpravou (*izba*, 894×1590): 133 ms, z toho SSIM dopredu 33 ms, backward
  (render + SSIM) 53 ms, Adam 25 ms, render 10 ms. Výsledky (GPU počas meraní zdieľali aj iné
  aplikácie, časy porovnateľné len v rámci jednej scény):

  | Scéna | Variant | Tréning | PSNR | SSIM | LPIPS |
  |---|---|---|---|---|---|
  | *izba* (189 train) | pôvodné SSIM, Adam | 1 649 s | 23,26 | 0,869 | 0,367 |
  | *izba* | `compile_ssim` + fused Adam | 1 304 s | 23,13 | 0,867 | 0,371 |
  | *izba* | + `resolution_schedule` (default) | **901 s** | 23,19 | 0,870 | 0,365 |
  | counter (210 train) | `compile_ssim` + fused Adam | 1 025 s | 27,90 | 0,908 | 0,165 |
  | counter | + `resolution_schedule` (default) | **819 s** | 27,84 | 0,906 | 0,163 |

  Rovnaká konfigurácia na *izbe* dala raz PSNR 22,83 a raz 23,26: tréning na GPU nie je
  deterministický a rozdiely do ~0,4 dB sú šum. Spolu −45 % času na *izbe*, −20 % na counter
  (menšie snímky, menší podiel ceny za pixel) bez merateľnej straty kvality.
- [ ] overenie na *counter* a *kitchen* s plným tréningom (30k), porovnanie s 3DGS s LPIPS(vgg)
- [ ] orezanie floaterov mimo objemu kamier (pozri Riziká)
- [ ] test kompatibility PLY so SuperSplat

## Zdroje

- Kerbl et al.: 3D Gaussian Splatting for Real-Time Radiance Field Rendering, SIGGRAPH 2023
  (citovať ako metódu; kód len na porovnanie, nekomerčná licencia).
- Ye et al.: gsplat: An Open-Source Library for Gaussian Splatting, 2024. https://github.com/nerfstudio-project/gsplat
- Kheradmand et al.: 3D Gaussian Splatting as Markov Chain Monte Carlo, NeurIPS 2024.
