# Fáza 03: Refinement (vyčistenie gaussiánov)

`contract_version: 1`

## Účel

Vyčistiť natrénované gaussiány z fázy 02 skôr, než nad nimi začne pracovať sémantická vrstva.
Tréning 3DGS nechá v scéne artefakty, ktoré v tréningových pohľadoch nevadia, ale z iných uhlov
sú viditeľné a kazia ďalšie fázy:

- **takmer priehľadný „opar“**: veľa gaussiánov s veľmi nízkou opacity, ktoré sa v tréningových
  pohľadoch spriemerujú, ale z nového uhla vyzerajú ako šmuhy,
- **floatery**: gaussiány mimo objemu pokrytého kamerami (pri okrajoch, za kamerami),
- **obrovské gaussiány** pozadia, ktoré pretínajú viac objektov (nejednoznačný lifting v 07),
- **ihlicovité gaussiány** („hviezdičky“): veľmi pretiahnuté, z iného uhla trčia ako lúče.

Floatery zhoršujú lifting (07) aj bounding boxy v grafe (08). Preto sa čistí tu, v samostatnej
fáze, a nie až pri úprave. Fáza sa dá vypnúť (`impl: none`) a porovnať pred a po.

Prevencia počas tréningu (regularizácia v loss) patrí do fázy 02. Táto fáza pracuje iba
s hotovými gaussiánmi.

## Kontrakt

**Vstup**
- `02_reconstruction/gaussians.ply`
- `02_reconstruction/report.json` (`test_frame_ids`, `eval_resolution`, `sh_degree`: metriky na tých istých pohľadoch)
- `02_reconstruction/manifest.json` (`config.antialiased`: render rovnakým režimom ako pri tréningu)
- `01_sfm/cameras.json`, `01_sfm/images_undistorted/`
- `01_sfm/points.npz` (ohraničenie scény, iba pri `crop: true`)

**Výstup** (`03_refinement/`)
- `gaussians.ply` — formát podľa `docs/data-contracts.md`, sekcia 5, v tých istých svetových
  súradniciach ako vstup. Poradie ponechaných gaussiánov sa zachová.
- `report.json` — `num_input`, `num_output`, `removed` (počet odstránených podľa kritéria:
  `crop`, `opacity`, `views`, `scale`, `aniso`; jeden gaussián sa započíta pri prvom kritériu, ktoré ho
  vyradí, v tomto poradí), `clamped` (počet skrátených ihiel), `crop_box` (`center`, `axes`,
  `lo`, `hi`, pozri Postup; `null` pri `crop: false`), `scene_radius`, `num_train_views`,
  `before` a `after` (`psnr`, `ssim`, `lpips` priemer cez testovacie pohľady alebo `null`),
  `per_view` (`frame_id`, `before`, `after` pre každý testovací pohľad), `test_frame_ids`,
  `eval_resolution`, `sh_degree`, `antialiased`, `lpips_net`, `warnings`.
  Pri `eval: false` alebo bez testovacích pohľadov v 02 sú `per_view` a `test_frame_ids` prázdne
  a metriky `null`.
- `test_renders/<frame_id>.jpg` — najviac `num_test_renders` pohľadov: render pred, render po,
  referencia vedľa seba.
- `manifest.json`.

Fázy 07 a 08 čítajú gaussiány odtiaľto, nie z `02_reconstruction/`.

## Postup (implementácia `prune`)

1. **Polomer scény**: maximálna vzdialenosť stredov kamier od ich ťažiska × 1,1 (`scene_scale`
   ako v 02, `core/cameras.py`).
2. **Viditeľnosť**: pre každý gaussián počet tréningových kamier, do ktorých záberu zasahuje:
   stred sa premietne a obraz sa rozšíri o okraj 3σ najdlhšej osi v pixeloch (`fx · 3σ / z`), takže
   veľký gaussián so stredom tesne mimo záberu sa počíta. Ak je kamera vnútri gaussiánu, počíta sa
   vždy. Testovacie pohľady sa nepočítajú, aby metriky ostali nestranné.
3. **Ohraničenie** (`crop: true`): kváder okolo bodov SfM. Vodorovné osi z PCA súradníc XY bodov
   (+Z ostáva hore, kváder sleduje steny izby), rozsah od kvantilu `crop_quantile` po
   `1 − crop_quantile` pozdĺž každej osi, zväčšený o `crop_margin` × rozsah na každej strane.
   `axes` sú riadky = osi kvádra vo svete, lokálne súradnice `(x − center) · axesᵀ` ležia v
   `[lo, hi]`. Odstránia sa gaussiány so stredom mimo kvádra.
4. **Kritériá odstránenia** (každé sa dá vypnúť hodnotou 0):
   - `sigmoid(opacity) < min_opacity`,
   - viditeľnosť `< min_views`,
   - najdlhšia os `> max_scale_frac × polomer scény`.
5. **Ihly**: ak `max_aniso > 0` a pomer najdlhšej a strednej osi prekročí limit, najdlhšia os sa
   skráti na `max_aniso × stredná os` (`aniso_action: clamp`), alebo sa gaussián odstráni (`remove`).
6. **Metriky**: render testovacích pohľadov pred a po (gsplat, rovnaké rozlíšenie ako v reporte 02).

Implementácia `none` skopíruje `gaussians.ply` bez zmeny a zapíše report s `removed` = 0.

## Implementácie

| Názov | Popis | Licencia |
|---|---|---|
| `prune` (primárna) | orezanie podľa opacity, viditeľnosti z kamier, mierky a anizotropie; vlastný kód (NumPy, metriky cez gsplat) | Apache 2.0 (gsplat) |
| `none` | bez zmeny, na porovnanie „pred a po“ | — |

Možné rozšírenia (zatiaľ nenavrhnuté):
- krátky dotréning po orezaní (gaussiány okolo odstránených floaterov sa doladia),
- generatívne doplnenie miest, ktoré nevidela žiadna kamera. Takéto metódy majú zväčša
  nekomerčné licencie, smú byť iba alternatívou v samostatnom prostredí (pozri Licenčné pravidlo).

## Konfigurácia

```yaml
refinement:
  impl: prune              # prune | none
  crop: false              # ohraničenie kvádrom okolo bodov SfM
  crop_quantile: 0.01
  crop_margin: 0.1
  min_opacity: 0.02        # 0 = vypnuté
  min_views: 3             # 0 = vypnuté
  max_scale_frac: 0        # napr. 0.5; 0 = vypnuté
  max_aniso: 0             # napr. 10; 0 = vypnuté
  aniso_action: clamp      # clamp | remove
  eval: true               # metriky pred a po
  lpips: true
  num_test_renders: 8
```

Predvolené hodnoty vychádzajú z meraní na *m360_room* (pozri Stav): orezanie mierky a ihiel
je vypnuté, lebo bez dotréningu robí diery.

## Hardvér

Orezanie beží na CPU (NumPy), pre 1 M gaussiánov a ~300 kamier rádovo sekundy až desiatky sekúnd.
Metriky pred a po potrebujú CUDA GPU (render cez gsplat, ~1 GB VRAM). Prostredie `sp-gs`.
S `eval: false` fáza GPU nepotrebuje.

## Validácia výstupu

- PLY sa dá načítať, má vlastnosti zo sekcie 5 v správnom poradí, `0 < num_output ≤ num_input`,
  bez NaN/Inf.
- `num_input − num_output` sa rovná súčtu `removed`.
- Pri `eval: true`: render testovacieho pohľadu z uloženého PLY dáva PSNR v tolerancii voči `after`.
- Pokles PSNR po orezaní o viac než 1 dB sa zapíše do `warnings` (beh nezlyhá).

## Riziká a otvorené otázky

- Orezanie znižuje PSNR na testovacích pohľadoch, aj keď scéna z nových uhlov vyzerá lepšie.
  Metriky na holdout pohľadoch kvalitu z nových uhlov nezachytia. Treba porovnať aj očami
  (`test_renders/`, viewer) a zaznamenať v práci ako kompromis.
- **Ohraničenie scény vyreže aj skutočný obsah.** 3DGS kreslí odrazy v lesklých povrchoch ako
  „zrkadlový svet“ za nimi (na *m360_room* gaussiány pod lesklou podlahou) a výhľad z okna ako
  geometriu za oknom. Zvonka izby to vyzerá ako škvrny pod izbou a bodky vedľa nej, zvnútra ako
  správny odlesk a výhľad. Kváder podľa bodov SfM ich odstráni: čierne okno, tmavá diera
  v podlahe, PSNR 30,68 → 26,02 (najhorší pohľad −14,9 dB). Preto je `crop` predvolene vypnutý;
  hodí sa tam, kde je dôležitejšia čistá geometria než vzhľad.
- Kritérium viditeľnosti pôvodne počítalo iba stred gaussiánu a na *m360_room* odstránilo
  89 veľkých gaussiánov pozadia (jeden testovací pohľad −3,2 dB). Preto sa ráta s okrajom 3σ.
- `max_scale_frac` sa vzťahuje na polomer kamier, nie na veľkosť miestnosti. Keď kamery krúžia
  v malom priestore uprostred izby, „obrovské“ gaussiány sú steny a strop: na *m360_room*
  odstránenie 41 takých gaussiánov stálo jeden pohľad −6,4 dB (diera v stene).
- Príliš prísne `min_opacity` odstráni aj jemné detaily (vlasy, tenké predmety, okraje).
- Skrátenie ihiel zmení vzhľad v tréningových pohľadoch; bez dotréningu môžu vzniknúť diery.
- Rendery z externých viewerov (SuperSplat) nepoužívajú kompenzáciu `antialiased` z gsplat,
  takže malé gaussiány v nich vyzerajú sýtejšie než v našom renderi.

## Evaluácia

Počet odstránených gaussiánov podľa kritérií; PSNR, SSIM, LPIPS pred a po; vizuálne porovnanie
z pohľadov mimo trajektórie kamier; vplyv na lifting (07): podiel gaussiánov s `instance_id = -1`
a confidence s fázou 03 a bez nej (`impl: none`).

## Stav

- [x] špecifikácia, zaradenie do reťazca a do configu
- [x] implementácia `none` (`phases/p03_refinement/none.py`)
- [x] implementácia `prune` (`phases/p03_refinement/prune.py`, spoločná časť v `common.py`)
- [x] kontraktový test (`tests/contracts/test_p03.py`: syntetická miestnosť s vloženými
  artefaktmi každého druhu; orezanie beží aj bez CUDA, metriky pred/po sa bez CUDA preskočia)
- [x] overenie na *m360_room* (2026-10-18, GTX 1650 vo WSL, výstup 02 z `fast`: 500k gaussiánov,
  39 testovacích pohľadov, 1556×1037), beh s metrikami 29 s:

  | Nastavenie | Ponechané | PSNR | SSIM | Najhorší pohľad |
  |---|---|---|---|---|
  | pred (`none`) | 500 000 | 30,68 | 0,929 | — |
  | **opacity ≥ 0,02, viditeľnosť ≥ 3 (default)** | 409 970 | **30,67** | 0,929 | −0,04 dB |
  | opacity ≥ 0,05 | 302 659 | 30,42 | 0,923 | −0,67 dB |
  | default + ihly skrátené (`max_aniso` 10) | 409 970 (12 108 skrátených) | 30,23 | 0,924 | −2,35 dB |
  | default + `max_scale_frac` 0,5, viditeľnosť iba zo stredu | 409 845 | 30,28 | 0,927 | −8,77 dB |
  | default + `crop` (kvantil 0,005) | 407 559 | 26,76 | — | −12,8 dB |
  | default + `crop` (kvantil 0,01) | 405 659 | 26,02 | — | −14,9 dB |

  Viditeľnosť s okrajom 3σ na tejto dobre nasnímanej scéne neodstráni nič; LPIPS 0,1525 → 0,1527.
- [ ] overenie na scéne z telefónu (*izba*), kde floaterov mimo kamier je viac
- [ ] dotréning po orezaní (umožnil by zapnúť skracovanie ihiel bez dier)

Predbežné merania (2026-10-18, výstupy 02 z profilu `fast`, 7k krokov, 500k gaussiánov):
- *izba2*: 37 % gaussiánov má opacity < 0,02 a 58 % < 0,05; 1,7 % nevidí žiadna tréningová
  kamera; ~1 % leží ďalej než 8 × polomer scény. Orezanie (opacity ≥ 0,02, aspoň 3 kamery,
  najdlhšia os < 0,5 × polomer) ponechá 63 % gaussiánov, pri opacity ≥ 0,05 ponechá 42 %.
- *m360_room*: 2,6 % gaussiánov je ihlicovitých (najdlhšia os > 10 × stredná), z toho
  0,9 % všetkých má opacity > 0,3 (viditeľné „hviezdičky“).

## Zdroje

- Kerbl et al.: 3D Gaussian Splatting for Real-Time Radiance Field Rendering, SIGGRAPH 2023
  (orezávanie priehľadných a veľkých gaussiánov počas densifikácie).
- Xie et al.: PhysGaussian, CVPR 2024 (regularizácia pomeru osí proti ihlicovitým gaussiánom).
- Ye et al.: gsplat: An Open-Source Library for Gaussian Splatting, 2024. https://github.com/nerfstudio-project/gsplat
