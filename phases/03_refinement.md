# Fáza 03: Refinement (vyčistenie gaussiánov)

`contract_version: 3`

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
  súradniciach ako vstup. Poradie ponechaných gaussiánov sa zachová. Pri dotréningu
  (`finetune_steps > 0`) sa ich parametre zmenia, počet a poradie nie.
- `report.json` — `num_input`, `num_output`, `removed` (počet odstránených podľa kritéria:
  `crop`, `opacity`, `views`, `scale`, `isolated`, `aniso`; jeden gaussián sa započíta pri prvom kritériu, ktoré ho
  vyradí, v tomto poradí), `clamped` (počet skrátených ihiel), `crop_box` (`center`, `axes`,
  `lo`, `hi`, pozri Postup; `null` pri `crop: false`), od verzie 3 `finetune` (`steps`,
  `time_s`; `null` bez dotréningu), `scene_radius`, `num_train_views`,
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
   - najdlhšia os `> max_scale_frac × polomer scény`,
   - **izolované gaussiány** (`isolated_min_count > 0`): priestor sa rozdelí na voxely s hranou
     `isolated_voxel × polomer scény` a pre každý gaussián sa spočíta, koľko gaussiánov leží v jeho
     voxeli a 26 susedných. Ak je ich menej než `isolated_min_count` (vrátane neho), odstráni sa.
     Počíta sa iba medzi gaussiánmi, ktoré prežili predchádzajúce kritériá, takže zhluk zložený
     z priehľadného oparu alebo ležiaci mimo kvádra nepodrží osamotený gaussián pri živote.
     Izolovanosť sa určuje zo stredov, takže za izolované považuje aj veľké gaussiány, ktoré
     pokrývajú riedke oblasti (výhľad z okna). `isolated_keep_extent > 0` ich ponechá:
     izolovaný gaussián sa odstráni, iba ak 3σ najdlhšej osi ≤ `isolated_keep_extent` voxelov.
5. **Ihly**: ak `max_aniso > 0` a pomer najdlhšej a strednej osi prekročí limit, najdlhšia os sa
   skráti na `max_aniso × stredná os` (`aniso_action: clamp`), alebo sa gaussián odstráni (`remove`).
6. **Dotréning** (`finetune_steps > 0`, potrebuje CUDA): krátky tréning orezaných gaussiánov na
   tréningových pohľadoch z 02 (testovacie ostávajú bokom), aby susedné gaussiány zaplnili
   miesta po odstránených a po skrátených ihlách. Loss ako v 02 (0,8 · L1 + 0,2 · (1 − SSIM)),
   rozlíšenie `eval_resolution` z reportu 02, Adam s rýchlosťami učenia z konca tréningu v 02
   (pozícia 1,6e-6 × polomer scény, ostatné rovnako ako v 02). Bez densifikácie: počet
   a poradie gaussiánov sa nemenia. Pri `max_aniso > 0` a `aniso_action: clamp` sa po každom
   kroku najdlhšia os znova obmedzí na `max_aniso × stredná os`. Implementácia
   `phases/p03_refinement/finetune.py`.
7. **Metriky**: render testovacích pohľadov pred a po (gsplat, rovnaké rozlíšenie ako v reporte 02).

Implementácia `none` skopíruje `gaussians.ply` bez zmeny a zapíše report s `removed` = 0.

## Implementácie

| Názov | Popis | Licencia |
|---|---|---|
| `prune` (primárna) | orezanie podľa ohraničenia, opacity, viditeľnosti z kamier, mierky, izolovanosti a anizotropie; vlastný kód (NumPy, metriky cez gsplat) | Apache 2.0 (gsplat) |
| `none` | bez zmeny, na porovnanie „pred a po“ | — |

Možné rozšírenia (zatiaľ nenavrhnuté):
- generatívne doplnenie miest, ktoré nevidela žiadna kamera. Takéto metódy majú zväčša
  nekomerčné licencie, smú byť iba alternatívou v samostatnom prostredí (pozri Licenčné pravidlo).

## Konfigurácia

```yaml
refinement:
  impl: prune              # prune | none
  crop: true               # ohraničenie kvádrom okolo bodov SfM
  crop_quantile: 0.01
  crop_margin: 0.1
  min_opacity: 0.02        # 0 = vypnuté
  min_views: 3             # 0 = vypnuté
  max_scale_frac: 0        # napr. 0.5; 0 = vypnuté
  isolated_voxel: 0.02     # hrana voxelu ako podiel polomeru scény
  isolated_min_count: 4    # najmenší počet gaussiánov v okolí 3×3×3 voxelov; 0 = vypnuté
  isolated_keep_extent: 0  # ponechať izolované s 3σ > toľko voxelov (napr. 5); 0 = vypnuté
  max_aniso: 0             # napr. 10; 0 = vypnuté
  aniso_action: clamp      # clamp | remove
  finetune_steps: 0        # dotréning po orezaní (napr. 1000); 0 = vypnuté, inak treba CUDA
  seed: 0                  # poradie pohľadov pri dotréningu
  eval: true               # metriky pred a po
  lpips: true
  lpips_net: alex          # alex | vgg
  num_test_renders: 8
```

Predvolené hodnoty vychádzajú z meraní na *m360_room* (pozri Stav): orezanie mierky a ihiel
je vypnuté, lebo bez dotréningu robí diery. Ohraničenie a izolované gaussiány sú zapnuté
(rozhodnutie z 2026-10-26 po vizuálnom porovnaní v SuperSplate): pre sémantickú vrstvu a úpravy
je dôležitejšia čistá geometria než odlesky a výhľad z okna, pozri Riziká.

## Hardvér

Orezanie beží na CPU (NumPy), pre 1 M gaussiánov a ~300 kamier rádovo sekundy až desiatky sekúnd.
Metriky pred a po a dotréning potrebujú CUDA GPU (render cez gsplat, ~1 GB VRAM na metriky,
dotréning približne ako tréning v 02 bez densifikácie). Prostredie `sp-gs`.
S `eval: false` a `finetune_steps: 0` fáza GPU nepotrebuje. Dotréning drží všetky tréningové
snímky v RAM (uint8, pri 272 snímkach 1556×1037 ~1,3 GB, ako 02).

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
  v podlahe, PSNR 30,68 → 26,02 (najhorší pohľad −14,9 dB). Napriek tomu je `crop` predvolene zapnutý,
  lebo zvonka sú tieto gaussiány rušivé škvrny a pre úpravy scény je dôležitejšia čistá geometria.
  Kde záleží na vzhľade (čísla do porovnania s literatúrou), treba `crop: false`.
- Prah izolovanosti závisí od hustoty gaussiánov: pri veľmi malom `cap_max` v 02 (riedke
  gaussiány) by `isolated_min_count` začal odstraňovať aj skutočné povrchy. Voxel je relatívny
  k polomeru scény, nie k veľkosti gaussiánov.
- Kritérium viditeľnosti pôvodne počítalo iba stred gaussiánu a na *m360_room* odstránilo
  89 veľkých gaussiánov pozadia (jeden testovací pohľad −3,2 dB). Preto sa ráta s okrajom 3σ.
- `max_scale_frac` sa vzťahuje na polomer kamier, nie na veľkosť miestnosti. Keď kamery krúžia
  v malom priestore uprostred izby, „obrovské“ gaussiány sú steny a strop: na *m360_room*
  odstránenie 41 takých gaussiánov stálo jeden pohľad −6,4 dB (diera v stene).
- Príliš prísne `min_opacity` odstráni aj jemné detaily (vlasy, tenké predmety, okraje).
- Skrátenie ihiel zmení vzhľad v tréningových pohľadoch; bez dotréningu môžu vzniknúť diery.
- Dotréning optimalizuje iba tréningové pohľady: môže čiastočne vrátiť artefakty, ktoré
  v nich nevadia (napr. zvýšiť opacity zvyšného oparu). Bez densifikácie však nevzniknú nové
  gaussiány a orezané floatery sa nevrátia.
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
- [x] kritérium izolovaných gaussiánov a zapnuté ohraničenie (2026-10-26, `contract_version` 2):
  na *m360_room* (`fast`) ladiaca kópia bez gaussiánov mimo kvádra a bez izolovaných (spolu
  ~4 500 z 409 000) vyzerala v SuperSplate najčistejšie z porovnávaných variantov. Merania
  fázy na tom istom výstupe 02 (39 testovacích pohľadov zvnútra izby):

  | Nastavenie | Odstránené | PSNR | Najhorší pohľad |
  |---|---|---|---|
  | len opacity (pôvodný default) | 90 508 | 30,22 | — |
  | + izolované, bez `crop` | 92 188 | 27,56 | −16,4 dB |
  | **+ `crop` + izolované (default)** | 95 232 | **25,16** | −20,2 dB |

  Zvnútra je okno čierne a chýbajú odlesky v podlahe (prijatý kompromis, pozri Riziká).
  Izolovanosť sa počíta zo stredov: 934 z 1 680 izolovaných gaussiánov je veľkých (3σ > 5 voxelov,
  opacity ~0,4), sú to riedke oblasti ako výhľad z okna; drobných smietok (≤ 1 voxel) je ~70.
- [x] voľba `isolated_keep_extent` (2026-10-27, `impl_version` 3): veľké izolované gaussiány
  sa neodstránia. Počty na výstupoch 02 (`fast`) po ostatných kritériách (opacity, viditeľnosť),
  odstránené ako izolované pri rôznych hodnotách (CPU, bez renderu):

  | Scéna | `crop` | Izolované (0 = vypnuté) | ≤ 1 voxel | ≤ 2 | ≤ 5 | ≤ 10 |
  |---|---|---|---|---|---|---|
  | *m360_room* | áno | 893 | 56 | 141 | 453 | 719 |
  | *m360_room* | nie | 1 680 | 69 | 198 | 746 | 1 246 |
  | *m360_kitchen* | áno | 36 | 2 | 8 | 19 | 32 |
  | *m360_kitchen* | nie | 2 163 | 39 | 281 | 901 | 1 432 |

  Default ostáva 0 (správanie ako doteraz). Či hodnota 5 vráti výhľad z okna bez návratu
  škvŕn, treba porovnať renderom a v SuperSplate (čaká na GPU).
- [x] dotréning po orezaní (`finetune_steps`, 2026-10-27, `contract_version` 3): implementácia
  a test na CPU (obmedzenie ihiel); kontraktový test dotréningu (`test_finetune`) potrebuje CUDA
  a zatiaľ nebežal
- [ ] dotréning na *m360_room*: PSNR pred a po pri 500 / 1000 / 2000 krokoch, so skracovaním
  ihiel (`max_aniso` 10) a bez neho; potom rozhodnúť o defaulte (čaká na súhlas s GPU)
- [ ] overenie na scéne z telefónu (*izba*), kde floaterov mimo kamier je viac (video *izby*
  nie je na disku, čaká na dáta)

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
