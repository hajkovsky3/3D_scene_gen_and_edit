# Evaluácia

Prehľad metrík a dát. Konkrétne metriky každej fázy sú v sekcii *Evaluácia* jej dokumentu.

## Dáta

| Dataset | Na čo | Licencia / poznámka |
|---|---|---|
| Mip-NeRF 360 (room, counter, kitchen, bonsai, garden) | rekonštrukcia (02), úpravy (12), porovnanie s literatúrou | autori licenciu neuvádzajú (stav 2026-10-15; tretie strany tvrdia rôzne: Apache 2.0, MIT), len na evaluáciu, nedistribuovať; scény sa sťahujú jednotlivo z `360_v2.zip` cez HTTP range (`images_2` + `sparse`, ~0,3–0,4 GB na scénu) |
| LERF-Mask / 3D-OVS | open-vocabulary segmentácia v 3D (07), grounding (10) | overiť podmienky |
| Vlastné scény (izba, stôl s predmetmi) | celý reťazec; scéna „pred a po“ fyzickom odstránení objektu na meranie inpaintingu | vlastné |

Datasety s nekomerčnou licenciou (napr. ScanNet++) sú na akademickú evaluáciu v poriadku,
ale **nesmú** sa stať súčasťou repozitára ani tréningových dát modulov.

## Metriky

| Oblasť | Metrika |
|---|---|
| SfM (01) | podiel registrovaných snímok, reprojekčná chyba, čas (`report.json` → `timings_s`); pri známych pózach chyba rotácie a polohy kamier po zarovnaní podobnosťou (`python -m scenepipe.eval.sfm_poses --scene X --ref-colmap <sparse>` alebo `--ref-scene Y`) |
| Rekonštrukcia (02, 03) | PSNR, SSIM, LPIPS na holdout pohľadoch; počet gaussiánov; VRAM; čas; z nových uhlov bez referencie podiel dier a nesúlad hĺbky s monokulárnym odhadom (`python -m scenepipe.eval.novel_views --scene X [--gaussians 03_refinement]`) |
| Tagging (04) | recall a presnosť slovníka voči ručnému slovníku scény (sekcia nižšie) |
| Segmentácia (05, 07) | mIoU 2D masiek; mIoU renderovaných 3D inštancií; počet inštancií vs. GT (ručná anotácia, sekcia nižšie) |
| Graf (08) | presnosť labelov; precision/recall hrán *on*, *near* |
| Prompt (09, 10) | presnosť typu operácie; top-1 presnosť groundingu; podiel nejednoznačných |
| Úprava (12) | PSNR/LPIPS mimo zasiahnutej oblasti (izolácia); CLIP directional similarity; PSNR voči reálnej scéne po odstránení objektu; čas |

## Ablácie (návrhy do práce)

- Lifting: `gsplat_vote` vs. `pointcloud_project` (07). Bez 3D ground truth zatiaľ
  `python -m scenepipe.eval.lifting_consistency --scene <scéna> [--lifting <adresár>]`:
  inštancie sa vyrenderujú do snímok 05 a porovnajú s maskami (IoU, presnosť, úplnosť,
  rozpis podľa labelov). Masky sú vstupom 07, takže meria vernosť liftingu, nie presnosť.
- Tracking v 05 zapnutý/vypnutý → vplyv na kvalitu inštancií v 07.
- Parsovanie: `llm_local` vs. `rules` (09); grounding: `clip_graph` vs. `label_only` (10).
- Inpainting: per-view LaMa vs. referenčný pohľad + reprojekcia (12).

## Ručná anotácia (ground truth pre 04, 05, 07)

Mip-NeRF 360 nemá sémantické anotácie, preto sa ground truth kreslí ručne v lokálnom
nástroji (`src/scenepipe/annotate/`, prostredie `sp-gs` kvôli SAM 2):

```bash
python -m scenepipe.annotate --scene m360_counter        # http://localhost:8765, Ctrl+C ukončí
python -m scenepipe.annotate --scene m360_counter --no-sam   # iba štetec, bez modelu
```

Pri prvom spustení nástroj vyberie `--frames` (12) snímok rovnomerne zo snímok, ktoré
spracovala fáza 05 (profil `fast` berie každú 3. snímku, takže masky majú oba profily).
Maska sa navrhne cez SAM 2.1 Hiera-S z kliknutých bodov (pozitívne / negatívne) alebo
obdĺžnika a opraví štetcom. Každý objekt dostane názov inštancie `trieda#číslo`
(`chair#1`): **rovnaký názov v rôznych snímkach je ten istý 3D objekt**, trieda je časť
pred `#`. Masky v snímke sa neprekrývajú (novší objekt vyhrá svoje pixely). Slovník scény
(`vocabulary.txt`) obsahuje všetky triedy objektov v scéne, aj tie, ktoré sa v snímkach
nekreslia. Nástroj neukazuje návrhy z 04, aby slovník nebol nimi ovplyvnený.

Súbory v `workspace/<scéna>/annotations/`:
- `frames.json` — `{"frames": [frame_id, …]}` vybrané snímky,
- `frames/<frame_id>.json` — `frame_id`, `image` (`01_sfm/images_undistorted/…`), `width`,
  `height`, `objects`: `[{"instance", "label", "rle"}]` (COCO RLE ako v 05),
- `vocabulary.txt` — triedy, jedna na riadok,
- `aliases.yaml` (voliteľné, píše sa ručne) — `trieda GT: [akceptované labely modelu]`, napr.
  `oven mitt: [glove]`, pre rozhodnutia o ekvivalencii názvov nad rámec
  `configs/tag_synonyms.yaml`.

Vyhodnotenia (labely sa porovnávajú po synonymách a aliasoch; triedy z `configs/tag_stuff.txt`,
napr. podlaha a steny, sa v 05 a 07 nepočítajú, ak nie je `--with-stuff`):

| Príkaz | Fáza | Metriky |
|---|---|---|
| `python -m scenepipe.eval.tagging_recall --scene X` | 04 | recall a presnosť slovníka voči `vocabulary.txt`, zoznam chýbajúcich a navyše |
| `python -m scenepipe.eval.segmentation_gt --scene X` | 05 | mIoU (párovanie 1:1 podľa IoU), recall a presnosť pri IoU ≥ 0,5, presnosť labelov, rozpis podľa tried |
| `python -m scenepipe.eval.lifting_gt --scene X [--lifting dir]` | 07 | mIoU po snímkach, mIoU s jednotným priradením GT objekt → 3D inštancia, recall@0,5, fragmentácia, zlúčené objekty, presnosť labelov (CUDA) |

Na vizuálne porovnanie variantov 02/03 slúži
`python -m scenepipe.eval.render_compare --scene X --gaussians A.ply B.ply --frames … --out …`
(mriežka renderov, riadky = pohľady, stĺpce = varianty).
