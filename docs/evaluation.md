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
| Segmentácia (05, 07) | mIoU 2D masiek; mIoU renderovaných 3D inštancií; počet inštancií vs. GT |
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
