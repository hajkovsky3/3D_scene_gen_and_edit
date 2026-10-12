# Evaluácia

Prehľad metrík a dát. Konkrétne metriky každej fázy sú v sekcii *Evaluácia* jej dokumentu.

## Dáta

| Dataset | Na čo | Licencia / poznámka |
|---|---|---|
| Mip-NeRF 360 (room, counter, kitchen, bonsai, garden) | rekonštrukcia (02), úpravy (11), porovnanie s literatúrou | overiť podmienky použitia |
| LERF-Mask / 3D-OVS | open-vocabulary segmentácia v 3D (06), grounding (09) | overiť podmienky |
| Vlastné scény (izba, stôl s predmetmi) | celý reťazec; scéna „pred a po“ fyzickom odstránení objektu na meranie inpaintingu | vlastné |

Datasety s nekomerčnou licenciou (napr. ScanNet++) sú na akademickú evaluáciu v poriadku,
ale **nesmú** sa stať súčasťou repozitára ani tréningových dát modulov.

## Metriky

| Oblasť | Metrika |
|---|---|
| SfM (01) | podiel registrovaných snímok, reprojekčná chyba, čas |
| Rekonštrukcia (02) | PSNR, SSIM, LPIPS na holdout pohľadoch; počet gaussiánov; VRAM; čas |
| Segmentácia (04, 06) | mIoU 2D masiek; mIoU renderovaných 3D inštancií; počet inštancií vs. GT |
| Graf (07) | presnosť labelov; precision/recall hrán *on*, *near* |
| Prompt (08, 09) | presnosť typu operácie; top-1 presnosť groundingu; podiel nejednoznačných |
| Úprava (11) | PSNR/LPIPS mimo zasiahnutej oblasti (izolácia); CLIP directional similarity; PSNR voči reálnej scéne po odstránení objektu; čas |

## Ablácie (návrhy do práce)

- Lifting: `gsplat_vote` vs. `pointcloud_project` (06).
- Tracking v 04 zapnutý/vypnutý → vplyv na kvalitu inštancií v 06.
- Parsovanie: `llm_local` vs. `rules` (08); grounding: `clip_graph` vs. `label_only` (09).
- Inpainting: per-view LaMa vs. referenčný pohľad + reprojekcia (11).
