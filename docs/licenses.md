# Licencie závislostí a modelov

Pravidlo projektu: všetko musí byť **free to use**; primárna implementácia každej fázy navyše
permisívna, nekomerčné licencie len pre alternatívy, baseline a vyhodnotenie (pozri `CLAUDE.md`).
Každú novú závislosť alebo model pred použitím zapísať sem. Overovať licenciu **kódu aj váh**.

Stav:
- `OK` = permisívna OSI licencia, smie byť v primárnej implementácii,
- `HRANIČNÉ` = vlastná licencia, free, ale nie OSI — len po výslovnom rozhodnutí autora,
- `NEKOMERČNÉ` = nekomerčná / výskumná licencia alebo copyleft (GPL) pribalený v binárkach —
  iba alternatívne implementácie, baseline a vyhodnotenie; nikdy primárna cesta, kód sa iba volá
  (samostatné prostredie), nekopíruje sa do repozitára,
- `ZAKÁZANÉ` = pre nás nie je free to use (územné vylúčenie EÚ, platené / iba API),
- `OVERIŤ` = ešte neoverené.

## Používané / plánované

| Komponent | Fáza | Kód | Váhy | Stav | Poznámka |
|---|---|---|---|---|---|
| OpenCV (`opencv-python-headless`) | 00 | Apache 2.0 | — | OK | wheel pribaľuje FFmpeg v LGPL zostave (dynamicky linkovaný) |
| Pillow | 00 | MIT-CMU (HPND) | — | OK | |
| NumPy | všetky | BSD-3 | — | OK | |
| PyYAML | jadro | MIT | — | OK | |
| COLMAP 4.x (+ GLOMAP, ALIKED, LightGlue) | 01 | BSD | — / ONNX modely | OVERIŤ váhy ALIKED/LightGlue | GLOMAP je od 4.0 súčasťou COLMAP |
| `pycolmap` 4.2 (wheel z PyPI) | 01 | BSD-3 | — | HRANIČNÉ (schválené) | wheel pribaľuje SuiteSparse: SPQR (GPL-2+), CHOLMOD (časti GPL-2+/LGPL-2.1+), ostatné BSD; ďalej Ceres (BSD), OpenImageIO (Apache 2.0), OpenBLAS (BSD). Autor schválil 2026-10-12 ako tranzitívnu binárnu závislosť externého nástroja: náš kód GPL kód neobsahuje ani nedistribuuje. Oficiálne binárky COLMAP majú to isté. |
| VGGT (`VGGT-1B-Commercial`) | 01 alt | VGGT License | VGGT License, prístup cez formulár | HRANIČNÉ | pôvodný checkpoint `VGGT-1B` je nekomerčný |
| gsplat 1.5 | 02, 06, 11, 12 | Apache 2.0 | — | OK | CUDA jadro sa kompiluje lokálne (JIT); obsahuje GLM (MIT) |
| PyTorch, torchvision | 02, 06, 11, 12 | BSD-3 | — | OK | CUDA wheely pribaľujú knižnice NVIDIA (cuBLAS, cuDNN …) pod NVIDIA licenciou, len na beh |
| CUDA Toolkit 13.0 | 02 (build gsplat) | NVIDIA EULA (proprietárna, bezplatná) | — | OK (systémový nástroj, schválené autorom 2026-10-14) | iba kompilátor `nvcc` a hlavičky na zostavenie gsplat, nič sa z neho nedistribuuje; bez neho sa nedá použiť žiadna CUDA knižnica |
| torchmetrics | 02 (metriky) | Apache 2.0 | LPIPS lineárne vrstvy pribalené (BSD-2, pôvodný repozitár LPIPS) | OK | |
| AlexNet (`torchvision`, `alexnet-owt-7be5be79.pth`) | 02 (LPIPS) | BSD-3 | BSD-3 (torchvision), trénované na ImageNet | OK | stiahnuté so súhlasom autora 2026-10-14, iba na meranie LPIPS; vypnuteľné `lpips: false` |
| RAM++ | 03 | Apache 2.0 | OVERIŤ | OVERIŤ | |
| Grounding DINO (Swin-T/B, open-source verzia) | 04 | Apache 2.0 | Apache 2.0 | OK | verzie 1.5/1.6 a DINO-X sú len API |
| SAM 2 / 2.1 | 04 | Apache 2.0 | Apache 2.0 | OK | |
| SAM 3 | 04 alt | SAM License | SAM License, prístup cez HF | HRANIČNÉ | prečítať LICENSE pred použitím |
| OpenCLIP (ViT-H-14 laion2b) | 05, 09 | MIT | MIT | OK | |
| SigLIP / SigLIP 2 | 05 alt | Apache 2.0 | Apache 2.0 | OK | |
| Qwen2.5-VL-7B-Instruct | 07 | Apache 2.0 | Apache 2.0 | OK | 3B a 72B majú iné licencie! |
| Qwen3-8B | 08 | Apache 2.0 | Apache 2.0 | OVERIŤ konkrétnu veľkosť | |
| Florence-2 | 07 alt | MIT | MIT | OK | |
| FLUX.1-schnell | 10 | Apache 2.0 | Apache 2.0 | OK | FLUX.1-dev je nekomerčný! |
| BiRefNet | 10 | MIT | MIT | OVERIŤ | |
| TRELLIS / TRELLIS.2 | 10 | MIT | MIT | OVERIŤ závislosti | nvdiffrast (NVIDIA, nekomerčná), rasterizéry odvodené od INRIA; ak sú tieto závislosti nutné, TRELLIS sa stáva `NEKOMERČNÉ` alternatívou a primárnou cestou 10 zostáva `primitive` / permisívny pipeline |
| LaMa (big-lama) | 11 | Apache 2.0 | OVERIŤ | OVERIŤ | |
| viser | 12 | Apache 2.0 | — | OK | |
| CustomTkinter 5.2 (`customtkinter`) | monitor | MIT (repozitár; PyPI metadáta uvádzajú CC0 1.0) | — | OK | okno monitora behu; závislosti `darkdetect` (BSD-3), `packaging` (Apache 2.0 / BSD-2), Tk zo štandardnej knižnice Pythonu |
| SuperSplat | 12 (externé) | MIT | — | OK | iba nástroj na prezeranie |
| pycocotools | 04+ | BSD-2 | — | OK | |
| NetworkX | 07 | BSD-3 | — | OK | |

## Nekomerčné (iba alternatívy, baseline a vyhodnotenie)

Smú sa použiť ako alternatívna implementácia fázy alebo na porovnanie v práci, nie v primárnej
ceste. Pred použitím doplniť riadok do tabuľky vyššie (konkrétna verzia, váhy, podmienky).

| Komponent | Licencia | Možné použitie |
|---|---|---|
| 3DGS oficiálna implementácia INRIA, `diff-gaussian-rasterization` | Gaussian-Splatting License (nekomerčná) | porovnanie s 02; závislosť nekomerčných alternatív |
| GaussianEditor | S-Lab License 1.0 (nekomerčná) | baseline úprav (11) |
| LangSplat, Feature 3DGS, Gaussian Grouping a iné nad INRIA kódom | dedia INRIA licenciu | baseline / alternatíva liftingu (06) |
| MASt3R, DUSt3R | CC BY-NC-SA 4.0 | alternatíva SfM pri málo snímkach (01) |
| VGGT-1B (pôvodný checkpoint), VGGT-Omega | FAIR Noncommercial | alternatíva SfM (01); primárne radšej `VGGT-1B-Commercial` |
| SuperPoint (váhy Magic Leap) | nekomerčná | hloc pipeline (01) |
| `nvdiffrast` | NVIDIA Source Code License (nekomerčná) | závislosť TRELLIS (10) |
| FLUX.1-dev, SDXL-Turbo, SD 3.x | nekomerčné / vlastné (pri SD 3.x overiť konkrétnu verziu) | alternatíva text → obrázok (10) |
| `pillow-heif` (wheel balíčky) | BSD-3, ale pribaľuje x265 (GPL-2) | voliteľná podpora HEIC vo fáze 00 |
| PyAV (wheel balíčky) | BSD-3, ale pribaľuje x264/x265 (GPL-2) | alternatívne čítanie videa vo fáze 00 |

## Zakázané (pre nás nie sú free to use; citovať v práci je OK)

| Komponent | Licencia | Prečo sa spomína |
|---|---|---|
| Hunyuan3D | Tencent licencia vylučujúca EÚ | image-to-3D |
| Llama (multimodálne verzie) | Llama licencia, obmedzenia v EÚ | VLM |
| Grounding DINO 1.5 / 1.6, DINO-X | iba platené API | detekcia (04) |
