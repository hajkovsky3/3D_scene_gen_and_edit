# Licencie závislostí a modelov

Pravidlo projektu: iba **free to use a open source** (pozri `CLAUDE.md`).
Každú novú závislosť alebo model pred použitím zapísať sem. Overovať licenciu **kódu aj váh**.

Stav: `OK` = permisívna OSI licencia · `HRANIČNÉ` = vlastná licencia, free, ale nie OSI — len po
výslovnom rozhodnutí · `ZAKÁZANÉ` = nekomerčné / územne obmedzené · `OVERIŤ` = ešte neoverené.

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
| TRELLIS / TRELLIS.2 | 10 | MIT | MIT | OVERIŤ závislosti | nvdiffrast (NVIDIA, nekomerčná), rasterizéry odvodené od INRIA |
| LaMa (big-lama) | 11 | Apache 2.0 | OVERIŤ | OVERIŤ | |
| viser | 12 | Apache 2.0 | — | OK | |
| SuperSplat | 12 (externé) | MIT | — | OK | iba nástroj na prezeranie |
| pycocotools | 04+ | BSD-2 | — | OK | |
| NetworkX | 07 | BSD-3 | — | OK | |

## Zakázané (citovať v práci je OK, kód/váhy nepoužívať)

| Komponent | Licencia | Prečo sa spomína |
|---|---|---|
| 3DGS oficiálna implementácia INRIA, `diff-gaussian-rasterization` | Gaussian-Splatting License (nekomerčná) | pôvodná metóda |
| GaussianEditor | S-Lab License 1.0 (nekomerčná) | pôvodný návrh fázy úprav |
| LangSplat, Feature 3DGS, Gaussian Grouping a iné nad INRIA kódom | dedia INRIA licenciu | alternatívy liftingu |
| MASt3R, DUSt3R | CC BY-NC-SA 4.0 | alternatíva SfM |
| VGGT-1B (pôvodný checkpoint), VGGT-Omega | FAIR Noncommercial | alternatíva SfM |
| SuperPoint (váhy Magic Leap) | nekomerčná | hloc pipeline |
| Hunyuan3D | Tencent licencia vylučujúca EÚ | image-to-3D |
| FLUX.1-dev, SDXL-Turbo, SD 3.x | nekomerčné / vlastné | text-to-image |
| `pillow-heif` (wheel balíčky) | BSD-3, ale pribaľuje x265 (GPL-2) | podpora HEIC vo fáze 00 |
| PyAV (wheel balíčky) | BSD-3, ale pribaľuje x264/x265 (GPL-2) | čítanie videa vo fáze 00, nahradené `cv2.VideoCapture` |
| Llama (multimodálne verzie) | Llama licencia, obmedzenia v EÚ | VLM |
