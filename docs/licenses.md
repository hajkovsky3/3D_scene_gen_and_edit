# Licencie závislostí a modelov

Pravidlo projektu: iba **free to use a open source** (pozri `CLAUDE.md`).
Každú novú závislosť alebo model pred použitím zapísať sem. Overovať licenciu **kódu aj váh**.

Stav: `OK` = permisívna OSI licencia · `HRANIČNÉ` = vlastná licencia, free, ale nie OSI — len po
výslovnom rozhodnutí · `ZAKÁZANÉ` = nekomerčné / územne obmedzené · `OVERIŤ` = ešte neoverené.

## Používané / plánované

| Komponent | Fáza | Kód | Váhy | Stav | Poznámka |
|---|---|---|---|---|---|
| COLMAP 4.x (+ GLOMAP, ALIKED, LightGlue) | 01 | BSD | — / ONNX modely | OVERIŤ váhy ALIKED/LightGlue | GLOMAP je od 4.0 súčasťou COLMAP |
| VGGT (`VGGT-1B-Commercial`) | 01 alt | VGGT License | VGGT License, prístup cez formulár | HRANIČNÉ | pôvodný checkpoint `VGGT-1B` je nekomerčný |
| gsplat | 02, 06, 11, 12 | Apache 2.0 | — | OK | |
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
| Llama (multimodálne verzie) | Llama licencia, obmedzenia v EÚ | VLM |
