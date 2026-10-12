# Prostredia

Fázy komunikujú iba cez súbory, takže každá skupina môže mať vlastné prostredie
(iná verzia PyTorch, CUDA, Pythonu). Navrhované skupiny — upraviť podľa reality pri inštalácii:

| Prostredie | Fázy | Kľúčové balíky |
|---|---|---|
| `sp-core` | 00, 01, runner | opencv, pillow, pycolmap, numpy |
| `sp-gs` | 02, 06, 11, 12 | torch, gsplat, viser, LaMa |
| `sp-seg` | 03, 04, 05, 09 | torch, Grounding DINO, SAM 2, open_clip, RAM++ |
| `sp-llm` | 07 (VLM), 08 | transformers / vLLM, Qwen |
| `sp-gen` | 10 | TRELLIS, diffusers (FLUX) |

Balík `scenepipe` (jadro) sa inštaluje do každého prostredia (`pip install -e .`), aby fázy
zdieľali I/O kód a kontrakty. Runner spúšťa fázu cez `conda run -n <env>` alebo podobne.
Súbory prostredí (`sp-*.yml` alebo `requirements-*.txt`) pribudnú pri implementácii fáz.
