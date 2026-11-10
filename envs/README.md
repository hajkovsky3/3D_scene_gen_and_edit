# Prostredia

Fázy komunikujú iba cez súbory, takže každá skupina môže mať vlastné prostredie
(iná verzia PyTorch, CUDA, Pythonu). Navrhované skupiny — upraviť podľa reality pri inštalácii:

| Prostredie | Fázy | Kľúčové balíky |
|---|---|---|
| `sp-core` | 00, 01, runner | opencv, pillow, pycolmap, numpy |
| `sp-gs` | 02, 03, 07, 12, 13 (+ 01 s GPU párovaním, zatiaľ aj 04 a 05) | torch, gsplat, transformers (hĺbkový prior), viser, LaMa, pycolmap |
| `sp-seg` | 06, 10 (04 a 05 sa presunú sem, ak bude treba) | torch, Grounding DINO, SAM 2, open_clip, RAM++ |
| `sp-llm` | 08 (VLM; zatiaľ v `sp-gs`), 09 | transformers / vLLM, Qwen |
| `sp-gen` | 11 | TRELLIS, diffusers (FLUX) |

Balík `scenepipe` (jadro) sa inštaluje do každého prostredia (`pip install -e .`), aby fázy
zdieľali I/O kód a kontrakty. Runner spúšťa fázu cez `conda run -n <env>` alebo podobne.
Súbory prostredí (`sp-*.yml` alebo `requirements-*.txt`) pribudnú pri implementácii fáz.

## `sp-core` (fázy 00, 01)

```bash
python -m venv .venv            # Python 3.11+ (na Windowse overené 3.14)
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev,ingest,sfm]"
```

## `sp-gs` (fázy 02, 03, 07, 12, 13)

gsplat má hotové wheely iba pre Python 3.10 a starý PyTorch, preto sa jeho CUDA jadro
kompiluje lokálne (JIT pri prvom použití, ~6 min). Na Windowse sa to robí vo **WSL2**
(Ubuntu), kde stačí `nvcc` a `gcc`; GPU ovládač zostáva windowsový. Overené: Ubuntu 26.04,
Python 3.14, PyTorch 2.14.1+cu130, gsplat 1.5.3, CUDA Toolkit 13.0, GTX 1650 (sm_75).

Major verzia CUDA Toolkitu musí sedieť s CUDA verziou PyTorch wheelu (`cu130` → 13.0)
a ovládač ju musí podporovať (`nvidia-smi` vo Windowse ukazuje najvyššiu verziu).

```bash
# 1. raz, ako root (wsl -u root): CUDA Toolkit pre WSL (bez ovládača) + kompilátor
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update
apt-get install -y build-essential python3-venv python3-dev ninja-build cuda-toolkit-13-0

# 2. prostredie (venv na Linux disku, repozitár ostáva na /mnt/c/...)
python3 -m venv ~/venvs/sp-gs && source ~/venvs/sp-gs/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
cd /mnt/c/Users/<user>/Documents/3D_scene_gen_and_edit
pip install -e ".[dev,gs,depth,ingest,sfm,lightglue]"   # sfm: 01 s párovaním na GPU, lightglue: SIFT + LightGlue v 01, depth: hĺbkový prior v 02

# 3. pred každým použitím (gsplat JIT potrebuje nvcc)
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
export TORCH_CUDA_ARCH_LIST=7.5    # architektúra GPU (GTX 1650 = 7.5), skráti kompiláciu
pytest tests/contracts/test_p02.py # prvý beh skompiluje gsplat do ~/.cache/torch_extensions
```

Na vzdialenom Linux serveri s GPU platí to isté bez kroku WSL (CUDA repozitár podľa distribúcie,
`TORCH_CUDA_ARCH_LIST` podľa GPU).

### Fáza 04 (RAM++) v `sp-gs`

Fáza 04 zatiaľ beží v `sp-gs` (torch a transformers tam už sú). Do `sp-seg` sa presunie,
ak si SAM 2 alebo Grounding DINO vynútia iné verzie.

```bash
source ~/sp-gs.sh
pip install -e ".[tagging]"   # ram (pripnutý commit), timm, scipy
# váhy (3,01 GB) a slovník BERT tokenizera do checkpoints/ (iba so súhlasom, pozri docs/licenses.md):
python -c "from huggingface_hub import hf_hub_download as d; d('xinyu1205/recognize-anything-plus-model', 'ram_plus_swin_large_14m.pth', local_dir='checkpoints/ram_plus'); [d('google-bert/bert-base-uncased', f, local_dir='checkpoints/bert-base-uncased') for f in ('vocab.txt', 'config.json', 'tokenizer_config.json')]"
pytest tests/contracts/test_p04.py
```

### Fáza 05 (Grounding DINO + SAM 2) v `sp-gs`

Oba modely sú v `transformers` (už v `sp-gs`), netreba nič kompilovať. Váhy (iba so súhlasom):

```bash
python -c "from huggingface_hub import snapshot_download as d; [d(r, cache_dir='checkpoints/huggingface', allow_patterns=['*.json', '*.txt', 'model.safetensors'])  for r in ('IDEA-Research/grounding-dino-tiny', 'facebook/sam2.1-hiera-small')]"
pytest tests/contracts/test_p05.py tests/test_masks.py
```

### Fáza 08 (VLM Qwen3-VL-4B) v `sp-gs`

Fáza 08 zatiaľ beží v `sp-gs` (transformers 5 už má `Qwen3VLForConditionalGeneration`), do
`sp-llm` sa presunie, keď vznikne. 4-bit kvantizácia potrebuje `bitsandbytes` a `accelerate`.
Váhy (iba so súhlasom, 8,9 GB):

```bash
source ~/sp-gs.sh
pip install bitsandbytes accelerate
python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3-VL-4B-Instruct', cache_dir='checkpoints/huggingface', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', 'LICENSE*', 'README.md'])"
HF_HUB_OFFLINE=1 python -m scenepipe run 08_scene_graph --scene room
```
