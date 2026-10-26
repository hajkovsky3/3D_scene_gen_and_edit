# Inštalácia na čistý počítač

Postup, ako po preinštalovaní Windowsu a `git clone` tohto repozitára znova rozbehať celý vývoj.
Stav overený 2026-10-23 na notebooku i5-9300H, 24 GB RAM, GTX 1650 4 GB, Windows 11 Pro.
Podrobnosti o prostrediach sú v [`envs/README.md`](envs/README.md).

## 0. Pred zmazaním disku zálohovať

Tieto veci **nie sú v gite** (`.gitignore`) a po zmazaní disku zmiznú:

| Čo | Kde | Poznámka |
|---|---|---|
| vlastné vstupy scén (videá, fotky) | `workspace/<scéna>/input/` (`izba`, `izba2`, `tete_momo`) | jediná vec, ktorá sa nedá znova stiahnuť ani prepočítať, **zálohovať** |
| výsledky fáz | `workspace/<scéna>/` okrem `input/` | dajú sa prepočítať; zálohovať len ak treba čísla do práce bez nového behu |
| Mip-NeRF 360 | `workspace/_datasets/mipnerf360/` (`room`, `counter`, `kitchen`, ~1,1 GB) | dá sa znova stiahnuť (krok 5) |
| váhy modelov | `checkpoints/huggingface/` (Depth Anything V2 Small, ~95 MB) | stiahnu sa samé pri prvom behu |
| nastavenie monitora | `configs/local.yaml` | obsah = `configs/local.example.yaml`, stačí skopírovať |
| pamäť Claude Code | `C:\Users\sismo\.claude\` | nepovinné; podstatné fakty sú v tomto súbore |

## 1. Základné nástroje (Windows)

```powershell
winget install Git.Git
winget install Python.Python.3.14        # projekt: Python 3.11+, overené 3.14
winget install Nvidia.GeForceExperience  # alebo ovládač z nvidia.com; potrebná CUDA 13.x (nvidia-smi)
```

Po inštalácii ovládača overiť `nvidia-smi` (pravý horný roh: `CUDA Version` musí byť ≥ 13.0;
novšie ovládače, napr. 621.01, tam píšu `CUDA UMD Version`).

```powershell
git clone https://github.com/hajkovsky3/3D_scene_gen_and_edit.git
cd 3D_scene_gen_and_edit
```

## 2. Prostredie `sp-core` vo Windowse (fázy 00, monitor, testy, lint)

```powershell
py -3.14 -m venv .venv       # bez spúšťača py (winget ho nemusí nainštalovať): python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,ingest,sfm,monitor]"
copy configs\local.example.yaml configs\local.yaml
pytest
```

Overené verzie: numpy 2.5, opencv-python-headless 5.0.0.93, pillow 12.3, pycolmap 4.2.1,
customtkinter 5.2.2, pytest 9.1, ruff 0.17. V tomto prostredí nie je torch, takže testy
`test_p02` a `test_torch_match` sa preskočia (to je v poriadku).

## 3. WSL2 Ubuntu (GPU fázy 01, 02, 03)

V PowerShelli ako administrátor:

```powershell
wsl --install -d Ubuntu      # overené Ubuntu 26.04 LTS; po reštarte zadať meno a heslo
```

GPU ovládač zostáva windowsový, do WSL sa ovládač **neinštaluje**.

Ak `wsl --status` hlási, že virtualizácia nie je zapnutá (hoci v BIOS je), najprv
`wsl --install --no-distribution`, reštart, a až potom inštalácia Ubuntu.

**Hypervisor len podľa potreby.** Virtual Machine Platform zapne Hyper-V hypervisor pri každom
štarte Windowsu (spomaľuje VirtualBox/VMware, niektoré anti-cheaty s ním nefungujú). Samotné
Ubuntu beží len počas používania (`wsl --shutdown` ho hneď vypne). Ak nechceš, aby hypervisor
bežal stále, prepínať ho (PowerShell ako administrátor, po zmene reštart):

```powershell
bcdedit /set hypervisorlaunchtype off    # vypnúť, WSL potom nefunguje
bcdedit /set hypervisorlaunchtype auto   # zapnúť pred prácou s GPU fázami
```

### 3a. CUDA Toolkit a kompilátor (raz, ako root)

```powershell
wsl -d Ubuntu -u root
```

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update
apt-get install -y build-essential python3-venv python3-dev ninja-build cuda-toolkit-13-3
exit
```

Major verzia CUDA Toolkitu (13) musí sedieť s PyTorch wheelom (`cu130`); minor verzia môže byť
vyššia. Na Ubuntu 26.04 (glibc 2.43) **nefunguje CUDA 13.0**: `nvcc` padá na chybe
`exception specification is incompatible ... "rsqrt"`. Overené: CUDA 13.3 + gcc 15.2.

### 3b. Prostredie `sp-gs`

```bash
python3 -m venv ~/venvs/sp-gs && source ~/venvs/sp-gs/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
cd /mnt/c/Users/sismo/Documents/3D_scene_gen_and_edit
pip install -e ".[dev,gs,depth,ingest,sfm]"
```

Ak `pip` nenájde wheel `pycolmap` 4.2.1 pre daný Python, stiahnuť ho ručne z
<https://pypi.org/project/pycolmap/4.2.1/#files> (`cp314 manylinux_2_28_x86_64`) a nainštalovať
zo súboru (`pip install ./pycolmap-4.2.1-...whl`).

Overené verzie: Python 3.14.4, torch 2.14.1+cu130, torchvision 0.29.1+cu130, gsplat 1.5.3,
torchmetrics 1.9.0, transformers 5.20.2, pycolmap 4.2.1 (CUDA Toolkit 13.3, Ubuntu 26.04).

### 3c. Aktivačný skript `~/sp-gs.sh`

Monitor (`configs/local.yaml`, runner `wsl_gs`) ho volá pred každou GPU fázou. Vytvoriť
`~/sp-gs.sh` s obsahom:

```bash
# Activate the sp-gs environment (phases 01, 02, 03): source ~/sp-gs.sh
source ~/venvs/sp-gs/bin/activate
export CUDA_HOME=/usr/local/cuda-13.3
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=7.5      # GTX 1650 = sm_75; iná GPU → iné číslo
export MAX_JOBS=4
cd /mnt/c/Users/sismo/Documents/3D_scene_gen_and_edit
```

### 3d. Prvé spustenie (kompilácia gsplat, ~6 min)

```bash
source ~/sp-gs.sh
pytest tests/contracts/test_p02.py   # skompiluje gsplat do ~/.cache/torch_extensions
pytest
```

Pri prvom behu sa samé stiahnu aj váhy AlexNet pre LPIPS (`~/.cache/torch/hub`) a
Depth Anything V2 Small do `checkpoints/huggingface/` (keď `depth_reg > 0`).

## 4. Overenie celku

```powershell
.venv\Scripts\activate
python run_monitor.py
```

Alebo jedna fáza z príkazového riadka (WSL):

```bash
source ~/sp-gs.sh
python -m scenepipe run 02_reconstruction --scene izba --profile fast
```

## 5. Dáta na vyhodnotenie (nepovinné)

Mip-NeRF 360 (`room`, `counter`, `kitchen`) do `workspace/_datasets/mipnerf360/<scéna>/`:
z `360_v2.zip` (<https://jonbarron.info/mipnerf360/>) stačí pre každú scénu `images_2/`
a `sparse/` (~0,3–0,4 GB na scénu). Licencia a podmienky: [`docs/evaluation.md`](docs/evaluation.md).

## Tipy

- WSL volať z **PowerShellu**, nie z Git Bash (Git Bash prepisuje cesty `/mnt/...`, `/tmp`
  a rozbaľuje `$PREMENNÉ`). Zložitejšie veci zapísať do `.sh` skriptu a spustiť
  `wsl.exe -d Ubuntu -- bash -l /mnt/c/.../skript.sh`.
- `/tmp` vo WSL je tmpfs a zmaže sa pri zastavení distribúcie.
- SQLite na `/mnt/c` je pomalé, fáza 01 preto drží COLMAP databázu v systémovom temp adresári.