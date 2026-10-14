# Nastavenie projektu na novom počítači

Tento súbor je jednorazový návod pre Claude Code. Keď sa na novom počítači spustí session
v tomto repozitári, Claude prejde kroky nižšie, overí ich a na konci súbor zmaže
(aj odkaz naň v `CLAUDE.md`) a commitne to.

Používateľ nechce nič ručne prenášať: dáta scén a váhy modelov sa na nový počítač nekopírujú.

## Stav projektu pri prechode (2026-10-14)

- Hotové fázy: `00_ingest` (basic) a `01_sfm` (colmap, pycolmap 4.2). Obe majú kontraktové testy
  na syntetických dátach (`tests/contracts/test_p00.py`, `test_p01.py`).
- Ďalší krok: fáza `02_reconstruction` (gsplat, potrebuje CUDA GPU).
- `workspace/` a `checkpoints/` na starom počítači neboli naplnené, takže sa nič nestráca.
  Žiadne modely sa zatiaľ nesťahovali.
- Starý počítač: AMD Ryzen 5 7530U, iGPU AMD, 7,4 GB RAM, **bez NVIDIA GPU**, Windows 11,
  Python 3.13.7 v `.venv`. Preto CUDA fázy (02, 04, 06–12) čakali na vzdialenú GPU.

## Kroky pre Claude

1. **Git identita.** Over `git config user.name` a `user.email`. Ak chýbajú, nastav
   (globálne, po potvrdení používateľom): `hajkovsky3` / `simon.hajkovsky@junity.sk`.
   Over, že `git pull` a `git push` na `origin` fungujú (prihlásenie do GitHubu).

2. **Hardvér.** Zisti CPU, RAM, GPU (`nvidia-smi`, ak existuje; na Windowse aj
   `Get-CimInstance Win32_VideoController`), OS a dostupné verzie Pythonu.
   Výsledok ulož do pamäte Claude ako `project` memory (pamäť sa medzi počítačmi neprenáša).
   Ak má počítač NVIDIA GPU s CUDA, povedz to používateľovi: fáza 02 a ďalšie sa dajú
   vyvíjať priamo tu, zapíš to aj do pamäte.

3. **Python prostredie `sp-core`** (fázy 00, 01). Potrebný Python 3.11+ (3.13 funguje).
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate   Linux/macOS: source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -e ".[dev,ingest,sfm]"
   ```
   Na starom počítači bol `pycolmap` 4.2.1 nainštalovaný zo stiahnutého wheelu z PyPI
   (`pycolmap-4.2.1-cp313-cp313-win_amd64.whl`). Ak `pip install` zlyhá, skús rovnaký postup:
   stiahni wheel pre danú verziu Pythonu a OS z PyPI (stránka pycolmap → Download files)
   do scratchpadu a nainštaluj ho. Verzia musí spĺňať `pycolmap>=4.2,<4.3`.
   Pip wheel pycolmap je iba CPU, to stačí.

4. **Overenie.**
   ```bash
   pytest tests/
   ruff check src tests
   ```
   Všetky testy musia prejsť (test_p01 trvá na CPU ~11 s). Ak niečo zlyhá, oprav príčinu
   (prostredie, nie kontrakt) alebo sa opýtaj.

5. **Upratanie.** Zmaž tento súbor a riadok s odkazom naň v `CLAUDE.md`,
   commitni (`Remove one-time new-machine setup notes`) a po potvrdení pushni.

6. **Zhrnutie používateľovi.** Čo bolo nainštalované, výsledok testov, hardvér
   a odporúčaný ďalší krok (fáza 02).

## Bežná práca medzi dvoma počítačmi

- Pred začiatkom práce: `git pull`.
- Po skončení: commit a `git push`.
- Dáta scén (`workspace/`) a váhy (`checkpoints/`) nie sú v gite. Generujú sa na každom
  počítači samostatne, prípadne sa prenášajú mimo gitu, ak to bude treba.
