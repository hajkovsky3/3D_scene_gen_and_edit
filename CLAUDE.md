# CLAUDE.md

Tento súbor číta Claude Code na začiatku každej session. Popisuje projekt, pravidlá a spôsob práce.

## O projekte

Bakalárska práca (FMFI UK, Aplikovaná informatika, školiteľ doc. RNDr. Martin Madaras, PhD.):
**Generovanie 3D scén na základe textových promptov.**

Reťazec (pipeline) z bežných fotiek alebo videa reálnej scény vytvorí fotorealistickú 3D scénu
v reprezentácii 3D Gaussian Splatting a k nej graf scény (scene graph). Scénu potom upravuje
textovými a obrázkovými promptmi (zmena farby, odstránenie, presun, výmena, vloženie objektu, štýl).

Reťazec má dve časti:

- **Build** (fázy 00–07): spustí sa raz pre scénu. Výsledkom je počiatočný stav scény `state/v000`.
- **Edit loop** (fázy 08–12): spustí sa pre každý prompt. Zo stavu `vN` vytvorí stav `vN+1`.

Prehľad fáz, DAG závislostí a tabuľka vstupov a výstupov: `phases/README.md`.
Formáty súborov a konvencie (súradnice, ID, stav scény): `docs/data-contracts.md`.

## Najdôležitejšie pravidlo: modularita

Každá fáza je samostatný modul s jasným kontraktom **vstupné súbory → výstupné súbory**.

1. **Fázy komunikujú iba cez súbory na disku** v adresári `workspace/<scene_id>/`.
   Fáza nikdy neimportuje kód inej fázy. Zdieľaný kód patrí do `src/scenepipe/core/`.
2. **Kontrakt je posvätný.** Formát výstupu fázy je definovaný v `phases/XX_*.md` a v
   `docs/data-contracts.md`. Ak treba kontrakt zmeniť, najprv uprav dokumentáciu,
   potom kód všetkých dotknutých fáz, a zvýš `contract_version` v príslušnom `.md`.
3. **Každá fáza môže mať viac implementácií** (napr. `01_sfm` → `colmap`, `vggt`).
   Všetky implementácie jednej fázy produkujú výstup v tom istom formáte a prechádzajú tými istými
   kontraktovými testami. Implementácia sa vyberá v `configs/default.yaml` alebo cez CLI.
4. **Každá fáza zapíše `manifest.json`** do svojho výstupného adresára (pozri `core/phase.py`):
   názov fázy, implementácia, verzia, konfigurácia, vstupy s hashmi, výstupy, čas behu, git commit.
5. **Každá fáza validuje svoj výstup** metódou `validate_outputs()` ešte pred zápisom manifestu.
6. Fázy môžu bežať v **rôznych Python prostrediach** (konfliktné závislosti: SAM, gsplat, TRELLIS,
   VLM). Preto žiadne zdieľanie objektov v pamäti medzi fázami, iba súbory.

## Licenčné pravidlo

Všetky použité moduly, knižnice a váhy modelov musia byť **free to use a open source**.

- Povolené: MIT, BSD, Apache 2.0 a podobné permisívne licencie.
- Zakázané: nekomerčné a výskumné licencie (INRIA Gaussian-Splatting License, S-Lab License,
  CC BY-NC, FAIR Noncommercial), licencie s územným vylúčením EÚ (napr. Hunyuan3D).
- Hraničné (vlastné licencie bez OSI schválenia, napr. SAM License, VGGT License, OpenRAIL-M):
  len po výslovnom súhlase autora a so záznamom v `docs/licenses.md`.
- **Pozor na tranzitívne závislosti.** Napríklad `diff-gaussian-rasterization` (INRIA) nepoužívať,
  na rasterizáciu gaussiánov vždy `gsplat` (Apache 2.0).
- Pred pridaním novej závislosti alebo modelu over licenciu kódu **aj váh** a zapíš ju do
  `docs/licenses.md`. Ak si nie si istý, opýtaj sa.

## Jazyk a štýl

- Dokumentácia (`*.md`) po slovensky.
- Kód, názvy premenných, docstringy, komentáre a commit správy po anglicky.
- Python 3.11+ (fáza so SAM 3 vyžaduje 3.12), type hints všade, `ruff` na formátovanie a lint.
- Numerické polia ukladať ako `.npy`/`.npz`, štruktúrované dáta ako JSON (UTF-8, `indent=2`).
- Žiadne absolútne cesty v manifestoch ani vo výstupoch; cesty vždy relatívne k `workspace/<scene_id>/`.

## Štruktúra repozitára

```
CLAUDE.md                 tento súbor
README.md                 krátky úvod a inštalácia
phases/                   podrobná špecifikácia každej fázy (zdroj pravdy pre kontrakty)
docs/data-contracts.md    spoločné formáty, súradnicové systémy, stav scény
docs/licenses.md          licencie všetkých závislostí a modelov
docs/evaluation.md        metriky a datasety na vyhodnotenie
configs/default.yaml      výber implementácie a parametre pre každú fázu
envs/                     definície prostredí (conda/uv) podľa skupín fáz
src/scenepipe/core/       zdieľaný kód: Phase rozhranie, manifest, workspace, I/O formátov
src/scenepipe/phases/     jeden balík na fázu: pXX_name/<implementacia>.py
tests/contracts/          kontraktové testy výstupov fáz
workspace/                dáta scén (gitignored)
checkpoints/              váhy modelov (gitignored)
```

## Ako pracovať na fáze (workflow)

Fázy sa programujú **po jednej**. Pri práci na fáze XX:

1. Prečítaj `phases/XX_*.md` celé, vrátane sekcií Kontrakt, Validácia a Riziká,
   a relevantné časti `docs/data-contracts.md`.
2. Ak vstupy fázy ešte nevie vyprodukovať predchádzajúca fáza, priprav malý syntetický alebo
   ručne vytvorený vstup v `tests/fixtures/` (nie do `workspace/`).
3. Implementuj v `src/scenepipe/phases/pXX_name/<impl>.py` ako podtriedu `Phase`.
4. Napíš kontraktový test v `tests/contracts/test_pXX.py` (overuje formát výstupu,
   nie kvalitu) a podľa potreby jednotkové testy.
5. Over na malej scéne (nízke rozlíšenie, málo snímok), potom na reálnej.
6. Aktualizuj sekciu **Stav** v `phases/XX_*.md` (checklist) a prípadne `docs/licenses.md`.

Pri neistote ohľadom návrhu (algoritmus, formát, výber modelu) sa radšej opýtaj, než aby si
potichu zmenil kontrakt alebo pridal ťažkú závislosť.

## Príkazy

```bash
# spustenie jednej fázy (CLI zatiaľ nie je implementované, cieľový tvar):
python -m scenepipe run 01_sfm --scene room --impl colmap
# celý build:
python -m scenepipe build --scene room
# jedna úprava:
python -m scenepipe edit --scene room --prompt "zmeň farbu gauča na modrú"
# testy a lint:
pytest tests/
ruff check src tests && ruff format src tests
```

## Hardvér

Vývojový počítač je slabší, prístup k výkonnejšej GPU sa ešte rieši. Preto:

- každá fáza musí mať parameter na zníženie nárokov (rozlíšenie, počet snímok, veľkosť modelu),
- ťažké fázy (02, 04, 07 captioning, 08, 10, 11) písať tak, aby sa dali spustiť samostatne
  na inom stroji: stačí skopírovať vstupné súbory a výstupy vrátiť späť,
- odhadované nároky na VRAM sú v každom `phases/XX_*.md`.

## Čo nerobiť

- Necommitovať dáta scén, váhy modelov, `*.ply`, `*.npz`, renderované videá.
- Nesťahovať modely ani datasety bez opýtania (veľkosť, licencia).
- Nekopírovať kód z repozitárov s nekompatibilnou licenciou, ani „inšpirovať sa“ doslovne.
- Nemeniť kontrakt fázy len v kóde bez úpravy jej `.md`.
