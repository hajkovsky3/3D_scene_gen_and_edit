# Dátové kontrakty a konvencie

Tento dokument definuje všetko, čo je spoločné viacerým fázam. Konkrétne výstupy jednotlivých
fáz sú v `phases/XX_*.md`, tu sú formáty, na ktoré sa odkazujú.

## 1. Workspace

```
workspace/<scene_id>/
├── input/                      pôvodné fotky / video (nemenia sa)
├── 00_ingest/ … 07_scene_graph/  výstupy build fáz (každá s manifest.json)
├── state/
│   ├── v000/state.json         počiatočný stav (po fáze 07)
│   ├── v001/…                  stav po prvej úprave
│   └── current -> v001         symlink alebo textový súbor s názvom aktuálneho stavu
├── edits/
│   └── e001/
│       ├── prompt.json
│       ├── 08_prompt_parsing/ 09_grounding/ 10_asset_generation/ 11_editing/
│       └── (11 zapíše nový stav do state/v001)
└── renders/
```

- `scene_id`: `[a-z0-9_-]+`.
- Každá fáza zapisuje **iba** do svojho adresára (výnimka: fáza 11 vytvára nový `state/vNNN/`).
- Všetky cesty vo výstupoch a manifestoch sú relatívne k `workspace/<scene_id>/`.

## 2. Manifest fázy

Každý výstupný adresár fázy obsahuje `manifest.json`:

```json
{
  "phase": "01_sfm",
  "impl": "colmap",
  "impl_version": "4.2.0",
  "contract_version": 1,
  "config": { "...": "efektívna konfigurácia po zlúčení defaultov" },
  "inputs":  { "frames": {"path": "00_ingest/frames.json", "sha256": "…"} },
  "outputs": { "cameras": {"path": "01_sfm/cameras.json", "sha256": "…"} },
  "started_at": "2026-10-12T10:00:00Z",
  "finished_at": "2026-10-12T10:07:31Z",
  "git_commit": "abc1234",
  "host": "hostname",
  "notes": []
}
```

Manifest slúži na reprodukovateľnosť (do práce sa dá presne uviesť, čím vznikol výsledok)
a na cache: ak sa hashe vstupov a konfigurácia nezmenili, fázu netreba púšťať znova.

## 3. Identifikátory

| ID | Tvar | Kto ho tvorí | Poznámka |
|---|---|---|---|
| `frame_id` | `f000123` | 00 | stabilné počas celého reťazca, poradie = poradie snímok |
| `mask_id` | `f000123_m007` | 04 | unikátne v scéne |
| `track_id` | `t0042` alebo `null` | 04 | rovnaký objekt naprieč snímkami, ak ho segmentátor vie sledovať |
| `instance_id` | int ≥ 0, `-1` = pozadie | 06 | 3D inštancia, index do `instances.json` |
| `node_id` | `n0042` | 07 | uzol grafu, viaže sa 1:1 na `instance_id` pri vzniku |
| `op_id` | `op01` | 08 | operácia v pláne úpravy |

Po úprave (11) sa `instance_id` a `node_id` existujúcich objektov **nemenia**. Nové objekty
dostanú nové ID (max + 1), zmazané ID sa znovu nepoužívajú.

## 4. Súradnicové systémy

- **Kamera**: OpenCV konvencia — x doprava, y dole, z dopredu (do scény).
- **Extrinsika**: `world_to_camera` ako 4×4 matica (rotácia + translácia), riadkovo v JSON.
- **Intrinsika**: `fx, fy, cx, cy` v pixeloch pre rozlíšenie snímky uvedené v `cameras.json`,
  model `PINHOLE` (snímky sú po fáze 01 undistortované).
- **Svet**: po fáze 01 je scéna **zarovnaná s gravitáciou: +Z hore**. Mierka je ľubovoľná
  (SfM nepozná metre), ale je normalizovaná tak, aby rozptyl pozícií kamier bol približne 1.
  Transformácia z pôvodného SfM rámca je uložená v `01_sfm/alignment.json`.
- **Kanonický asset** (fáza 10): stred spodnej podstavy bounding boxu v počiatku, +Z hore,
  najdlhší rozmer bounding boxu = 1, predná strana objektu do −Y (ak sa dá určiť).

## 5. Formát gaussiánov (`*.ply`)

Binárny little-endian PLY, jeden element `vertex` s N záznamami, vlastnosti `float32`
v tomto poradí (de facto štandard 3DGS, kompatibilný s viewermi ako SuperSplat):

| Vlastnosť | Význam | Uloženie |
|---|---|---|
| `x, y, z` | stred | svetové súradnice |
| `nx, ny, nz` | nepoužité | nuly (kvôli kompatibilite) |
| `f_dc_0..2` | SH stupňa 0 (farba) | koeficienty, nie RGB |
| `f_rest_0..44` | SH stupňov 1–3 | 15 koeficientov × 3 kanály, kanálovo usporiadané |
| `opacity` | priehľadnosť | **logit** (pred sigmoidou) |
| `scale_0..2` | mierka osí | **log** (pred exp) |
| `rot_0..3` | rotácia | kvaternión **wxyz**, nemusí byť normalizovaný |

Prevod SH DC ↔ RGB: `rgb = 0.5 + C0 * f_dc`, kde `C0 = 0.28209479177387814`.
Ak implementácia používa nižší stupeň SH, chýbajúce `f_rest_*` sa doplnia nulami
(počet `f_rest` je v manifeste ako `sh_degree`).

Poradie gaussiánov v súbore je **index gaussiánu**. Všetky per-gaussián polia
(`gaussian_instances.npz`) sa viažu na tento index, preto fáza 11 pri mazaní a pridávaní
gaussiánov musí rovnako preusporiadať aj ich.

## 6. Kamery (`cameras.json`)

```json
{
  "contract_version": 1,
  "convention": "opencv_world_to_camera",
  "up_axis": "+z",
  "cameras": [
    {
      "frame_id": "f000001",
      "image": "01_sfm/images_undistorted/f000001.jpg",
      "width": 1600, "height": 1200,
      "fx": 1234.5, "fy": 1234.5, "cx": 800.0, "cy": 600.0,
      "world_to_camera": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    }
  ]
}
```

Snímky, ktoré SfM nezaregistrovalo, v zozname nie sú. Ďalšie fázy pracujú iba s nimi.

## 7. Masky

Masky sa ukladajú ako **COCO RLE** (kompaktné, prekryvy povolené), čítané/zapisované cez
`pycocotools` (BSD-2) alebo vlastnú implementáciu v `core/masks.py`. Každá maska má rozmer
snímky z `cameras.json` (undistortovanej), nie pôvodnej fotky.

## 8. Embeddingy

- Uložené ako `float16`, **L2-normalizované**, tvar `[M, D]`.
- Vedľa vždy `meta.json` s `model` (napr. `open_clip:ViT-H-14/laion2b_s32b_b79k`), `dim`
  a `normalized: true`.
- Fáza 09 musí na text použiť **ten istý model**, ktorý je uvedený v `meta.json` fázy 05.
  Ak sa model zmení, treba znova spustiť 05, 06, 07.

## 9. Stav scény (`state/vNNN/state.json`)

```json
{
  "version": "v001",
  "parent": "v000",
  "created_by": "edits/e001/11_editing/manifest.json",
  "gaussians": "edits/e001/11_editing/gaussians.ply",
  "gaussian_instances": "edits/e001/11_editing/gaussian_instances.npz",
  "scene_graph": "edits/e001/11_editing/scene_graph.json",
  "node_features": "edits/e001/11_editing/node_features.npy",
  "embedding_model": "open_clip:ViT-H-14/laion2b_s32b_b79k",
  "cameras": "01_sfm/cameras.json"
}
```

Stav je len **ukazovateľ** na súbory, nič sa nekopíruje. `v000` vytvorí runner po fáze 07
(ukazuje na výstupy 02, 06, 07). Vďaka tomu sa dá ľubovoľne vrátiť k staršiemu stavu
a porovnať pred a po úprave.

## 10. Prompt (`edits/eNNN/prompt.json`)

```json
{
  "text": "vymeň hnedý gauč za modrý stôl",
  "images": [ {"path": "edits/e001/ref.jpg", "role": "auto"} ],
  "base_state": "v000",
  "interactive": false
}
```

`role` obrázka: `target` (ukazuje, ktorý objekt upraviť), `asset` (čo vložiť / čím nahradiť),
`style` (referencia štýlu) alebo `auto` (rozhodne fáza 08).
