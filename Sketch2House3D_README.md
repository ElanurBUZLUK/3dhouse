# 🏠 Sketch2House3D — Çocuk Çiziminden 3D Ev Üretimi (End‑to‑End)

> **Amaç:** Bir çocuk çiziminin fotoğrafını al, **duvar/çatı/kapı/pencere** gibi bileşenleri anla, **parametrik plana** dönüştür, **CSG/Extrude** ile **oyunda kullanılabilir 3D ev (glTF/FBX/USDZ)** üret.

---

## İçindekiler
- [Özellikler](#özellikler)
- [Mimari & Akış](#mimari--akış)
- [Klasör Yapısı](#klasör-yapısı)
- [Kurulum](#kurulum)
- [Veri & Etiketleme](#veri--etiketleme)
- [Model Eğitimi (Segmentasyon)](#model-eğitimi-segmentasyon)
- [Layout (2D → Parametrik Plan)](#layout-2d--parametrik-plan)
- [Constraint-Repair](#constraint-repair)
- [3D Geometri Üretimi](#3d-geometri-üretimi)
- [Mesh İşlemleri (Kalite/LOD)](#mesh-işlemleri-kalitelod)
- [Malzeme & UV & Doku](#malzeme--uv--doku)
- [Backends](#backends)
- [Export (glTF/FBX/USDZ) & Round‑Trip](#export-gltffbxusdz--roundtrip)
- [Uçtan Uca Pipeline Kullanımı](#uçtan-uca-pipeline-kullanımı)
- [Kalite Kapıları (Definition of Done)](#kalite-kapıları-definition-of-done)
- [Test Paketi](#test-paketi)
- [Konfigürasyonlar](#konfigürasyonlar)
- [API & UI](#api--ui)
- [Performans Hedefleri](#performans-hedefleri)
- [Roadmap](#roadmap)
- [Lisans & Katkı](#lisans--katkı)

---

## Özellikler
- **AI Segmentasyon:** U‑Net/DeepLabV3+ ile {"bg","wall","roof","opening"} sınıfları.
- **Layout Solver:** Manhattan‑snap, px→m ölçek, çatı parametreleri, kapı/pencere bant kısıtları.
- **Constraint‑Repair:** `opening ⊂ wall`, `door touches ground`, `roof on top` otomatik onarım.
- **3D Üretim:** Extrude, **robust CSG** ile kapı/pencere kesimleri, gable/hip/shed/flat çatı.
- **Mesh Kalitesi:** triangulation, cleanup, watertight doğrulama, LOD/decimation.
- **UV & Doku:** Planar unwrap, **çizim dokusunu UV’ye bake**, PBR presetleri.
- **Backend Seçimi:** Open3D/Trimesh (hızlı) veya Blender (üretim kalitesi).
- **Export & Round‑Trip:** glTF2 (units=meters), FBX/USDZ; Unity/Blender round‑trip testleri.
- **MLOps:** Çalıştırma izleme, model versiyonlama (opsiyonel).

---

## Mimari & Akış

### Genel Mimari
![Architecture](docs/architecture.png)

*(ChatGPT içinden görüntülemek için: [sandbox:/mnt/data/docs/architecture.png](sandbox:/mnt/data/docs/architecture.png))*

### Kısa Akış — “Sketch → glTF”
![Sketch→glTF](docs/sketch_to_gltf.png)

*(ChatGPT içinden görüntülemek için: [sandbox:/mnt/data/docs/sketch_to_gltf.png](sandbox:/mnt/data/docs/sketch_to_gltf.png))*

---

## Klasör Yapısı
```text
Sketch2House3D/
├─ data/
│  ├─ raw/            # çizim fotoğrafları
│  ├─ interim/        # ön işleme çıktıları
│  ├─ processed/      # segment/normalize edilmiş veri
│  ├─ augment/        # augmentation çıktıları
│  ├─ 3d_models/      # GT 3D (varsa)
│  └─ metadata/       # açıklama JSON/YAML
├─ notebooks/         # araştırma & prototip
├─ src/
│  ├─ preprocessing/  # image_normalizer.py, edge_detection.py, shape_detector.py, perspective_estimator.py
│  ├─ models/         # segmentation_model.py, depth_model.py, completion_model.py, trainer.py
│  ├─ reconstruction/
│  │  ├─ layout/      # facade_solver.py, opening_solver.py, roof_params.py, constraints.py, repair.py
│  │  ├─ geometry/    # primitives.py, sweep_extrude.py, walls_builder.py, openings_boolean.py, floor_foundation.py
│  │  ├─ geometry/roof_builder/  # gable.py, hip.py, shed.py, flat.py
│  │  ├─ meshops/     # triangulation.py, remesh_cleanup.py, normals_tangents.py, decimation_lod.py, watertight_check.py
│  │  ├─ materials_uv/# uv_unwrap.py, texture_bake.py, pbr_materials.py, texel_density_check.py
│  │  ├─ neural_implicit/ (ops.) # sdf_field.py, marching_cubes.py
│  │  ├─ validators/  # scale_units.py, structure_plausibility.py
│  │  └─ builder.py   # yüksek seviye inşa
│  ├─ backends/       # open3d_backend.py, blender_backend.py
│  ├─ export/         # gltf_exporter.py, usdz_exporter.py, fbx_exporter.py
│  ├─ api/            # main.py, routes.py, schemas.py, services.py
│  ├─ utils/          # dataset_loader.py, visualization.py, logger.py, metrics.py, seed_everywhere.py
│  └─ pipeline.py     # uçtan uca orkestrasyon
├─ tests/             # unit, integration, e2e
├─ scripts/           # gen_synthetic_sketches.py, train_model.sh, export_demo.sh
├─ configs/           # training_config.yaml, geometry_tolerances.yaml, perf_budgets.yaml
├─ docs/              # architecture.png, sketch_to_gltf.png, usage.md, project_plan.md
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml
├─ README.md
└─ LICENSE
```

---

## Kurulum

### 1) Ortam
```bash
# Python env
conda env create -f environment.yml
conda activate sketch2house3d

# veya
pip install -r requirements.txt
```

### 2) Docker (Blender headless + Python)
```bash
docker build -t sketch2house3d:latest .
docker run --rm -it -v $PWD:/app sketch2house3d:latest
```

### 3) Pre-commit & Lint
```bash
pre-commit install
pre-commit run --all-files
```

---

## Veri & Etiketleme

### Toplama ve Split
- En az **500** çocuk çizimi (çeşitlilik: perspektif, stil, detay seviyesi).  
- Train/Val/Test = **70/15/15**.

### Etiketleme Sınıfları
- `bg`, `wall`, `roof`, `opening` (kapı+pencere). İstersen kapı/pencere ayrı sınıf.

### Sentetik Üretici
```bash
python scripts/gen_synthetic_sketches.py   --n 5000 --styles pencil,marker,crayon   --jitter 0.5 --missing-strokes 0.3 --closed-contour-prob 0.7
```

### Augmentations
- blur, elastic, color jitter, stroke thick/thin, random gaps.

---

## Model Eğitimi (Segmentasyon)
```bash
python -m src.models.trainer   --config configs/training_config.yaml   --dataset data/processed   --out runs/seg_unet_v1
```

**Öneriler**
- Mimari: U‑Net (encoder=ResNet18) **veya** DeepLabV3+ (MobileNetV3)
- Loss: CE(class weights) + Dice, Label smoothing=0.05
- Calibration: temperature scaling
- Belirsizlik: MC‑Dropout (inference’te n=10) → layout’a confidence aktar

**Metrikler**: mIoU, Boundary‑F1, class‑wise IoU.

---

## Layout (2D → Parametrik Plan)
- `facade_solver.py`: dış kontur, **Manhattan‑snap**, **px→m** (kapı ≈ 2.0 m ⇒ ölçek).
- `opening_solver.py`: pencere/kapı bant kısıtları (örn. pencere yüksekliği [0.3H, 0.8H]).
- `roof_params.py`: çatı tipi (gable/hip/…), **pitch**, **overhang**.
- `constraints.py`: `opening ⊂ wall`, `roof ∩ opening = ∅`, `door touches ground`.
- Şema: `schemas/layout_params.schema.json` (ara veri kontratı).

---

## Constraint‑Repair
- `repair.py`: ILP/greedy ile minimum düzenleme; çakışan pencereler için NMS.
- Metrikler: **constraint‑satisfaction rate**, min‑edit distance.

---

## 3D Geometri Üretimi
- `sweep_extrude.py`: 2D profil → extrude/loft.
- `walls_builder.py`: kalın duvar (t), yükseklik (H), köşelerde birleşim.
- `openings_boolean.py`: CSG difference ile kapı/pencere delikleri (**numerik tolerans**: `configs/geometry_tolerances.yaml`).
- Çatılar:
  - `gable.py`: beşik çatı (ridge line, pitch)
  - `hip.py`: kırma çatı
  - `shed.py`: tek eğimli çatı
  - `flat.py`: düz + parapet

---

## Mesh İşlemleri (Kalite/LOD)
- `triangulation.py`: earcut / CDT
- `remesh_cleanup.py`: non‑manifold/self‑intersection fix
- `normals_tangents.py`: düzgün aydınlatma
- `watertight_check.py`: manifold/watertight rapor
- `decimation_lod.py`: LOD0/1/2 üretimi → `configs/perf_budgets.yaml`

---

## Malzeme & UV & Doku
- `uv_unwrap.py`: planar per‑facade; adaları paketle (overlap=0)
- `texture_bake.py`: **çizim dokusunu UV’ye projeksiyon & bake**
- `pbr_materials.py`: glTF PBR presetleri (baseColor, metallicRoughness, occlusion, emissive)
- `texel_density_check.py`: 256–512 px/m hedefi

---

## Backends
- `open3d_backend.py`: hızlı kurulum, Python‑yalın; bazı CSG köşelerinde kırılgan olabilir.
- `blender_backend.py`: **bmesh + exact boolean + smart UV + bake** (üretim kalitesi).
- Strategy pattern ile tek arayüz: `merge_meshes`, `boolean_diff`, `unwrap`, `bake`…

---

## Export (glTF/FBX/USDZ) & Round‑Trip
```bash
python -m src.export.gltf_exporter --input out/house.fbx --out out/house.gltf
```
- glTF2 (units=**meters**, right‑handed)
- Round‑trip CI: glTF → Unity/Blender import → **ölçek**, **normal yönü**, **üçgen sayısı** assert.

---

## Uçtan Uca Pipeline Kullanımı
```bash
python -m src.pipeline   --input data/raw/sample_001.jpg   --backend open3d   --out out/sample_001/   --export gltf
```
**Çıktılar:** `house.gltf`, `textures/`, `report.json` (metrikler & log).

**Fallback:** `pipeline_fallback.py` (kural‑tabanlı yol) feature‑flag ile otomatik devreye girer.

---

## Kalite Kapıları (Definition of Done)
- **Segmentation:** mIoU ≥ **0.80**, Boundary‑F1 ≥ **0.75**
- **Layout:** Constraint pass ≥ **%98**; `layout_params.json` şemaya **%100** uyum
- **Geometry:** Watertight ≥ **%95**; CSG fail ≤ **%2**; normal yönü hatası ≤ **%1**
- **UV/Materials:** UV overlap = **0**; texel density **256–512 px/m**
- **Export:** glTF 2.0; **units=meters**; Unity/Blender **round‑trip hatasız**
- **Performance:** LOD0 ≤ **50k tris**; LOD1 ≤ **20k**; LOD2 ≤ **5k**; tek ev üretimi ≤ **2 s** (desktop GPU)

---

## Test Paketi
```bash
pytest -q
```
- `test_layout_constraints.py`, `test_csg_boolean.py`, `test_watertight.py`
- `test_export_gltf.py` (round‑trip), `test_uv_overlap.py`
- `test_lod_budget.py`, `test_numeric_tolerances.py`
- **Golden set** E2E: 10 çizim → beklenen metrikler

---

## Konfigürasyonlar

### `configs/training_config.yaml` (örnek)
```yaml
model: deeplabv3plus
encoder: mobilenetv3
input_size: [256, 256]
classes: ["bg", "wall", "roof", "opening"]
loss:
  type: ce_dice
  class_weights: [0.1, 0.4, 0.3, 0.2]
  label_smoothing: 0.05
optimizer: adamw
lr: 0.001
sched: cosine
augment:
  sketch_style: true
  blur: true
  elastic: true
  stroke_jitter: true
train:
  batch_size: 16
  epochs: 80
  mixed_precision: true
```

### `configs/geometry_tolerances.yaml` (örnek)
```yaml
units: meters
epsilon: 1e-5
weld_distance: 1e-4
min_wall_thickness: 0.12
door_height_m: 2.0
window_height_ratio: [0.3, 0.8]
```

### `schemas/layout_params.schema.json` (özet)
```json
{
  "type":"object",
  "properties":{
    "scale_m_per_px":{"type":"number"},
    "facade_polygon":{"type":"array"},
    "openings":{"type":"array","items":{"type":"object","properties":{"kind":{"enum":["door","window"]},"bbox_px":{"type":"array","minItems":4,"maxItems":4}}}},
    "roof":{"type":"object","properties":{"type":{"enum":["gable","hip","shed","flat"]},"pitch_deg":{"type":"number"},"overhang_m":{"type":"number"}}}
  },
  "required":["scale_m_per_px","facade_polygon","openings","roof"]
}
```

---

## API & UI
- **FastAPI**: `/infer` (image upload) → `house.gltf` + `report.json`
- Web preview: **three.js** ile 3D önizleme, parametre ayar paneli (çatı tipi, kapı genişliği).

---

## Performans Hedefleri
- Tek ev üretimi ≤ **2 saniye** (desktop GPU)  
- Toplu işleme desteği (batch) ve caching (ara mesh/texture atlas)

---

## Roadmap
- Hip/Shed/Flat çatı implementasyonları
- USDZ export (iOS AR)
- Neural implicit (SDF) ile kalite yükseltme (opsiyon)
- AR önizleme (WebXR)
- A/B testleri (farklı repair stratejileri)

---

## Lisans & Katkı
- **LICENSE:** MIT (öneri)
- **Katkı:** `CONTRIBUTING.md` + `CODE_OF_CONDUCT.md`
- PR’larda **lint/test/round‑trip** CI zorunlu.

---

> Diyagramlar:  
> • Architecture: [sandbox:/mnt/data/docs/architecture.png](sandbox:/mnt/data/docs/architecture.png)  
> • Sketch→glTF: [sandbox:/mnt/data/docs/sketch_to_gltf.png](sandbox:/mnt/data/docs/sketch_to_gltf.png)
