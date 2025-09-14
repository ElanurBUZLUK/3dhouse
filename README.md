# 🏠 Sketch2House3D — Çocuk Çiziminden 3D Ev Üretimi

> **Amaç:** Bir çocuk çiziminin fotoğrafını al, **duvar/çatı/kapı/pencere** gibi bileşenleri anla, **parametrik plana** dönüştür, **CSG/Extrude** ile **oyunda kullanılabilir 3D ev (glTF/FBX/USDZ)** üret.

## 🚀 Hızlı Başlangıç

### Kurulum

```bash
# Repository'yi klonlayın
git clone https://github.com/your-username/sketch2house3d.git
cd sketch2house3d

# Python environment oluşturun
conda env create -f environment.yml
conda activate sketch2house3d

# veya pip ile
pip install -r requirements.txt
```

### Docker ile Çalıştırma

```bash
# Docker image oluşturun
docker build -t sketch2house3d:latest .

# Container'ı çalıştırın
docker run --rm -it -v $PWD:/app -p 8000:8000 sketch2house3d:latest
```

### API'yi Başlatma

```bash
# FastAPI server'ı başlatın
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

API dokümantasyonu: http://localhost:8000/docs

### Pipeline'ı Çalıştırma

```bash
# Tek bir çizim işleyin
python -m src.pipeline --input data/raw/sample_001.jpg --output out/sample_001/ --export gltf

# Batch işleme
python scripts/batch_process.py --input_dir data/raw/ --output_dir out/batch/
```

## 📁 Proje Yapısı

```
Sketch2House3D/
├─ data/                    # Veri klasörleri
│  ├─ raw/                 # Ham çizim fotoğrafları
│  ├─ processed/           # İşlenmiş veri
│  └─ 3d_models/           # GT 3D modeller (varsa)
├─ src/                    # Kaynak kod
│  ├─ preprocessing/       # Ön işleme modülleri
│  ├─ models/             # AI modelleri
│  ├─ reconstruction/     # 3D yeniden yapılandırma
│  │  ├─ layout/          # 2D layout çıkarımı
│  │  └─ geometry/        # 3D geometri üretimi
│  ├─ export/             # Export modülleri
│  ├─ api/                # FastAPI backend
│  └─ pipeline.py         # Uçtan uca pipeline
├─ configs/               # Konfigürasyon dosyaları
├─ tests/                 # Test dosyaları
├─ scripts/               # Yardımcı scriptler
└─ docs/                  # Dokümantasyon
```

## 🔧 Özellikler

### AI Segmentasyon
- **U-Net/DeepLabV3+** ile {"bg","wall","roof","opening"} sınıfları
- **Manhattan-snap** ve **px→m** ölçek dönüşümü
- **Constraint-Repair** ile otomatik onarım

### 3D Geometri Üretimi
- **Extrude** ve **robust CSG** ile duvar/kapı/pencere kesimleri
- **Gable/Hip/Shed/Flat** çatı tipleri
- **Mesh kalitesi** ve **LOD** desteği

### Export & Formatlar
- **glTF2** (units=meters), **FBX**, **USDZ**
- **Unity/Blender** round-trip testleri
- **UV mapping** ve **texture baking**

## 📊 API Kullanımı

### Tek Çizim İşleme

```bash
curl -X POST "http://localhost:8000/api/v1/infer" \
  -F "image=@sample_sketch.jpg" \
  -F "backend=open3d" \
  -F "export_format=gltf" \
  -F "quality=medium"
```

### Batch İşleme

```bash
curl -X POST "http://localhost:8000/api/v1/batch-infer" \
  -F "images=@sketch1.jpg" \
  -F "images=@sketch2.jpg" \
  -F "backend=open3d" \
  -F "export_format=gltf"
```

### Durum Kontrolü

```bash
curl "http://localhost:8000/api/v1/status/{request_id}"
```

## 🎯 Kalite Kapıları

- **Segmentation:** mIoU ≥ 0.80, Boundary-F1 ≥ 0.75
- **Layout:** Constraint pass ≥ %98
- **Geometry:** Watertight ≥ %95, CSG fail ≤ %2
- **Export:** glTF 2.0, units=meters, round-trip hatasız
- **Performance:** LOD0 ≤ 50k tris, tek ev üretimi ≤ 2s

## 🧪 Test

```bash
# Tüm testleri çalıştırın
pytest

# Belirli test kategorisi
pytest tests/test_geometry.py
pytest tests/test_export.py
pytest tests/test_api.py
```

## 📈 Performans

- **Tek ev üretimi:** ≤ 2 saniye (desktop GPU)
- **LOD0:** ≤ 50k üçgen
- **LOD1:** ≤ 20k üçgen  
- **LOD2:** ≤ 5k üçgen

## 🛠️ Geliştirme

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

### Code Formatting

```bash
black src/
isort src/
flake8 src/
```

## 📝 Lisans

MIT License - detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📞 İletişim

- **Proje:** [Sketch2House3D](https://github.com/your-username/sketch2house3d)
- **Issues:** [GitHub Issues](https://github.com/your-username/sketch2house3d/issues)
- **Discussions:** [GitHub Discussions](https://github.com/your-username/sketch2house3d/discussions)

## 🙏 Teşekkürler

- **Open3D** - 3D veri işleme
- **PyTorch** - Deep learning framework
- **FastAPI** - Web framework
- **Trimesh** - 3D mesh işleme
- **Segmentation Models PyTorch** - Pre-trained modeller

---

**Not:** Bu proje aktif geliştirme aşamasındadır. Production kullanımı için dikkatli olun.
