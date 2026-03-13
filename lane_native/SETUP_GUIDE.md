# 🛠️ lane_native — C++ Hızlandırılmış Şerit Algılama Kurulum Kılavuzu

Bu kılavuz, `lane_native` C++ modülünün **Raspberry Pi** (aarch64) üzerinde kurulumunu adım adım anlatır.

---

## 📋 Gereksinimler

| Paket | Açıklama |
|-------|----------|
| `build-essential` | gcc/g++ derleyici |
| `cmake` | Build sistemi (≥3.14) |
| `python3-dev` | Python C API headers |
| `pybind11-dev` | Python↔C++ köprüsü |
| `libopencv-dev` | OpenCV C++ kütüphanesi |
| `python3-pybind11` | pybind11 Python paketi |

---

## 1️⃣ Sistem Paketlerini Kur

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    python3-dev \
    pybind11-dev \
    libopencv-dev \
    python3-pybind11
```

> **Not:** Raspberry Pi OS (Bookworm) üzerinde bu paketler varsayılan repo'da mevcuttur.

---

## 2️⃣ pip ile pybind11 Kur (CMake desteği için)

```bash
pip3 install pybind11
```

Bu adım, `setup.py`'nin CMake'e pybind11 yolunu otomatik geçirmesi için gereklidir.

---

## 3️⃣ Modülü Derle ve Kur

```bash
cd ~/Desktop/HAILO8L_OTONOM_CODES/lane_native
pip3 install .
```

Bu komut:
1. CMake ile `lane_native.cpp` dosyasını derler (`-O3` optimizasyonuyla)
2. Oluşan `.so` dosyasını Python'ın site-packages'ına yükler
3. `import lane_native` artık çalışır

> **Süre:** İlk build ~1-2 dakika (Raspberry Pi 4'te). Sonraki build'ler çok daha hızlı.

---

## 4️⃣ Kurulumu Test Et

```bash
python3 -c "import lane_native; print('✅ lane_native modülü hazır!')"
```

Eğer hata alıyorsan:

| Hata | Çözüm |
|------|-------|
| `ModuleNotFoundError: No module named 'pybind11'` | `pip3 install pybind11` |
| `Could NOT find OpenCV` | `sudo apt install libopencv-dev` |
| `Could NOT find pybind11` | `sudo apt install pybind11-dev` |
| `cmake: command not found` | `sudo apt install cmake` |
| `fatal error: Python.h` | `sudo apt install python3-dev` |

---

## 5️⃣ Kullanım

`lane_opencv.py` otomatik olarak `lane_native` modülünü kullanır. Eğer modül kurulu değilse, Python fallback ile çalışmaya devam eder.

```
# C++ aktif ise konsolda göreceksin:
[INFO] ✅ lane_native C++ modülü aktif — FPS artışı bekleniyor.

# Kurulu değilse:
[INFO] lane_native bulunamadı, Python fallback kullanılıyor.
```

---

## 6️⃣ Yeniden Derleme (Değişiklik Sonrası)

`lane_native.cpp` dosyasında değişiklik yaparsan:

```bash
cd ~/Desktop/HAILO8L_OTONOM_CODES/lane_native
pip3 install . --force-reinstall --no-deps
```

---

## 7️⃣ Kaldırma

```bash
pip3 uninstall lane_native
```

---

## ⚡ Beklenen Performans

| Mod | Tahmini FPS (640×360 crop, Pi4) |
|-----|--------------------------------|
| Pure Python | ~12-18 FPS |
| C++ Native | ~25-40 FPS |

Artış nedenleri:
- OpenCV fonksiyonları Python→C++ wrapper overhead'ı olmadan doğrudan çağrılır
- Segment loop'ları (filter, cluster, fit) native hızda koşar
- NumPy ara-buffer kopyaları elimine edilir
- Tüm pipeline tek C++ çağrısında tamamlanır (6 ayrı Python fonksiyon çağrısı yerine)

---

## 🔧 Manuel Build (pip olmadan)

```bash
cd ~/Desktop/HAILO8L_OTONOM_CODES/lane_native
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# .so dosyasını proje dizinine kopyala
cp lane_native*.so ../../
```

Bu durumda `lane_native*.so` dosyası `lane_opencv.py` ile aynı dizinde olmalı.
