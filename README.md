# 📦 Smart Sack Counter & Analysis System

Sistem cerdas berbasis Computer Vision dan Depth Sensing untuk mengotomatisasi perhitungan stok karung di lingkungan industri/gudang. Sistem ini mampu menghitung karung dalam kondisi **diam (tumpukan/stapel)** maupun **bergerak (konveyor)** menggunakan kamera Intel RealSense dan algoritma YOLO.

---

## 🚀 Fitur Utama

* **Analisis Stapel Cerdas:** Mendeteksi pola tumpukan (Duo/Trio) dan menghitung estimasi total karung berdasarkan ketinggian tumpukan menggunakan sensor kedalaman.
* **Filter Kedalaman Dinamis:** Menggunakan algoritma variansi kedalaman dan gradien untuk membedakan karung asli dari noise atau objek latar belakang.
* **Penghitung Real-Time:** Menghitung karung yang lewat di konveyor dengan logika *line crossing* (Gate A/B) dua arah.
* **Kalibrasi Visual:** Tool GUI untuk mengatur posisi garis hitung secara presisi tanpa menyentuh kode.

---

## ⚙️ Persyaratan Sistem

### Perangkat Keras (Hardware)

* **Kamera:** Intel RealSense Depth Camera (D435, D435i, atau D455).
* **Koneksi:** Kabel USB 3.0 (Wajib untuk streaming Depth & RGB simultan).
* **Komputer:**
* CPU: Min. Intel Core i5 / AMD Ryzen 5.
* GPU: Disarankan NVIDIA GTX/RTX (untuk performa YOLO optimal).



### Perangkat Lunak (Software)

* OS: Windows 10/11 atau Ubuntu 20.04+.
* Python 3.8 - 3.10.
* Driver Intel RealSense SDK 2.0.

---

## 📥 Instalasi

1. **Clone Repository ini:**
```bash
https://github.com/projectkita-id/Bulog---Karung.git

```


2. **Install Library Python yang dibutuhkan:**
```bash
pip install opencv-python numpy pyrealsense2 ultralytics

```


3. **Siapkan Model YOLO:**
Pastikan file model `.pt` (contoh: `lastbaru.pt` atau `bestbaru.pt`) sudah berada di dalam folder root proyek.

---

## 📖 Panduan Penggunaan

### 1. Analisis Tumpukan (Static Stack Analysis)

Modul ini digunakan untuk menghitung stok karung yang ditumpuk diam di gudang.

* **File:** `cek_stapel.py`
* **Cara Menjalankan:**
```bash
python cek_stapel.py

```


* **Logika Kerja:**
Sistem mendeteksi susunan "Duo" (2 karung) dan "Trio" (3 karung). Jika terdeteksi tumpukan vertikal (stapel), sistem menghitung ketinggian rata-rata menggunakan RMS (*Root Mean Square*) dari data depth, lalu mengonversinya menjadi jumlah tingkat dan total karung.
* **Kontrol:**
* `s`: Simpan screenshot hasil deteksi.
* `q`: Keluar aplikasi.



### 2. Kalibrasi Garis (Calibration Tool)

Gunakan modul ini **sebelum** menjalankan penghitung jalur untuk menentukan posisi garis hitung (Gate) yang pas.

* **File:** `Linecalibration.py`
* **Cara Menjalankan:**
```bash
python Linecalibration.py

```


* **Fitur:**
Gunakan *slider/trackbar* yang muncul untuk menggeser posisi garis vertikal (X) dan batas ketinggian (Y) untuk **Gate A** dan **Gate B**. Tekan `s` untuk menyimpan konfigurasi ke file `kalibrasi.txt`.

### 3. Penghitung Jalur (Conveyor Tracking)

Modul untuk menghitung karung berjalan di konveyor secara otomatis.

* **File:** `tracking_new.py`
* **Cara Menjalankan:**
```bash
python tracking_new.py

```


* **Logika Kerja:**
Sistem melacak ID unik setiap objek. Perhitungan bertambah ketika titik tengah objek melewati garis yang telah dikalibrasi (Gate A atau Gate B).

---

## 🔧 Konfigurasi & Tuning

Anda dapat menyesuaikan variabel di bagian atas setiap script untuk hasil yang lebih akurat sesuai kondisi lapangan.

**Pada `cek_stapel.py`:**

```python
MODEL_PATH = "lastbaru.pt"       # Path ke model YOLO
CONF_THRESH = 0.1                # Sensitivitas deteksi
STAPEL_TWO_LEVELS_THRESH = 4.10  # Tinggi kamera dari lantai (Meter) - KRUSIAL
TINGGI_KARUNG_CM = 30.0          # Tebal rata-rata satu karung (CM)

```

> **Catatan Penting:** `STAPEL_TWO_LEVELS_THRESH` harus diisi dengan jarak ukur aktual dari lensa kamera ke lantai agar perhitungan jumlah tingkat akurat.

**Pada `tracking_new.py`:**

```python
CONF_THRES = 0.5    # Tingkatkan jika banyak deteksi palsu
DEVICE = "cuda"     # Ganti ke "cpu" jika tidak memiliki NVIDIA GPU

```

---

## ❓ Troubleshooting

| Masalah | Penyebab | Solusi |
| --- | --- | --- |
| **RuntimeError: No device connected** | Kamera tidak terdeteksi | Cek kabel USB, pastikan tertancap di port USB 3.0 (biasanya warna biru). |
| **FPS Rendah / Lag** | Komputasi berat di CPU | Gunakan GPU (`device='cuda'`) atau kurangi resolusi stream di script. |
| **Total Stapel Salah** | Kalibrasi tinggi salah | Ukur ulang jarak kamera ke lantai dan update `STAPEL_TWO_LEVELS_THRESH`. |
| **Objek Tidak Terdeteksi** | Model tidak mengenali objek | Pastikan pencahayaan cukup atau latih ulang model YOLO dengan dataset baru. |

---

## 📄 Lisensi

Project ini dibuat untuk kebutuhan operasional internal. Silakan sesuaikan lisensi (MIT, Apache, dll) jika ingin dipublikasikan secara terbuka.

---


*Dibuat dengan ❤️ menggunakan Python dan Intel RealSense Technology.*


