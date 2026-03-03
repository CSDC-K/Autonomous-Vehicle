# =============================================================================
# HAILO8L OTONOM — YAPILANDIRMA  v3
# =============================================================================

# --- Model ---
HEF_PATH = "/home/kurttepe-2/Desktop/otonom_model.hef"
MODEL_W  = 416
MODEL_H  = 416

# --- Kamera ---
CAMERA_W   = 640
CAMERA_H   = 480
CAMERA_FPS = 30

# --- Sınıf İsimleri (dataset.yaml sırasıyla) ---
CLASS_NAMES  = ["line_dashed", "line_solid"]
CLASS_COLORS = [(0, 255, 0), (0, 0, 255)]   # BGR: yeşil, kırmızı

# --- Quantization (DFC sdk_quantized, 13 görüntüden) ---
# float = (uint8 - QP_ZP) * QP_SCALE
QP_SCALE = 0.75708008
QP_ZP    = 124   # uint8=124 → float=0 → sigmoid=0.5

# --- Tespit eşikleri ---
# Raw uint8 class score eşiği (başlangıç değeri)
# +/- tuşlarıyla canlı ayarlanabilir
# ZP=124 olduğundan: uint8=124→sigmoid=0.50, 127→0.90, 130→0.99
# 130 iyi başlangıç, artır/azalt ile ayarla
SCORE_UINT8_THRESHOLD = 200
IOU_THRESHOLD         = 0.45
TOP_K                 = 30     # NMS öncesi max aday sayısı

# --- Kutu filtreleri (şerit çizgileri için) ---
MIN_BOX_AREA    = 0.002   # Min kutu alanı (normalize: 0..1 aralığında)
MAX_BOX_AREA    = 5.0    # Max kutu alanı
MIN_ASPECT      = 0.3     # Min en/boy oranı (0=çok ince dikey)
MAX_ASPECT      = 8.0     # Max en/boy oranı (çok yatay olmasın)

# --- Karanlık çerçeve kontrolü ---
DARK_FRAME_THRESHOLD = 25    # Ortalama piksel < bu → tespit yapma

# --- Görüntüleme ---
SHOW_FPS        = True
SHOW_LABELS     = True
SHOW_CONFIDENCE = True
SHOW_BBOX       = True
WINDOW_TITLE    = "Hailo8L - Serit Takip"
DISPLAY_W       = 640
DISPLAY_H       = 480

# --- Post-process ---
DEBUG_SO   = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libdebug.so"
POST_FUNC  = "dump_tensors_to_npy"

# --- Tensor dump ---
# libdebug.so dosyaları CWD'ye yazar
NPY_DIR         = "."
TENSOR_DET_FILE = "model_concat18.npy"
