"""
BEV + Steering Kalibrasyon Araci
================================
Trackbar'larla tum kritik parametreleri canli ayarla.

Pencereler:
  1) "Orijinal"  : Kamera + BEV trapezoid overlay
  2) "BEV"       : Kusbakisi donusum sonucu
  3) "Maske"     : Binary beyaz serit maskesi
  4) "Histogram" : Alt yaridaki beyaz piksel dagalimi

Tuslar:
  'e' = Mevcut degerleri konsola EXPORT et (kopyala-yapistir formatinda)
  'r' = Varsayilan degerlere RESET
  'q' = Cik

Kullanim:
  python bev_calibrator.py
"""

import sys, os, time
import numpy as np
import cv2

os.environ["GPIOZERO_PIN_FACTORY"] = "lgpio"

try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False

# ============================================================
# VARSAYILAN DEGERLER (lane_opencv.py'den)
# ============================================================
DEFAULTS = {
    # Kamera
    'CAM_W': 1280, 'CAM_H': 720,
    'PROCESS_W': 640, 'CROP_DIV': 5,

    # BEV_SRC noktalar (lane_opencv.py ile senkron)
    'src_tl_x': 90,     'src_tl_y': 0,
    'src_tr_x': 510,    'src_tr_y': 12,
    'src_br_x': 700,    'src_br_y': 240,
    'src_bl_x': -150,   'src_bl_y': 240,

    # BEV boyut
    'BEV_W': 600, 'BEV_H': 400,

    # Parlaklik esigi
    'BRIGHT_THR': 190,

    # Blur
    'BLUR_KERNEL': 9,

    # Steering parametreleri
    'Kp_x100': 240,        # Kp * 100
    'Ki_x100': 5,          # Ki * 100
    'Kd_x100': 20,         # Kd * 100
    'DEADZONE_x1000': 20,  # DEADZONE * 1000  (0.02)
    'ALPHA_x100': 60,      # ALPHA * 100

    # Viraj parametreleri
    'curve_cap_x100': 40,      # curve_boost cap * 100
    'curve_mult': 65,           # curve_boost multiplier
    'far_w_base_x100': 30,     # far_weight base * 100
    'far_w_range_x100': 40,    # far_weight range * 100
    'y_far_pct': 30,            # y_far = h * y_far_pct / 100
    'turn_alpha_x100': 45,     # TURN_CURVATURE_ALPHA * 100
}

# ============================================================
# GLOBAL STATE
# ============================================================
params = dict(DEFAULTS)
cam = None
cap = None
use_usb = True


def nothing(x):
    pass


def create_trackbars():
    """Tum trackbar'lari olustur."""
    cv2.namedWindow("BEV Ayar", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("BEV Ayar", 600, 800)

    # -- BEV_SRC Noktalari --
    # Sol-Ust
    cv2.createTrackbar("SRC TL X",   "BEV Ayar", params['src_tl_x'] + 400, 800, nothing)
    cv2.createTrackbar("SRC TL Y",   "BEV Ayar", params['src_tl_y'], 400, nothing)
    # Sag-Ust
    cv2.createTrackbar("SRC TR X",   "BEV Ayar", params['src_tr_x'], 1000, nothing)
    cv2.createTrackbar("SRC TR Y",   "BEV Ayar", params['src_tr_y'], 400, nothing)
    # Sag-Alt (ekran disi olabilir)
    cv2.createTrackbar("SRC BR X",   "BEV Ayar", params['src_br_x'] + 400, 1600, nothing)
    cv2.createTrackbar("SRC BR Y",   "BEV Ayar", params['src_br_y'], 500, nothing)
    # Sol-Alt (negatif olabilir)
    cv2.createTrackbar("SRC BL X",   "BEV Ayar", params['src_bl_x'] + 600, 1200, nothing)
    cv2.createTrackbar("SRC BL Y",   "BEV Ayar", params['src_bl_y'], 500, nothing)

    # -- BEV Boyut --
    cv2.createTrackbar("BEV W", "BEV Ayar", params['BEV_W'], 1000, nothing)
    cv2.createTrackbar("BEV H", "BEV Ayar", params['BEV_H'], 800, nothing)

    # -- Goruntu Isleme --
    cv2.createTrackbar("BRIGHT THR",  "BEV Ayar", params['BRIGHT_THR'], 254, nothing)
    cv2.createTrackbar("BLUR",        "BEV Ayar", params['BLUR_KERNEL'], 31, nothing)

    # -- Steering Penceresi --
    cv2.namedWindow("Steering Ayar", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Steering Ayar", 600, 600)

    cv2.createTrackbar("Kp x100",           "Steering Ayar", params['Kp_x100'], 500, nothing)
    cv2.createTrackbar("Ki x100",           "Steering Ayar", params['Ki_x100'], 50, nothing)
    cv2.createTrackbar("Kd x100",           "Steering Ayar", params['Kd_x100'], 100, nothing)
    cv2.createTrackbar("DEADZONE x1000",    "Steering Ayar", params['DEADZONE_x1000'], 100, nothing)
    cv2.createTrackbar("ALPHA x100",        "Steering Ayar", params['ALPHA_x100'], 100, nothing)

    # -- Viraj Parametreleri --
    cv2.createTrackbar("CurveCap x100",     "Steering Ayar", params['curve_cap_x100'], 100, nothing)
    cv2.createTrackbar("CurveMult",         "Steering Ayar", params['curve_mult'], 200, nothing)
    cv2.createTrackbar("FarW Base x100",    "Steering Ayar", params['far_w_base_x100'], 80, nothing)
    cv2.createTrackbar("FarW Range x100",   "Steering Ayar", params['far_w_range_x100'], 80, nothing)
    cv2.createTrackbar("Y Far %",           "Steering Ayar", params['y_far_pct'], 60, nothing)
    cv2.createTrackbar("TurnAlpha x100",    "Steering Ayar", params['turn_alpha_x100'], 100, nothing)


def read_trackbars():
    """Trackbar degerlerini oku."""
    p = {}
    # BEV SRC (offset'li: negatif degerler icin)
    p['src_tl_x'] = cv2.getTrackbarPos("SRC TL X", "BEV Ayar") - 400
    p['src_tl_y'] = cv2.getTrackbarPos("SRC TL Y", "BEV Ayar")
    p['src_tr_x'] = cv2.getTrackbarPos("SRC TR X", "BEV Ayar")
    p['src_tr_y'] = cv2.getTrackbarPos("SRC TR Y", "BEV Ayar")
    p['src_br_x'] = cv2.getTrackbarPos("SRC BR X", "BEV Ayar") - 400
    p['src_br_y'] = cv2.getTrackbarPos("SRC BR Y", "BEV Ayar")
    p['src_bl_x'] = cv2.getTrackbarPos("SRC BL X", "BEV Ayar") - 600
    p['src_bl_y'] = cv2.getTrackbarPos("SRC BL Y", "BEV Ayar")

    p['BEV_W'] = max(100, cv2.getTrackbarPos("BEV W", "BEV Ayar"))
    p['BEV_H'] = max(100, cv2.getTrackbarPos("BEV H", "BEV Ayar"))

    p['BRIGHT_THR'] = max(1, cv2.getTrackbarPos("BRIGHT THR", "BEV Ayar"))
    blur = cv2.getTrackbarPos("BLUR", "BEV Ayar")
    p['BLUR_KERNEL'] = blur if blur % 2 == 1 else blur + 1
    p['BLUR_KERNEL'] = max(1, p['BLUR_KERNEL'])

    # Steering
    p['Kp_x100'] = cv2.getTrackbarPos("Kp x100", "Steering Ayar")
    p['Ki_x100'] = cv2.getTrackbarPos("Ki x100", "Steering Ayar")
    p['Kd_x100'] = cv2.getTrackbarPos("Kd x100", "Steering Ayar")
    p['DEADZONE_x1000'] = cv2.getTrackbarPos("DEADZONE x1000", "Steering Ayar")
    p['ALPHA_x100'] = cv2.getTrackbarPos("ALPHA x100", "Steering Ayar")

    # Viraj
    p['curve_cap_x100'] = cv2.getTrackbarPos("CurveCap x100", "Steering Ayar")
    p['curve_mult'] = cv2.getTrackbarPos("CurveMult", "Steering Ayar")
    p['far_w_base_x100'] = cv2.getTrackbarPos("FarW Base x100", "Steering Ayar")
    p['far_w_range_x100'] = cv2.getTrackbarPos("FarW Range x100", "Steering Ayar")
    p['y_far_pct'] = cv2.getTrackbarPos("Y Far %", "Steering Ayar")
    p['turn_alpha_x100'] = cv2.getTrackbarPos("TurnAlpha x100", "Steering Ayar")

    return p


def build_bev_transform(p):
    """Mevcut parametrelerden BEV matrislerini hesapla."""
    bev_w = p['BEV_W']
    bev_h = p['BEV_H']

    src = np.float32([
        [p['src_tl_x'], p['src_tl_y']],
        [p['src_tr_x'], p['src_tr_y']],
        [p['src_br_x'], p['src_br_y']],
        [p['src_bl_x'], p['src_bl_y']],
    ])
    dst = np.float32([
        [0,     0],
        [bev_w, 0],
        [bev_w, bev_h],
        [0,     bev_h],
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    # Aracin BEV'deki konumu
    process_w = p.get('PROCESS_W', 640)
    cam_src = np.float32([[[process_w / 2.0, float(src[2][1])]]])
    cam_dst = cv2.perspectiveTransform(cam_src, M)
    car_x = float(cam_dst[0][0][0])

    return M, M_inv, src, bev_w, bev_h, car_x


def create_lane_mask(warped, bright_thr, blur_k):
    """BEV goruntusunden beyaz serit maskesi."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    _, binary = cv2.threshold(blur, bright_thr, 255, cv2.THRESH_BINARY)

    k_close = np.ones((7, 7), np.uint8)
    k_dilate = np.ones((5, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=2)
    binary = cv2.dilate(binary, k_dilate, iterations=1)
    k_open = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            cv2.drawContours(binary, [cnt], 0, 0, -1)

    return binary


def draw_histogram(binary, bev_w, bev_h, car_x):
    """Histogram gorsellemesi.
    Dondurur: (hist_img, lane_center_x veya None)
    """
    hist_region = binary[bev_h // 2:, :]
    histogram = np.sum(hist_region, axis=0).astype(np.float64)

    # Yumusatma
    kernel = np.ones(15) / 15
    histogram = np.convolve(histogram, kernel, mode='same')

    # Normalize
    h_max = max(histogram.max(), 1)
    hist_h = 300  # Daha yuksek — aciklama metni icin yer
    hist_img = np.zeros((hist_h, max(bev_w, 500), 3), dtype=np.uint8)

    # Histogram barlari
    bar_area_h = 180  # Barlarin kullanabilecegi dikey alan
    for x in range(min(bev_w, len(histogram))):
        bar_h = int(histogram[x] / h_max * (bar_area_h - 20))
        if bar_h > 0:
            cv2.line(hist_img, (x, bar_area_h), (x, bar_area_h - bar_h), (200, 200, 200), 1)

    # Peak'leri bul
    peaks = []
    for i in range(1, len(histogram) - 1):
        if (histogram[i] > histogram[i-1] and
            histogram[i] >= histogram[i+1] and
            histogram[i] > 500):
            peaks.append((i, histogram[i]))

    # Peak'leri isaretle (kirmizi noktalar)
    for px, ph in peaks:
        bar_h = int(ph / h_max * (bar_area_h - 20))
        cv2.circle(hist_img, (px, bar_area_h - bar_h), 6, (0, 0, 255), -1)
        cv2.putText(hist_img, f"x={px}", (px - 15, bar_area_h - bar_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # ============================================================
    # LANE CENTER: iki en guclu peak'in TAM ORTASI
    # ============================================================
    lane_center_x = None
    if len(peaks) >= 2:
        # En guclu 2 peak'i al
        sorted_peaks = sorted(peaks, key=lambda p: p[1], reverse=True)
        top2 = sorted(sorted_peaks[:2], key=lambda p: p[0])  # x sirasina gore
        left_peak_x = top2[0][0]
        right_peak_x = top2[1][0]
        lane_center_x = (left_peak_x + right_peak_x) / 2.0
        lc_xi = int(lane_center_x)

        # KIRMIZI KALIN CIZGI = SERIT MERKEZI
        cv2.line(hist_img, (lc_xi, 0), (lc_xi, bar_area_h), (0, 0, 255), 3)
        cv2.putText(hist_img, f"SERIT ORTA={lc_xi}", (lc_xi + 5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # SARI CIZGI = bev_w/2 (programin kullandigi referans)
    mid = bev_w // 2
    cv2.line(hist_img, (mid, 0), (mid, bar_area_h), (0, 255, 255), 2)
    cv2.putText(hist_img, f"bev_w/2={mid}", (mid + 5, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    # YESIL CIZGI = BEV_CAR_X
    car_xi = int(car_x)
    cv2.line(hist_img, (car_xi, 0), (car_xi, bar_area_h), (0, 255, 0), 2)
    cv2.putText(hist_img, f"CAR_X={car_xi}", (car_xi + 5, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # ============================================================
    # ACIKLAMA METNI (histogram'in altinda)
    # ============================================================
    y_text = bar_area_h + 20
    cv2.putText(hist_img, "KIRMIZI = Serit merkezi (iki peak'in ortasi)",
                (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.putText(hist_img, "SARI = bev_w/2 (programin referansi)",
                (10, y_text + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(hist_img, "YESIL = BEV_CAR_X (kameranin BEV konumu)",
                (10, y_text + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Fark bilgisi
    if lane_center_x is not None:
        diff_sari = lane_center_x - mid
        diff_yesil = lane_center_x - car_x
        status_sari = "TAMAM" if abs(diff_sari) < 15 else f"{diff_sari:+.0f}px KAYIK!"
        status_yesil = "TAMAM" if abs(diff_yesil) < 15 else f"{diff_yesil:+.0f}px KAYIK!"
        cv2.putText(hist_img, f"KIRMIZI-SARI fark: {status_sari}",
                    (10, y_text + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(hist_img, f"KIRMIZI-YESIL fark: {status_yesil}",
                    (10, y_text + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    return hist_img, lane_center_x


def export_values(p, car_x):
    """Degerleri konsola yazdir."""
    bev_w_half = p['BEV_W'] / 2.0
    print("\n" + "=" * 60)
    print("  KALIBRE EDILMIS DEGERLER — lane_opencv.py'ye KOPYALA")
    print("=" * 60)
    print(f"""
# -- BEV Perspektif Donusumu --
BEV_SRC = np.float32([
    [{p['src_tl_x']:>4},  {p['src_tl_y']:>3}],   # Sol - Ust
    [{p['src_tr_x']:>4},  {p['src_tr_y']:>3}],   # Sag - Ust
    [{p['src_br_x']:>4},  {p['src_br_y']:>3}],   # Sag - Alt
    [{p['src_bl_x']:>4},  {p['src_bl_y']:>3}],   # Sol - Alt
])
BEV_W, BEV_H = {p['BEV_W']}, {p['BEV_H']}

# BEV_CAR_X = {car_x:.1f}  (bev_w/2 = {bev_w_half:.1f}, fark = {car_x - bev_w_half:+.1f}px)

# -- Goruntu Isleme --
BRIGHT_THR    = {p['BRIGHT_THR']}
BLUR_KERNEL   = {p['BLUR_KERNEL']}

# -- PID Kazanclari --
Kp = {p['Kp_x100'] / 100:.2f}
Ki = {p['Ki_x100'] / 100:.2f}
Kd = {p['Kd_x100'] / 100:.2f}
DEADZONE = {p['DEADZONE_x1000'] / 1000:.3f}
ALPHA = {p['ALPHA_x100'] / 100:.2f}

# -- Viraj Parametreleri --
TURN_CURVATURE_ALPHA = {p['turn_alpha_x100'] / 100:.2f}

# curve_boost: cap = {p['curve_cap_x100'] / 100:.2f}, mult = {p['curve_mult']}
# Satir 1196-1197 (dual):
#   curve_boost = np.sign(...) * min({p['curve_cap_x100'] / 100:.2f}, abs(...) * {p['curve_mult']})
# Satir 1288-1289 (single):
#   curve_boost = np.sign(...) * min({p['curve_cap_x100'] / 100:.2f}, abs(...) * SINGLE_LANE_CURVE_GAIN * {int(p['curve_mult'] / 1.5)})

# far_weight = {p['far_w_base_x100'] / 100:.2f} + {p['far_w_range_x100'] / 100:.2f} * curve_strength
#   duz: {p['far_w_base_x100'] / 100:.2f}  viraj: {(p['far_w_base_x100'] + p['far_w_range_x100']) / 100:.2f}

# y_far = int(h * {p['y_far_pct'] / 100:.2f})
""")
    print("=" * 60)


def reset_trackbars():
    """Trackbar'lari varsayilan degerlere sifirla."""
    d = DEFAULTS
    cv2.setTrackbarPos("SRC TL X", "BEV Ayar", d['src_tl_x'] + 400)
    cv2.setTrackbarPos("SRC TL Y", "BEV Ayar", d['src_tl_y'])
    cv2.setTrackbarPos("SRC TR X", "BEV Ayar", d['src_tr_x'])
    cv2.setTrackbarPos("SRC TR Y", "BEV Ayar", d['src_tr_y'])
    cv2.setTrackbarPos("SRC BR X", "BEV Ayar", d['src_br_x'] + 400)
    cv2.setTrackbarPos("SRC BR Y", "BEV Ayar", d['src_br_y'])
    cv2.setTrackbarPos("SRC BL X", "BEV Ayar", d['src_bl_x'] + 600)
    cv2.setTrackbarPos("SRC BL Y", "BEV Ayar", d['src_bl_y'])
    cv2.setTrackbarPos("BEV W", "BEV Ayar", d['BEV_W'])
    cv2.setTrackbarPos("BEV H", "BEV Ayar", d['BEV_H'])
    cv2.setTrackbarPos("BRIGHT THR", "BEV Ayar", d['BRIGHT_THR'])
    cv2.setTrackbarPos("BLUR", "BEV Ayar", d['BLUR_KERNEL'])
    cv2.setTrackbarPos("Kp x100", "Steering Ayar", d['Kp_x100'])
    cv2.setTrackbarPos("Ki x100", "Steering Ayar", d['Ki_x100'])
    cv2.setTrackbarPos("Kd x100", "Steering Ayar", d['Kd_x100'])
    cv2.setTrackbarPos("DEADZONE x1000", "Steering Ayar", d['DEADZONE_x1000'])
    cv2.setTrackbarPos("ALPHA x100", "Steering Ayar", d['ALPHA_x100'])
    cv2.setTrackbarPos("CurveCap x100", "Steering Ayar", d['curve_cap_x100'])
    cv2.setTrackbarPos("CurveMult", "Steering Ayar", d['curve_mult'])
    cv2.setTrackbarPos("FarW Base x100", "Steering Ayar", d['far_w_base_x100'])
    cv2.setTrackbarPos("FarW Range x100", "Steering Ayar", d['far_w_range_x100'])
    cv2.setTrackbarPos("Y Far %", "Steering Ayar", d['y_far_pct'])
    cv2.setTrackbarPos("TurnAlpha x100", "Steering Ayar", d['turn_alpha_x100'])
    print("[INFO] Tum degerler varsayilana sifirlandi.")


def main():
    global cam, cap, use_usb

    # -- Kamera Secimi --
    print("\n" + "=" * 50)
    print("  BEV KALIBRASYON ARACI")
    print("=" * 50)
    print("  1) usbcam  -- USB Kamera")
    print("  2) rpi     -- Raspberry Pi Kamera")
    print("=" * 50)
    choice = input("Secim [usbcam/rpi]: ").strip().lower()
    use_usb = choice in ("usbcam", "usb", "1", "")

    if use_usb:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULTS['CAM_W'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULTS['CAM_H'])
        cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():
            print("[HATA] USB kamera acilamadi!")
            sys.exit(1)
        print("[INFO] USB Kamera hazir.")
    else:
        if not HAS_PICAMERA2:
            print("[HATA] Picamera2 bulunamadi!")
            sys.exit(1)
        cam = Picamera2()
        cam.configure(cam.create_video_configuration(
            main={"size": (DEFAULTS['CAM_W'], DEFAULTS['CAM_H']), "format": "RGB888"},
            buffer_count=2
        ))
        cam.start()
        time.sleep(0.5)
        print("[INFO] RPi Kamera hazir.")

    create_trackbars()

    # Gorunum pencereleri
    cv2.namedWindow("Orijinal", cv2.WINDOW_NORMAL)
    cv2.namedWindow("BEV", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Maske", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Histogram", cv2.WINDOW_NORMAL)

    cv2.resizeWindow("Orijinal", 640, 400)
    cv2.resizeWindow("BEV", 600, 400)
    cv2.resizeWindow("Maske", 600, 400)
    cv2.resizeWindow("Histogram", 600, 320)

    print("\n[INFO] Tuslar: 'e'=Export  'r'=Reset  'q'=Cik")
    print("[INFO] Trackbar'larla degerleri ayarlayin.\n")

    while True:
        # Frame oku
        if use_usb:
            ret, frame = cap.read()
            if not ret:
                print("[HATA] Frame alinamadi!")
                break
        else:
            img = cam.capture_array("main")
            frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Trackbar'lardan parametreleri oku
        p = read_trackbars()

        # Crop + resize
        h_img = frame.shape[0]
        y_off = h_img // DEFAULTS['CROP_DIV']
        bottom = frame[y_off:, :]
        bh, bw = bottom.shape[:2]
        process_w = DEFAULTS['PROCESS_W']
        scale = process_w / bw
        proc_h = int(bh * scale)
        small = cv2.resize(bottom, (process_w, proc_h))

        # BEV transform
        try:
            M, M_inv, src_pts, bev_w, bev_h, car_x = build_bev_transform(p)
        except cv2.error:
            continue

        warped = cv2.warpPerspective(small, M, (bev_w, bev_h))
        binary = create_lane_mask(warped, p['BRIGHT_THR'], p['BLUR_KERNEL'])

        # ============================================================
        # 1) ORIJINAL + TRAPEZOID OVERLAY
        # ============================================================
        orig_viz = small.copy()
        pts = src_pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(orig_viz, [pts], True, (0, 255, 0), 2)
        for i, (label, color) in enumerate(zip(
                ['TL', 'TR', 'BR', 'BL'],
                [(255, 0, 0), (0, 0, 255), (0, 165, 255), (255, 255, 0)])):
            x, y = int(src_pts[i][0]), int(src_pts[i][1])
            if 0 <= x < process_w and 0 <= y < proc_h:
                cv2.circle(orig_viz, (x, y), 8, color, -1)
                cv2.putText(orig_viz, label, (x + 10, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Merkez cizgisi
        cx = process_w // 2
        cv2.line(orig_viz, (cx, 0), (cx, proc_h), (0, 255, 255), 1)
        cv2.putText(orig_viz, f"Cam Center={cx}", (cx + 5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # BEV_CAR_X bilgisi
        cv2.putText(orig_viz, f"BEV_CAR_X={car_x:.1f}  bev_w/2={bev_w/2:.1f}  fark={car_x - bev_w/2:+.1f}",
                    (10, proc_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Orijinal", orig_viz)

        # ============================================================
        # 2) BEV GORUNUMU
        # ============================================================
        bev_viz = warped.copy()
        mid_x = bev_w // 2
        cv2.line(bev_viz, (mid_x, 0), (mid_x, bev_h), (0, 255, 255), 1)
        car_xi = int(car_x)
        cv2.line(bev_viz, (car_xi, 0), (car_xi, bev_h), (0, 255, 0), 2)

        # y_far ve y_near cizgileri
        y_far_pct = p['y_far_pct']
        y_near_line = int(bev_h * 0.75)
        y_far_line = int(bev_h * y_far_pct / 100.0)
        cv2.line(bev_viz, (0, y_near_line), (bev_w, y_near_line), (255, 100, 0), 1)
        cv2.line(bev_viz, (0, y_far_line), (bev_w, y_far_line), (0, 100, 255), 1)
        cv2.putText(bev_viz, f"y_near={y_near_line}", (5, y_near_line - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 0), 1)
        cv2.putText(bev_viz, f"y_far={y_far_line}", (5, y_far_line - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 1)

        # Parametre degerleri overlay
        fw_max = (p['far_w_base_x100'] + p['far_w_range_x100']) / 100.0
        info_lines = [
            f"Kp={p['Kp_x100']/100:.2f} Ki={p['Ki_x100']/100:.2f} Kd={p['Kd_x100']/100:.2f}",
            f"DZ={p['DEADZONE_x1000']/1000:.3f} A={p['ALPHA_x100']/100:.2f}",
            f"CB cap={p['curve_cap_x100']/100:.2f} m={p['curve_mult']}",
            f"FW={p['far_w_base_x100']/100:.2f}..{fw_max:.2f}",
            f"TA={p['turn_alpha_x100']/100:.2f}",
        ]
        for i, txt in enumerate(info_lines):
            cv2.putText(bev_viz, txt, (5, 20 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

        cv2.imshow("BEV", bev_viz)

        # ============================================================
        # 3) BINARY MASKE
        # ============================================================
        mask_viz = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        cv2.line(mask_viz, (mid_x, 0), (mid_x, bev_h), (0, 255, 255), 1)
        cv2.line(mask_viz, (car_xi, 0), (car_xi, bev_h), (0, 255, 0), 2)
        cv2.imshow("Maske", mask_viz)

        # ============================================================
        # 4) HISTOGRAM
        # ============================================================
        hist_img, lane_center_x = draw_histogram(binary, bev_w, bev_h, car_x)
        cv2.imshow("Histogram", hist_img)

        # ============================================================
        # TUS ISLEMLERI
        # ============================================================
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            export_values(p, car_x)
        elif key == ord('r'):
            reset_trackbars()

    # Temizlik
    if cap:
        cap.release()
    if cam:
        cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
