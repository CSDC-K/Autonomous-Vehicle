"""
OpenCV Serit Takip — 18. MEB Yarismasi
BEV + Sliding Window Pipeline
---------------------------------------
Iki pencere:
  - "Kamera"   : gercek goruntu + serit overlay + PID HUD
  - "Algilama" : BEV mask + sliding window kutulari

Tuslar: 'a'=otonom  SPACE=acil dur  'q'=cik
        '+'/'-' = BRIGHT_THR canli ayar
        'z'/'x' = SERVO_OFFSET canli ayar (sola/saga trim)
"""

import sys, os, time, signal
import numpy as np
import cv2

os.environ["GPIOZERO_PIN_FACTORY"] = "lgpio"

try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False

from motor_control import MotorController


# +============================================================+
# |                       KALIBRASYON                           |
# +============================================================+

# -- Kamera -------------------------------------------------------
CAM_W, CAM_H  = 1280, 720
PROCESS_W     = 640
VIZ_SKIP      = 2

# -- Crop ---------------------------------------------------------
CROP_DIV      = 5

# -- Parlaklik Esigi ----------------------------------------------
BRIGHT_THR    = 190

# -- Zemin Turu ---------------------------------------------------
LIGHT_LINE_ON_DARK = True

# -- Gaussian Blur ------------------------------------------------
BLUR_KERNEL = 9

# -- BEV (Bird's Eye View) Perspektif Donusumu --------------------
# ISLEYIS MANTIGI:
# Kameranin gordugu gercek yol, ufuk noktasinda birlesen bir "yamuktur" (trapezoid).
# BEV algoritmasi bu yamugu (BEV_SRC) alip dumduz bir "dikdortgene" (BEV_DST) cevirir.
#
# SORUN: Neden kenarlarda siyah bosluklar olusuyor?
# Eger BEV_DST (Hedef) dikdortgeninin koselerini X=60 gibi iceriden baslatirsak,
# sistem X=0 ile X=60 arasina bilerek SİYAH BOSLUK koyar. Ayni sekilde yamugun
# disinda kalan pikseller de dahil edilmez.
#
# COZUM: 
# 1. BEV_SRC'yi kameranin tm sag/sol sinirlarini (x=0, x=640) kapsayacak sekilde ac.
# 2. BEV_DST'yi ekranin tam genisligine (x=0'dan BEV_W'ye) daya (siyah bosluklari iptal et).

# GERCEK ISLEME BOYUTU: 640 x 288 (1280x720 kameranin crop edilmis hali)
# Sira: sol-ust, sag-ust, sag-alt, sol-alt
# -- BEV Perspektif Donusumu --
BEV_SRC = np.float32([
    [ 118,   13],   # Sol - Ust
    [ 601,    9],   # Sag - Ust
    [ 957,  275],   # Sag - Alt
    [-184,  275],   # Sol - Alt
])


# Hedef dikdortgen (BEV ciktisi)
# BEV_W'yu 400'den 600'e cikardik ki Serit pikselleri ezilmesin, yanyana ferahca islensin.
BEV_W, BEV_H = 600, 400
BEV_DST = np.float32([
    [    0,       0],  # Eskiden X=60'ti (Siyah bosluk yaratiyordu), simdi SIFIR!
    [BEV_W,       0],  # Eskiden 340'ti, simdi tam ekran (600)
    [BEV_W,   BEV_H],  # Sag alt kumpas noktasi
    [    0,   BEV_H],  # Sol alt kumpas noktasi
])


# -- Sliding Window -----------------------------------------------
N_WINDOWS      = 12     # dikey dilim sayisi
WINDOW_MARGIN  = 70     # pencere yari genisligi (px) - yeni BEV icin olceklendi (80*0.818)
MIN_PIX        = 30     # pencereyi kaydirmak icin min piksel
HIST_SMOOTH    = 15     # histogram yumusatma kernel boyutu

# -- Histogram Peak Clustering (| || | yol yapisi) ----------------
# Birbirine yakin peak'leri (orta cift cizgi ||) birlestir
PEAK_CLUSTER_GAP   = 60    # px - || cift cizgiyi her zaman tek cluster olarak birlestir
PEAK_MIN_HEIGHT    = 500   # histogram peak minimum yuksekligi

# -- Serit Hafizasi -----------------------------------------------
DEFAULT_LANE_WIDTH_PX = 160   # BEV'de beklenen serit genisligi (px) — yeni BEV: 200*0.818=163→160
LANE_WIDTH_ALPHA      = 0.10  # serit genisligi EMA katsayisi
MAX_LANE_WIDTH_CHANGE = 80    # frame'ler arasi max genislik degisimi (px)
MIN_LANE_WIDTH_PX     = 100   # Iki serit arasi minimum mesafe — yeni BEV: 130*0.818=106→100
MAX_LANE_WIDTH_PX     = 380   # Iki serit arasi maximum mesafe — kamera kenar→kenar yeni BEV ≈366px
STRICT_PAIR_MIN_GAP_PX = 120  # || ciftinin iki cizgisini sahte dual lane saymamak icin katı alt sinir

# -- Polinom Stabilizasyonu ---------------------------------------
POLY_ALPHA     = 0.85   # polinom EMA — 0.90 -> 0.85: curvature gercek zamanli takip etsin
MAX_POLY_JUMP  = 200    # px - polinom base x fark esigi (virajda daha toleransli)

# -- Serit Yeniden Bulma ------------------------------------------
REACQ_MIN_SEPARATION = 80   # px - yeniden bulma minimum ayirimi — yeni BEV: 100*0.818=82→80
REACQ_EXPECTED_MARGIN = 90   # px - kayip tarafta beklenen x etrafinda tek-peak reacquire toleransi

# -- PID Kazanclari -----------------------------------------------
Kp = 2.80
Ki = 0.03
Kd = 0.30
INTEGRAL_LIMIT  = 0.20
D_FILTER_ALPHA  = 0.65   # 0.50 -> 0.65: D-term viraj girisini hissetsin

# -- Deadzone -----------------------------------------------------
DEADZONE = 0.05   # bev_calibrator.py ile ayarla

# -- EMA Yumusatma ------------------------------------------------
ALPHA = 0.60   # 0.70 -> 0.60: error EMA daha hizli, viraj tepkisi gec kalmasin

# -- Adaptif Hiz --------------------------------------------------
BASE_SPEED         = 1.0
MIN_SPEED          = 0.65  # 1.0->0.65: virajda hiz dusmesine izin ver
SPEED_CURVE_FACTOR = 0.20  # 0.0->0.30: max direksiyonda %30 yavaslama

# -- Guvenlik -----------------------------------------------------
NO_LINE_TIMEOUT     = 15
NO_LINE_SPEED_MULT  = 0.50
NO_LINE_STEER_DECAY = 0.85

# -- Viraj Hafizasi ------------------------------------------------
TURN_CURVATURE_ALPHA  = 0.40   # 0.55 -> 0.40: erken donus azaltma
TURN_CURVATURE_THR    = 0.0003 # Bu deger uzerinde "virajda" sayilir
TURN_MEMORY_FRAMES    = 90     # Viraj yonunu bu kadar frame hatirla (~3s) — uzun virajlar icin arttirildi
SINGLE_LANE_BASE_STEER = 0.35  # Viraj cikisi kurtarma icin (sadece LOST modda kullanilir)
SINGLE_LANE_CURVE_GAIN = 1.5   # Egrilik -> direksiyon carpani
CLASSIFY_HYSTERESIS    = 30    # px — flip-flop onleme bandi
TURN_CURVATURE_ALPHA_SINGLE = 0.60  # Single mode curvature hizli guncelleme (0.40->0.60)

# -- Polinom Kalite Kontrolu (Yatay Serit Reddi) -------------------
FIT_MIN_VERTICAL_SPAN  = 0.30  # Fit en az BEV yuksekliginin %30'unu kapsamali
FIT_MAX_HORIZONTAL_RATIO = 3.0 # x_range / y_range > bu deger ise "yatay" — reddet
FIT_MIN_SLOPE          = 0.3   # Polinom alt noktadaki min |dy/dx| (dikey olmali)

# -- Yaya Gecidi Filtresi -------------------------------------------
CROSSING_ROW_DENSITY_THR = 0.45  # Satirin %45'i beyazsa → gecit cizgisi (viraj FP onlemi icin yuksek)
CROSSING_MIN_ROWS        = 30    # Ardisik blok bu kadar satir olursa gecit onaylanir (viraj FP onlemi)
CROSSING_COOLDOWN        = 30    # Gecit algilandiktan sonra N frame aktif tut
CROSSING_SPEED           = 0.50  # Gecit algilandiginda hiz

# -- Servo Limitleri ------------------------------------------------
SERVO_MAX_ANGLE        = 75    # Derece — motor_control.py ile uyumlu (artirildi: 60 → 75)


# +============================================================+
# |                     PID CONTROLLER                          |
# +============================================================+

class PIDController:
    def __init__(self, kp, ki, kd,
                 integral_limit=0.3, d_filter_alpha=0.7):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._integral_limit = integral_limit
        self._d_alpha = d_filter_alpha
        self._prev_error = 0.0
        self._integral = 0.0
        self._d_filtered = 0.0
        self._prev_time = None

    def reset(self):
        self._prev_error = 0.0
        self._integral = 0.0
        self._d_filtered = 0.0
        self._prev_time = None

    def update(self, error):
        now = time.monotonic()
        dt = 0.033 if self._prev_time is None else max(now - self._prev_time, 0.001)
        self._prev_time = now

        if abs(error) < DEADZONE:
            error = 0.0

        p = self.kp * error

        self._integral += error * dt
        self._integral = max(-self._integral_limit,
                             min(self._integral_limit, self._integral))
        i = self.ki * self._integral

        raw_d = (error - self._prev_error) / dt
        self._d_filtered = (self._d_alpha * raw_d +
                            (1.0 - self._d_alpha) * self._d_filtered)
        d = self.kd * self._d_filtered
        self._prev_error = error

        output = p + i + d

        if abs(output) >= 1.0:
            self._integral -= error * dt

        return max(-1.0, min(1.0, output))


# +============================================================+
# |                  TURN TRACKER                              |
# +============================================================+

class TurnTracker:
    """Viraj yonu ve egrilik hafizasi.

    Iki serit gorunurken polinomlardan egriligi hesaplar.
    Tek serite gecildiginde son bilinen viraj yonunu ve
    hangi seridin kaybolduğunu hatirlar.

    turn_direction: -1=sol viraj, 0=duz, +1=sag viraj
    """

    def __init__(self):
        self._curvature = 0.0           # EMA filtered curvature
        self._turn_direction = 0        # -1, 0, +1
        self._frames_since_dual = 0     # Son iki-seritli goruntuden bu yana gecen frame
        self._last_left_base_x = None   # Son bilinen sol serit x (BEV altinda)
        self._last_right_base_x = None  # Son bilinen sag serit x (BEV altinda)
        self._last_lane_center_x = None # Son bilinen serit merkezi
        self._which_lost = None         # 'left' veya 'right' — hangi serit kayboldu
        self._locked_side = None        # Tek serit modunda kilitli siniflandirma
        self._lock_confidence = 0       # Kilidin guc seviyesi
        self._switch_votes = 0          # Kilitli taraftan cikis icin ardarda oy sayaci

    def update_dual(self, left_fit, right_fit, bev_h, bev_w):
        """Iki serit de gorunurken cagirilir. Egrilik ve pozisyonlari guncelle."""
        self._frames_since_dual = 0
        # Dual moda donuldu — kilidi serbest birak
        self._locked_side = None
        self._lock_confidence = 0
        self._switch_votes = 0

        # Serit pozisyonlarini kaydet
        y_bottom = bev_h - 1
        lx = left_fit(y_bottom)
        rx = right_fit(y_bottom)
        self._last_left_base_x = lx
        self._last_right_base_x = rx
        self._last_lane_center_x = (lx + rx) / 2.0

        # Egrilik: Her iki polinomun 2. derece katsayisinin ortalamasi
        # Pozitif a -> sola kivisli (BEV'de), negatif a -> saga kivisli
        a_left = left_fit.coefficients[0] if len(left_fit.coefficients) == 3 else 0
        a_right = right_fit.coefficients[0] if len(right_fit.coefficients) == 3 else 0
        curvature = (a_left + a_right) / 2.0

        self._curvature = (TURN_CURVATURE_ALPHA * curvature +
                           (1.0 - TURN_CURVATURE_ALPHA) * self._curvature)

        # Viraj yonu belirle
        if abs(self._curvature) > TURN_CURVATURE_THR:
            # BEV'de pozitif a -> parabolu sola aciyor -> SAG viraj
            # (Cunku y asagi artiyor, pozitif a x'i sola kaydiriyor, yol saga kivriliyor)
            self._turn_direction = 1 if self._curvature > 0 else -1
        else:
            self._turn_direction = 0

    def update_single(self, visible_fit, is_left_visible, bev_h, bev_w):
        """Tek serit gorunurken cagirilir."""
        self._frames_since_dual += 1

        # Hangi seridin kaybolduğunu belirle
        if is_left_visible:
            self._which_lost = 'right'
        else:
            self._which_lost = 'left'

        # Gorunen seridin egriliginden viraj yonu guncelle
        # Single mode'da TEK fit dominant olmamali — ayri (yavas) alpha kullanilir
        # ki dual mode TURN_CURVATURE_ALPHA artisi single tarafi gurultu yapmasin.
        if visible_fit is not None and len(visible_fit.coefficients) == 3:
            a = visible_fit.coefficients[0]
            self._curvature = (TURN_CURVATURE_ALPHA_SINGLE * a +
                               (1.0 - TURN_CURVATURE_ALPHA_SINGLE) * self._curvature)
            if abs(self._curvature) > TURN_CURVATURE_THR:
                self._turn_direction = 1 if self._curvature > 0 else -1

    def update_lost(self):
        """Hic serit gorunmuyorken cagirilir."""
        self._frames_since_dual += 1

    # Kilit suresi — bu kadar frame sonra kilit otomatik serbest birakilir
    # SORUN: 40 frame (~1.3s) keskin virajlar icin cok KISA!
    # Kilit virayin ortasinda acilip || yanlis tarafa ataniyor → arac yanlıs yone gidiyor.
    # COZUM: 150 frame (~5s) — dual moda donulunce zaten hemen sifirlanir (update_dual)
    LOCK_TIMEOUT_FRAMES = 45  # ~1.5s (30fps) — viraj sonu daha hizli yeniden degerlendirsin (phantom kalmasin)
    SINGLE_HARD_LOCK_FRAMES = 90  # Single moda geciste taraf kilidini katı tut
    # Dual mod pozisyon tiebreaker zaten dogru siniflandirma yapiyor

    def classify_single_lane(self, visible_fit, bev_h, bev_w, tracker_hint=None):
        """Gorunen tek seridin aracin SOLunda mi SAGinda mi oldugunu belirle.

        YAKLASIM (v4 — KILIT + TIMEOUT + MERKEZ BAZLI):
        1. Kilit varsa VE suresi dolmadiysa → HEMEN dondur (flip-flop engeli)
        2. Kilit suresi dolduysa → YENİDEN deger (viraj cikisi toparlanmasi)
        3. Referans: Son bilinen SERIT MERKEZI (dual moddan)
        4. Belirsiz bolgede: Son bilinen serit pozisyonlarina yakinlik

        Dondurur: 'left' veya 'right'
        """
        if visible_fit is None:
            return self._locked_side or 'right'

        if tracker_hint not in ('left', 'right'):
            tracker_hint = None

        y_bottom = bev_h - 1
        vx = visible_fit(y_bottom)

        # Dual mod pozisyonlari varsa DAIMA onlari kullan — bev_w/2 kiyasina gitme
        # Arac virajda konum kaydirmis olabilir; bev_w/2 referansi yanlis sonuc uretir
        # Bu ozellikle virajlar arasi geciste ikinci viraj flip sorununu cozuyor
        if self._last_left_base_x is not None and self._last_right_base_x is not None:
            dist_left  = abs(vx - self._last_left_base_x)
            dist_right = abs(vx - self._last_right_base_x)
            if abs(dist_left - dist_right) <= CLASSIFY_HYSTERESIS and self._last_lane_center_x is not None:
                decision = 'right' if vx >= self._last_lane_center_x else 'left'
            else:
                decision = 'left' if dist_left < dist_right else 'right'
        else:
            # Dual mod pozisyonu yok — bev_w/2 baz alinan basit karar
            ref_x = bev_w / 2.0
            if vx > ref_x + CLASSIFY_HYSTERESIS:
                decision = 'right'
            elif vx < ref_x - CLASSIFY_HYSTERESIS:
                decision = 'left'
            else:
                decision = 'right' if vx >= ref_x else 'left'

        # Ilk tek-serit kararinda tracker tarafini tohum olarak kullan.
        if self._locked_side is None:
            if tracker_hint is not None:
                decision = tracker_hint
            self._locked_side = decision
            self._switch_votes = 0
            return decision

        # Single'a gecisin ilk safhasinda tarafi katı tut; || flip'i engeller.
        if self._frames_since_dual < self.SINGLE_HARD_LOCK_FRAMES:
            return self._locked_side

        # Viraj disinda da kisa bir sure kilidi koru.
        if self._frames_since_dual < self.LOCK_TIMEOUT_FRAMES:
            return self._locked_side

        # Kilit suresi dolduktan sonra da ani flip'e izin verme:
        # yeni tarafa gecmek icin ardarda birkac frame ayni karar gerekli.
        if decision == self._locked_side:
            self._switch_votes = 0
            return self._locked_side

        self._switch_votes += 1
        if self._switch_votes < 3:
            return self._locked_side

        self._locked_side = decision
        self._switch_votes = 0
        return self._locked_side

    @property
    def turn_direction(self):
        return self._turn_direction

    @property
    def curvature(self):
        return self._curvature

    @property
    def which_lost(self):
        return self._which_lost

    @property
    def frames_since_dual(self):
        return self._frames_since_dual


# +============================================================+
# |              BIRD'S EYE VIEW TRANSFORM                      |
# +============================================================+

# Transform matrisleri bir kez hesapla (performans)
_M_bev = cv2.getPerspectiveTransform(BEV_SRC, BEV_DST)
_M_inv = cv2.getPerspectiveTransform(BEV_DST, BEV_SRC)

# ARACIN BEV'DEKİ GERCEK KONUMU
# Kamera merkezi (PROCESS_W/2) BEV_SRC alt kenarinda BEV'e donusturulur.
# BEV_SRC alt kenari asimetrik: sol=-28, sag=604.
# Bu yuzden kamera merkezi (x=320) BEV'de x=361.5 DEGIL, ~399.6 civarinda!
# Bu fark duzeltilmezse arac HER ZAMAN bir tarafa yapisir.
_cam_src = np.float32([[[PROCESS_W / 2.0, float(BEV_SRC[2][1])]]])
_cam_dst = cv2.perspectiveTransform(_cam_src, _M_bev)
BEV_CAR_X = float(_cam_dst[0][0][0])
print(f"[INIT] BEV_CAR_X = {BEV_CAR_X:.1f}  (bev_w/2 = {BEV_W/2.0:.1f}, fark = {BEV_CAR_X - BEV_W/2.0:+.1f}px)")


def bev_warp(img):
    """Kamera goruntusu -> kusbakisi (BEV)."""
    return cv2.warpPerspective(img, _M_bev, (BEV_W, BEV_H),
                               flags=cv2.INTER_LINEAR)


def bev_unwarp(img, orig_size):
    """BEV -> orijinal perspektif (viz icin)."""
    return cv2.warpPerspective(img, _M_inv, orig_size,
                               flags=cv2.INTER_LINEAR)


# +============================================================+
# |                    BEYAZ SERIT MASKESI                       |
# +============================================================+

def create_lane_mask(warped, bright_thr):
    """BEV goruntusunden beyaz serit maskesi olustur.
    Sadece threshold + morfoloji. Yaya gecidi filtresi devre disi."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (BLUR_KERNEL, BLUR_KERNEL), 0)

    thr_type = cv2.THRESH_BINARY if LIGHT_LINE_ON_DARK else cv2.THRESH_BINARY_INV
    _, binary = cv2.threshold(blur, bright_thr, 255, thr_type)

    # Morfoloji: kopuk/silik cizgileri birlestir, kavis bosluklarini doldur
    k_close = np.ones((7, 7), np.uint8)
    k_dilate = np.ones((5, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=2)
    binary = cv2.dilate(binary, k_dilate, iterations=1)
    # Ince gurultuyu temizle
    k_open = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)

    # -- Kucuk gurultuleri sil (alan bazli filtre) --
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 100:  # 100 pikselden kucuk alanlari sil
            cv2.drawContours(binary, [cnt], 0, 0, -1)

    return binary


def filter_pedestrian_crossing(mask, turn_active=False):
    """BEV maskesinden yaya gecidi cizgilerini temizle.

    Ardisik blok kontrolu ile viraj false positive onlenir.
    turn_active=True ise filtre tamamen bypass edilir (virajda FP onlemi).

    Dondurur: (temizlenmis_mask, crossing_row_count)
    """
    if turn_active:
        return mask, 0
    h, w = mask.shape[:2]
    threshold = int(w * CROSSING_ROW_DENSITY_THR)

    row_sums = np.sum(mask > 0, axis=1)
    dense_rows = row_sums > threshold

    # Ardisik blok bulma — max 5px bosluklari tolere et
    # Zebra deseni: beyaz-siyah-beyaz, aralarinda kucuk bosluklar olabilir
    best_block_size = 0
    current_start = -1
    gap = 0
    MAX_GAP = 5

    for y in range(h):
        if dense_rows[y]:
            if current_start < 0:
                current_start = y
            gap = 0
        else:
            if current_start >= 0:
                gap += 1
                if gap > MAX_GAP:
                    block_size = y - gap - current_start
                    if block_size > best_block_size:
                        best_block_size = block_size
                    current_start = -1
                    gap = 0

    # Son blok kontrolu
    if current_start >= 0:
        block_size = h - gap - current_start
        if block_size > best_block_size:
            best_block_size = block_size

    # Yeterli ardisik blok varsa → zebra onaylanir, temizle
    crossing_confirmed = best_block_size >= CROSSING_MIN_ROWS
    if crossing_confirmed:
        mask[dense_rows, :] = 0

    return mask, best_block_size if crossing_confirmed else 0


# +============================================================+
# |              SLIDING WINDOW TRACKER                         |
# +============================================================+

class SlidingWindowTracker:
    """Histogram bazli sliding window ile serit takibi.
    BEV binary mask uzerinde alttan uste pencere kaydirarak
    sol ve sag serit piksellerini toplar, polinom fit yapar."""

    def __init__(self):
        self._left_fit = None
        self._right_fit = None
        self._last_left_x = None
        self._last_right_x = None
        self._left_miss_count = 0
        self._right_miss_count = 0

    def _pick_peak_near_expected(self, histogram, expected_x, max_delta):
        """Beklenen x etrafinda en yakin ve yeterince guclu peak'i sec.
        Tek-taraf reacquire icin kullanilir."""
        w = len(histogram)
        ex = int(np.clip(expected_x, 1, w - 2))
        best = None
        best_dist = max_delta + 1
        best_h = -1.0
        for i in range(1, w - 1):
            if (histogram[i] > histogram[i - 1] and
                histogram[i] >= histogram[i + 1] and
                histogram[i] > PEAK_MIN_HEIGHT):
                d = abs(i - ex)
                if d <= max_delta and (d < best_dist or (d == best_dist and histogram[i] > best_h)):
                    best = i
                    best_dist = d
                    best_h = histogram[i]
        return best

    def search(self, binary, build_viz=True):
        """binary: BEV white mask
        Dondurur: (left_fit, right_fit, left_y_range, right_y_range, viz_img | None)
        left_fit/right_fit: np.poly1d veya None
        y_range: (y_min, y_max) tuple
        
        VIRAJ KORUMASI — DEMIR KURAL:
        Kayip serit TAMAMEN DEVRE DISI. Window bile calismaz.
        Yeniden bulma SADECE histogram IKI AYRIK peak gosterirse.
        """
        h, w = binary.shape[:2]
        win_h = h // N_WINDOWS
        midpoint = w // 2

        nonzero = binary.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        use_prev_left = (self._left_fit is not None)
        use_prev_right = (self._right_fit is not None)

        left_valid = False
        right_valid = False
        left_base = midpoint // 2
        right_base = midpoint + midpoint // 2

        # --- MEVCUT FIT VARSA: izle ---
        if use_prev_left:
            left_base = int(np.clip(self._left_fit(h - 1), 0, w - 1))
            left_valid = True
        if use_prev_right:
            right_base = int(np.clip(self._right_fit(h - 1), 0, w - 1))
            right_valid = True

        # --- KAYIP SERIT VARSA: histogram ile YENIDEN BULMA KONTROLU ---
        if not use_prev_left or not use_prev_right:
            hist_region = binary[h // 2:, :]
            histogram = np.sum(hist_region, axis=0).astype(np.float64)
            if HIST_SMOOTH > 1:
                kernel = np.ones(HIST_SMOOTH) / HIST_SMOOTH
                histogram = np.convolve(histogram, kernel, mode='same')
            lb_hist, rb_hist, lv_hist, rv_hist = self._find_lane_peaks(
                histogram, midpoint)

            # YENIDEN BULMA SARTI: IKI peak de gecerli + yeterli mesafe
            both_valid = lv_hist and rv_hist
            separation = rb_hist - lb_hist if both_valid else 0
            safe_to_reacquire = both_valid and separation >= max(REACQ_MIN_SEPARATION, STRICT_PAIR_MIN_GAP_PX)

            # Son bilinen serit genisligi (tek taraf geri kazanimi icin beklenti)
            if self._last_left_x is not None and self._last_right_x is not None:
                expected_width = int(np.clip(abs(self._last_right_x - self._last_left_x),
                                             STRICT_PAIR_MIN_GAP_PX, MAX_LANE_WIDTH_PX))
            else:
                expected_width = int(DEFAULT_LANE_WIDTH_PX)

            if not use_prev_left and not use_prev_right:
                # IKI serit de kayip — histogram ile ilk bulma
                left_base, left_valid = lb_hist, lv_hist
                right_base, right_valid = rb_hist, rv_hist
            elif not use_prev_left and use_prev_right:
                # SOL KAYIP — SADECE guvenli ise yeniden bul
                if safe_to_reacquire:
                    left_base = lb_hist
                    left_valid = True
                else:
                    expected_left = right_base - expected_width
                    cand_left = self._pick_peak_near_expected(histogram, expected_left, REACQ_EXPECTED_MARGIN)
                    if cand_left is not None and cand_left < right_base - STRICT_PAIR_MIN_GAP_PX:
                        left_base = int(cand_left)
                        left_valid = True
                    else:
                        left_valid = False  # DEVRE DISI — tek serit modu
                        left_base = max(0, right_base - expected_width)
            elif use_prev_left and not use_prev_right:
                # SAG KAYIP — SADECE guvenli ise yeniden bul
                if safe_to_reacquire:
                    right_base = rb_hist
                    right_valid = True
                else:
                    expected_right = left_base + expected_width
                    cand_right = self._pick_peak_near_expected(histogram, expected_right, REACQ_EXPECTED_MARGIN)
                    if cand_right is not None and cand_right > left_base + STRICT_PAIR_MIN_GAP_PX:
                        right_base = int(cand_right)
                        right_valid = True
                    else:
                        right_valid = False
                        right_base = min(w - 1, left_base + expected_width)

        # SOL her zaman SOLDA olmali
        if left_valid and right_valid and left_base >= right_base:
            if use_prev_right:
                left_valid = False
            else:
                right_valid = False

        # ============================================================
        # SLIDING WINDOW ARAMA
        # ============================================================
        left_current = left_base
        right_current = right_base
        left_lane_inds = []
        right_lane_inds = []

        viz = None
        if build_viz:
            viz = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        for win_idx in range(N_WINDOWS):
            y_low  = h - (win_idx + 1) * win_h
            y_high = h - win_idx * win_h
            y_mid  = (y_low + y_high) // 2

            # Predicted search: onceki fit varsa pencereyi tahmin edilen x'e tasi
            if use_prev_left and self._left_fit is not None:
                predicted_lx = int(np.clip(self._left_fit(y_mid), 0, w - 1))
                left_current = (left_current + predicted_lx) // 2

            if use_prev_right and self._right_fit is not None:
                predicted_rx = int(np.clip(self._right_fit(y_mid), 0, w - 1))
                right_current = (right_current + predicted_rx) // 2

            # Adaptif margin
            margin = WINDOW_MARGIN
            if win_idx > 0:
                shift = abs(left_current - left_base) + abs(right_current - right_base)
                margin = min(WINDOW_MARGIN * 2, WINDOW_MARGIN + shift // 3)

            # Sol pencere
            xl_lo = max(0, left_current - margin)
            xl_hi = min(w, left_current + margin)
            # Sag pencere
            xr_lo = max(0, right_current - margin)
            xr_hi = min(w, right_current + margin)

            # PENCERELER KESISMEZ
            dynamic_mid = (left_current + right_current) // 2
            if left_current < right_current:
                xl_hi = min(xl_hi, dynamic_mid)
                xr_lo = max(xr_lo, dynamic_mid)

            if build_viz and viz is not None:
                cv2.rectangle(viz, (xl_lo, y_low), (xl_hi, y_high), (255, 100, 0), 2)
                cv2.rectangle(viz, (xr_lo, y_low), (xr_hi, y_high), (0, 100, 255), 2)

            # Pencere icindeki pikseller
            good_left = ((nonzero_y >= y_low) & (nonzero_y < y_high) &
                         (nonzero_x >= xl_lo) & (nonzero_x < xl_hi)).nonzero()[0]
            good_right = ((nonzero_y >= y_low) & (nonzero_y < y_high) &
                          (nonzero_x >= xr_lo) & (nonzero_x < xr_hi)).nonzero()[0]

            left_lane_inds.append(good_left)
            right_lane_inds.append(good_right)

            if len(good_left) > MIN_PIX:
                left_current = int(np.mean(nonzero_x[good_left]))
            if len(good_right) > MIN_PIX:
                right_current = int(np.mean(nonzero_x[good_right]))

        left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([], dtype=int)
        right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([], dtype=int)

        # ============================================================
        # POLINOM FIT
        # ============================================================
        left_fit = None
        right_fit = None
        left_y_range = (0, h)
        right_y_range = (0, h)

        if len(left_lane_inds) > MIN_PIX * 3 and left_valid:
            lx = nonzero_x[left_lane_inds]
            ly = nonzero_y[left_lane_inds]
            left_y_range = (int(ly.min()), int(ly.max()))
            try:
                coeffs = np.polyfit(ly, lx, 2)
                left_fit = np.poly1d(coeffs)
            except (np.RankWarning, np.linalg.LinAlgError):
                pass

        if len(right_lane_inds) > MIN_PIX * 3 and right_valid:
            rx = nonzero_x[right_lane_inds]
            ry = nonzero_y[right_lane_inds]
            right_y_range = (int(ry.min()), int(ry.max()))
            try:
                coeffs = np.polyfit(ry, rx, 2)
                right_fit = np.poly1d(coeffs)
            except (np.RankWarning, np.linalg.LinAlgError):
                pass

        # -- Kalite kontrolu: yatay/bozuk fit'leri reddet --
        if left_fit is not None:
            if not self._is_fit_quality_ok(left_fit, left_y_range, h, w):
                left_fit = None
        if right_fit is not None:
            if not self._is_fit_quality_ok(right_fit, right_y_range, h, w):
                right_fit = None

        # -- Yakinlik kontrolu: SERT KURAL --
        # Iki serit MIN_LANE_WIDTH_PX'den daha yakin OLAMAZ
        if left_fit is not None and right_fit is not None:
            y_check_points = [int(h * 0.15), int(h * 0.35), int(h * 0.55),
                              int(h * 0.75), h - 1]
            overlap = False
            for y_pt in y_check_points:
                gap = right_fit(y_pt) - left_fit(y_pt)
                if gap < MIN_LANE_WIDTH_PX:
                    overlap = True
                    break
            if overlap:
                # Daha az pikseli olan fit'i sil
                if len(left_lane_inds) >= len(right_lane_inds):
                    right_fit = None
                else:
                    left_fit = None

        # -- Genislik kontrolu: SERT KURAL --
        # || + dis sag gibi yanlis cifti onlemek icin maksimum genislik siniri
        if left_fit is not None and right_fit is not None:
            y_mid = int(h * 0.55)
            gap_mid = right_fit(y_mid) - left_fit(y_mid)
            if gap_mid > MAX_LANE_WIDTH_PX:
                # Yanlis cift — az pikseli olan fit'i sil
                if len(left_lane_inds) >= len(right_lane_inds):
                    right_fit = None
                else:
                    left_fit = None

        # ============================================================
        # POLINOM STABILIZASYONU (EMA)
        # ============================================================
        is_single = (left_fit is None) != (right_fit is None)
        alpha = 0.88 if is_single else POLY_ALPHA   # 0.92 -> 0.88: single mode da hizlandirildi

        left_fit  = self._smooth_fit(left_fit,  self._left_fit,  alpha)
        right_fit = self._smooth_fit(right_fit, self._right_fit, alpha)

        self._left_fit  = left_fit
        self._right_fit = right_fit

        # -- Son bilinen pozisyonu guncelle (stale reset destekli) --
        if left_fit is not None:
            self._last_left_x = int(np.clip(left_fit(h - 1), 0, w - 1))
            self._left_miss_count = 0
        else:
            self._left_miss_count += 1
            if self._left_miss_count > 5:
                self._last_left_x = None

        if right_fit is not None:
            self._last_right_x = int(np.clip(right_fit(h - 1), 0, w - 1))
            self._right_miss_count = 0
        else:
            self._right_miss_count += 1
            if self._right_miss_count > 5:
                self._last_right_x = None

        # ============================================================
        # VIZUALIZASYON
        # ============================================================
        if build_viz and viz is not None:
            y_near = int(h * 0.75)
            y_far  = int(h * 0.25)  # viz: kontrol ile tutarli

            if left_fit is not None:
                ly_min, ly_max = left_y_range
                plot_y = np.arange(ly_min, ly_max + 1, dtype=np.int32)
                lx_pts = np.clip(left_fit(plot_y), 0, w - 1).astype(np.int32)
                if len(plot_y) > 1:
                    pts = np.column_stack((lx_pts, plot_y)).reshape(-1, 1, 2)
                    cv2.polylines(viz, [pts], False, (255, 180, 0), 3)
                for yy in [y_near, y_far]:
                    xx = int(np.clip(left_fit(yy), 0, w - 1))
                    cv2.circle(viz, (xx, yy), 8, (255, 0, 0), -1)

            if right_fit is not None:
                ry_min, ry_max = right_y_range
                plot_y = np.arange(ry_min, ry_max + 1, dtype=np.int32)
                rx_pts = np.clip(right_fit(plot_y), 0, w - 1).astype(np.int32)
                if len(plot_y) > 1:
                    pts = np.column_stack((rx_pts, plot_y)).reshape(-1, 1, 2)
                    cv2.polylines(viz, [pts], False, (0, 180, 255), 3)
                for yy in [y_near, y_far]:
                    xx = int(np.clip(right_fit(yy), 0, w - 1))
                    cv2.circle(viz, (xx, yy), 8, (0, 140, 255), -1)

            # Merkez cizgileri + mode text
            # SARI: eski yanlis merkez (bev_w/2)   YESIL: gercek arac pozisyonu (BEV_CAR_X)
            cv2.line(viz, (w // 2, 0), (w // 2, h), (0, 100, 100), 1)  # Eski (soluk)
            car_x_int = int(BEV_CAR_X)
            cv2.line(viz, (car_x_int, 0), (car_x_int, h), (0, 255, 0), 2)  # Gercek (parlak yesil)
            mode = "DUAL" if left_fit and right_fit else "SINGLE" if left_fit or right_fit else "LOST"
            mode_col = (0, 255, 0) if mode == "DUAL" else (0, 255, 255) if mode == "SINGLE" else (0, 0, 255)
            cv2.putText(viz, mode, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_col, 2)

        return left_fit, right_fit, left_y_range, right_y_range, viz

    def _find_lane_peaks(self, histogram, midpoint):
        """Histogram'dan peak'leri bul, yakin olanlari clusterla,
        | || | yapisinda dogru serit cizgilerini sec.
        Dondurur: (left_base, right_base, left_valid, right_valid)"""
        w = len(histogram)

        # Tum peak'leri bul (local maxima + minimum yukseklik)
        peaks = []
        for i in range(1, w - 1):
            if (histogram[i] > histogram[i-1] and
                histogram[i] >= histogram[i+1] and
                histogram[i] > PEAK_MIN_HEIGHT):
                peaks.append((i, histogram[i]))

        if not peaks:
            # Peak yok, klasik argmax fallback
            lb = int(np.argmax(histogram[:midpoint]))
            rb = int(np.argmax(histogram[midpoint:])) + midpoint
            return lb, rb, histogram[lb] > MIN_PIX * 2, histogram[rb] > MIN_PIX * 2

        # Peak'leri x pozisyonuna gore sirala
        peaks.sort(key=lambda p: p[0])

        # Yakin peak'leri clusterla (|| cift cizgi birlestirme)
        clusters = []
        current_cluster = [peaks[0]]
        for i in range(1, len(peaks)):
            if peaks[i][0] - current_cluster[-1][0] <= PEAK_CLUSTER_GAP:
                current_cluster.append(peaks[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [peaks[i]]
        clusters.append(current_cluster)

        # Her cluster icin merkez hesapla.
        # Tek peak (|) → direkt kullan
        # Coklu peak (||) → agirlikli ortalama
        cluster_centers = []
        for c in clusters:
            if len(c) == 1:
                cluster_centers.append(c[0])
            else:
                total_w = sum(p[1] for p in c)
                wx = int(sum(p[0] * p[1] for p in c) / total_w)
                mh = max(p[1] for p in c)
                cluster_centers.append((wx, mh))

        # ==============================================================
        # ONCELIK 1: ONCEKI POZISYONLARA YAKINLIK ILE PEAK SECIMI
        # ==============================================================
        # NEDEN: | || | yapisinda || cizgisi BEV midpoint (300) etrafinda
        # geziniyor. Midpoint bazli sol/sag ayirimi yapilirsa, || her
        # midpoint gecisinde serit ATLIYOR — duz yolda bile sapma yapiyor!
        # COZUM: Onceki frame'deki serit pozisyonlarina en yakin peak'leri sec.
        if (self._last_left_x is not None and self._last_right_x is not None
                and len(cluster_centers) >= 2):
            best_left = min(cluster_centers, key=lambda p: abs(p[0] - self._last_left_x))
            best_right = min(cluster_centers, key=lambda p: abs(p[0] - self._last_right_x))

            # Ayni peak'e kilitlenme + sol<sag + min genislik kontrol
            if (best_left[0] != best_right[0]
                    and best_left[0] < best_right[0]
                    and (best_right[0] - best_left[0]) >= STRICT_PAIR_MIN_GAP_PX):
                left_base = best_left[0]
                right_base = best_right[0]
                left_valid = best_left[1] > MIN_PIX * 2
                right_valid = best_right[1] > MIN_PIX * 2
                return left_base, right_base, left_valid, right_valid

        # ==============================================================
        # FALLBACK: Midpoint bazli sol/sag ayirimi (ilk frame veya pozisyon bilinmiyor)
        # ==============================================================
        left_candidates  = [(x, h) for x, h in cluster_centers if x < midpoint]
        right_candidates = [(x, h) for x, h in cluster_centers if x >= midpoint]

        if left_candidates:
            left_pick = max(left_candidates, key=lambda p: p[0])
        else:
            lb = int(np.argmax(histogram[:midpoint]))
            left_pick = (lb, histogram[lb])

        if right_candidates:
            right_pick = min(right_candidates, key=lambda p: p[0])
        else:
            rb = int(np.argmax(histogram[midpoint:])) + midpoint
            right_pick = (rb, histogram[rb])

        if abs(right_pick[0] - left_pick[0]) < MIN_LANE_WIDTH_PX:
            if len(left_candidates) > 1:
                left_pick = min(left_candidates, key=lambda p: p[0])
            if len(right_candidates) > 1:
                right_pick = max(right_candidates, key=lambda p: p[0])

        left_base = left_pick[0]
        right_base = right_pick[0]
        left_valid = left_pick[1] > MIN_PIX * 2
        right_valid = right_pick[1] > MIN_PIX * 2

        return left_base, right_base, left_valid, right_valid

    def _quick_window_search(self, binary, nonzero_x, nonzero_y,
                              base_x, h, w, win_h):
        """Hizli pencere arama: verilen base_x'ten baslayarak
        alttan uste serit pikselleri topla ve polyfit yap.
        Dondurur: (fit, (y_min, y_max)) veya (None, (0, h))"""
        current = base_x
        inds = []
        margin = WINDOW_MARGIN
        for win_idx in range(N_WINDOWS):
            y_low = h - (win_idx + 1) * win_h
            y_high = h - win_idx * win_h
            x_lo = max(0, current - margin)
            x_hi = min(w, current + margin)
            good = ((nonzero_y >= y_low) & (nonzero_y < y_high) &
                    (nonzero_x >= x_lo) & (nonzero_x < x_hi)).nonzero()[0]
            inds.append(good)
            if len(good) > MIN_PIX:
                current = int(np.mean(nonzero_x[good]))
        all_inds = np.concatenate(inds) if inds else np.array([], dtype=int)
        if len(all_inds) < MIN_PIX * 3:
            return None, (0, h)
        fx = nonzero_x[all_inds]
        fy = nonzero_y[all_inds]
        y_range = (int(fy.min()), int(fy.max()))
        try:
            coeffs = np.polyfit(fy, fx, 2)
            return np.poly1d(coeffs), y_range
        except (np.RankWarning, np.linalg.LinAlgError):
            return None, (0, h)

    def _is_fit_quality_ok(self, fit, y_range, h, w):
        """Polinom fit'in kalitesini kontrol et.
        Yatay seritler (viraj cikisinda BEV'de yataylasan cizgiler)
        bu kontrolle reddedilir.

        Kontroller:
        1. Dikey kapsama: Fit, BEV yuksekliginin en az %30'unu kapsamali
        2. Yataylik orani: x degisimi / y degisimi cok buyukse yatay
        3. Egim kontrolu: Polinomun alt noktadaki egimi cok dusukse yatay

        Dondurur: True = kaliteli, False = reddet
        """
        if fit is None:
            return False

        y_min, y_max = y_range
        y_span = y_max - y_min

        # Kontrol 1: Dikey kapsama
        if y_span < h * FIT_MIN_VERTICAL_SPAN:
            return False

        # Kontrol 2: Yataylik orani
        x_at_ymin = fit(y_min)
        x_at_ymax = fit(y_max)
        x_span = abs(x_at_ymax - x_at_ymin)

        if y_span > 0 and x_span / y_span > FIT_MAX_HORIZONTAL_RATIO:
            return False

        # Kontrol 3: Egim — polinomun turevini alt noktada hesapla
        # f(y) = ay^2 + by + c  =>  f'(y) = 2ay + b
        # Biz x = f(y) fonksiyonunun egimini olcuyoruz
        # Egim buyukse (|dx/dy| buyuk) serit yatay demek
        # AMA biz dy/dx'e bakmaliyiz — "seridin dikeyligi"
        if len(fit.coefficients) == 3:
            a, b, _ = fit.coefficients
            # dx/dy degeri (polinom alt noktasinda)
            y_bottom = y_max
            dxdy = abs(2 * a * y_bottom + b)
            # Eger dx/dy cok buyukse (serit yatay gidiyor)
            # dy/dx = 1/dxdy cok kucuk olur
            if dxdy > 0 and (1.0 / dxdy) < FIT_MIN_SLOPE:
                return False

        return True

    def _smooth_fit(self, new_fit, prev_fit, alpha=POLY_ALPHA):
        """Polinom katsayilarini EMA ile stabilize et.
        new_fit None ise None dondur (orijinal davranis — stale fit birikmez).
        Stale fit birikmesi _last_left_x/_last_right_x'i yanlis tutarak
        priority-1 peak secimini kilit alti yanlis hatta bagliyordu."""
        if new_fit is None:
            return None       # Fit kayboldu — temiz kayip (orijinal davranis)
        if prev_fit is None:
            return new_fit
        h = BEV_H
        new_base = new_fit(h)
        prev_base = prev_fit(h)
        if abs(new_base - prev_base) > MAX_POLY_JUMP:
            return prev_fit
        new_c = new_fit.coefficients
        prev_c = prev_fit.coefficients
        if len(new_c) != len(prev_c):
            return new_fit
        blended = alpha * np.array(new_c) + (1.0 - alpha) * np.array(prev_c)
        return np.poly1d(blended)


# +============================================================+
# |               LANE MEMORY (DYNAMIC OFFSET)                  |
# +============================================================+

class LaneMemory:
    """Serit genisligini hafizada tutarak tek serit
    goruldugunde diger seridi tahmin eder.
    TurnTracker ile entegre calisir."""

    def __init__(self, turn_tracker):
        self._lane_width = float(DEFAULT_LANE_WIDTH_PX)
        self._no_left = 0
        self._no_right = 0
        self._last_error = 0.0
        self._track_point = None
        self._track_center = None
        self._turn = turn_tracker

    def compute_error(self, left_fit, right_fit, left_y_range, right_y_range, bev_w, bev_h):
        """BEV koordinatlarinda merkez hatasi hesapla.
        Dondurur: (error_norm, has_left, has_right)
        error_norm: -1.0 .. +1.0"""
        h = bev_h
        # Arac ortadayken serit merkezi BEV'de ~bev_w/2 etrafinda gorunuyor.
        # BEV_CAR_X=326 kullanmak TERS bias yaratiyordu.
        center_x = bev_w / 2.0
        has_left = left_fit is not None
        has_right = right_fit is not None

        y_near = int(h * 0.75)

        if has_left and has_right:
            # Dual modda single-mode'dan kalan eski PID cizgisini temizle
            self._track_point  = None
            self._track_center = None

            lx = left_fit(y_near)
            rx = right_fit(y_near)
            width = rx - lx

            if width < MIN_LANE_WIDTH_PX:
                if abs(lx - center_x) > abs(rx - center_x):
                    return self._single_lane_error(
                        left_fit, None, left_y_range, right_y_range, center_x, bev_w, h)
                else:
                    return self._single_lane_error(
                        None, right_fit, left_y_range, right_y_range, center_x, bev_w, h)

            if 50 < width < MAX_LANE_WIDTH_PX:
                dw = abs(width - self._lane_width)
                if dw < MAX_LANE_WIDTH_CHANGE:
                    self._lane_width = (LANE_WIDTH_ALPHA * width +
                                        (1.0 - LANE_WIDTH_ALPHA) * self._lane_width)

            # TurnTracker'i iki-seritli veriyle guncelle
            self._turn.update_dual(left_fit, right_fit, h, bev_w)

            # === FAR-POINT LOOKAHEAD: virajlarda gec donus engellenir ===
            # Eskiden dual mode SADECE y_near (h*0.75) kullaniyordu → viraj
            # ortasinda anca tepki veriyordu. Single mode'daki gibi y_far
            # noktasini da hesaba katarak erken tepki saglariz.
            y_far = int(h * 0.25)   # 0.20->0.25: cok uzak bakis erken donus yapiyor, orta mesafe
            lx_f = left_fit(y_far)
            rx_f = right_fit(y_far)

            lane_center_n = (lx + rx) / 2.0
            lane_center_f = (lx_f + rx_f) / 2.0

            error_n = (lane_center_n - center_x) / (bev_w / 2.0)
            error_f = (lane_center_f - center_x) / (bev_w / 2.0)

            # Viraj derinlestikce uzak noktaya daha cok agirlik ver
            curve_strength = min(1.0, abs(self._turn.curvature) / 0.002)
            far_weight = 0.25 + 0.55 * curve_strength  # 0.25 (duz) .. 0.80 (viraj)
            error = far_weight * error_f + (1.0 - far_weight) * error_n

            # Polinom egriliginden ek viraj destegi (feed-forward)
            # Dual mode'a single mode'daki curve_boost mantigi getirildi.
            if abs(self._turn.curvature) > TURN_CURVATURE_THR:
                curve_boost = np.sign(self._turn.curvature) * min(
                    0.35, abs(self._turn.curvature) * 50)  # cap 0.55->0.35, mult 100->50
                error += curve_boost

            self._no_left = 0
            self._no_right = 0
            self._last_error = max(-1.0, min(1.0, error))
            return self._last_error, True, True

        # =================================================================
        # TEK SERIT MODU — AKILLI VIRAJ TAKIBI
        # =================================================================
        if has_left and not has_right:
            return self._single_lane_error(
                left_fit, None, left_y_range, right_y_range, center_x, bev_w, h)

        if has_right and not has_left:
            return self._single_lane_error(
                None, right_fit, left_y_range, right_y_range, center_x, bev_w, h)

        # Iki serit de kayip
        self._no_left += 1
        self._no_right += 1
        self._turn.update_lost()
        return self._last_error, False, False

    def _single_lane_error(self, left_fit, right_fit, left_y_range, right_y_range,
                            center_x, bev_w, h):
        """Tek serit modu — TurnTracker entegreli.

        Anahtar fark: Gorunen seridin SOL mu SAG mi oldugunu
        TurnTracker'dan sorar (son bilinen pozisyonlara gore).
        Boylece '||' orta cizgi her zaman dogru tarafa atanir.
        """
        self._track_point = None
        self._track_center = None

        visible_fit = left_fit if left_fit is not None else right_fit
        if visible_fit is None:
            return self._last_error, False, False

        # Tracker'in tek-serit slotu (left_fit/right_fit) ilk karar icin guclu ipucu.
        tracker_is_left = left_fit is not None
        tracker_side = 'left' if tracker_is_left else 'right'

        # Seridin gercekten SOL mu SAG mi oldugunu TurnTracker'dan sor
        # ONCE classify yap, SONRA update_single'a dogru bilgiyi gonder
        # (SlidingWindowTracker'in sol/sag atamasi guvenilmez, TurnTracker karar verir)
        lane_side = self._turn.classify_single_lane(visible_fit, h, bev_w,
                                tracker_hint=tracker_side)

        # TurnTracker'a classify sonucuna gore bildir
        is_left_side = (lane_side == 'left')
        self._turn.update_single(visible_fit, is_left_side, h, bev_w)

        # Serit araligi — hangi fit doluysa onun range'ini kullan
        y_range = left_y_range if tracker_is_left else right_y_range
        y_min, y_max = y_range

        # Yakin ve uzak olcum noktalari (seridin gorunen araligiyla sinirli)
        y_near = min(int(h * 0.80), y_max)
        y_far  = max(int(h * 0.25), y_min)  # 0.20->0.25: orta mesafe bakis

        vx_near = visible_fit(y_near)
        vx_far  = visible_fit(y_far)

        if lane_side == 'left':
            # Gorunen serit SOL -> Arac merkezini SAGA offset
            lane_center_near = vx_near + self._lane_width / 2.0
            lane_center_far  = vx_far  + self._lane_width / 2.0
        else:
            # Gorunen serit SAG -> Arac merkezini SOLA offset
            lane_center_near = vx_near - self._lane_width / 2.0
            lane_center_far  = vx_far  - self._lane_width / 2.0

        # Yakin ve uzak hatayi birlestir
        error_near = (lane_center_near - center_x) / (bev_w / 2.0)
        error_far  = (lane_center_far  - center_x) / (bev_w / 2.0)

        # Viraj derinlestikce uzak noktaya daha cok agirlik ver
        curve_strength = min(1.0, abs(self._turn.curvature) / 0.002)
        far_weight = 0.25 + 0.55 * curve_strength  # 0.25 (duz) .. 0.80 (viraj)
        error = far_weight * error_far + (1.0 - far_weight) * error_near

        # Polinom egriliginden ek viraj destegi
        # cap 0.35->0.50, carpan 33->100: tipik egrilik (0.002-0.004) icin anlamli boost saglar
        if abs(self._turn.curvature) > TURN_CURVATURE_THR:
            curve_boost = np.sign(self._turn.curvature) * min(
                0.50, abs(self._turn.curvature) * SINGLE_LANE_CURVE_GAIN * 100)
            error += curve_boost

        # Tek seritte virajdan cikmadan donusu korumak icin
        # TurnTracker hafizasini error'a adaptif sekilde karistir.
        # mem_weight 0.30->0.50 baslangic: single moda gecince hafiza hemen etkili olsun
        if self._turn.frames_since_dual < TURN_MEMORY_FRAMES:
            mem_dir = self._turn.turn_direction
            if mem_dir == 0:
                mem_dir = 1 if error_far >= 0 else -1

            mem_steer = mem_dir * SINGLE_LANE_BASE_STEER
            mem_steer += mem_dir * min(
                0.30, abs(self._turn.curvature) * SINGLE_LANE_CURVE_GAIN * 80)

            age = min(1.0, self._turn.frames_since_dual / float(max(1, TURN_MEMORY_FRAMES)))
            mem_weight = 0.50 + 0.20 * age   # 0.50 -> 0.70
            error = (1.0 - mem_weight) * error + mem_weight * mem_steer

        error = max(-1.0, min(1.0, error))

        if is_left_side:
            self._no_left = 0
            self._no_right += 1
        else:
            self._no_left += 1
            self._no_right = 0

        self._last_error = error
        self._track_point = (int(vx_near), int(y_near))
        self._track_center = (int(lane_center_near), int(y_near))

        return self._last_error, is_left_side, not is_left_side

    @property
    def lane_width(self):
        return self._lane_width


# +============================================================+
# |                  CIZIM FONKSIYONLARI                        |
# +============================================================+

def draw_camera_overlay(frame, left_fit, right_fit, steering,
                        has_left, has_right, autonomous, y_offset,
                        proc_h, proc_w, scale_inv, track_point=None, track_center=None):
    """Kamera goruntusu uzerine serit overlay + HUD."""
    h, w = frame.shape[:2]
    out = frame.copy()
    y_roi = y_offset

    # MERKEZ DIKEY CIZGI — sari, ekranin tam ortasi
    cv2.line(out, (w // 2, y_roi), (w // 2, h), (0, 255, 255), 2)

    # BEV'den serit overlay olustur ve geri yansit
    if left_fit is not None or right_fit is not None or track_center is not None:
        overlay_bev = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)
        plot_y = np.linspace(0, BEV_H - 1, BEV_H)

        if left_fit is not None and right_fit is not None:
            lx = np.clip(left_fit(plot_y), 0, BEV_W - 1).astype(int)
            rx = np.clip(right_fit(plot_y), 0, BEV_W - 1).astype(int)
            pts_l = np.column_stack((lx, plot_y.astype(int)))
            pts_r = np.column_stack((rx, plot_y.astype(int)))
            pts = np.vstack((pts_l, pts_r[::-1]))
            cv2.fillPoly(overlay_bev, [pts], (0, 180, 0))

        # Tek Serit Modu Icin Nokta Cizimi
        if track_point is not None and track_center is not None:
            # Algilanan serit uzerindeki gercek nokta (Sari)
            cv2.circle(overlay_bev, track_point, 10, (0, 255, 255), -1)
            # Aracin gitmesi gereken sanal orta nokta hedefi (Yesil)
            cv2.circle(overlay_bev, track_center, 12, (0, 255, 0), -1)
            # Aracin mevcut pozisyonundan hedefe yon cizgisi (Kirmizi)
            cv2.line(overlay_bev, (BEV_W // 2, BEV_H), track_center, (0, 0, 255), 3)

        if left_fit is not None:
            lx = np.clip(left_fit(plot_y), 0, BEV_W - 1).astype(int)
            for i in range(len(plot_y) - 1):
                cv2.line(overlay_bev,
                         (lx[i], int(plot_y[i])),
                         (lx[i+1], int(plot_y[i+1])),
                         (255, 80, 0), 3)

        if right_fit is not None:
            rx = np.clip(right_fit(plot_y), 0, BEV_W - 1).astype(int)
            for i in range(len(plot_y) - 1):
                cv2.line(overlay_bev,
                         (rx[i], int(plot_y[i])),
                         (rx[i+1], int(plot_y[i+1])),
                         (0, 80, 255), 3)

        # BEV overlay'i orijinal perspektife geri cevir
        unwarped = bev_unwarp(overlay_bev, (proc_w, proc_h))
        # Isleme boyutundan orijinal boyuta scale
        if scale_inv != 1.0:
            unwarped = cv2.resize(unwarped, (frame.shape[1], frame.shape[0] - y_offset))

        # Overlay'i frame'e ekle (sadece crop bolgesi)
        mask = (unwarped[:, :, 0] > 0) | (unwarped[:, :, 1] > 0) | (unwarped[:, :, 2] > 0)
        roi = out[y_offset:y_offset + unwarped.shape[0], :unwarped.shape[1]]
        if roi.shape[:2] == unwarped.shape[:2]:
            roi[mask] = cv2.addWeighted(roi, 0.6, unwarped, 0.4, 0)[mask]

    # Direksiyon bari (ust orta)
    bx, by, bl = w // 2, 18, w // 4
    sx = int(bx + steering * bl)
    cv2.rectangle(out, (bx - bl, by - 8), (bx + bl, by + 8), (40, 40, 40), -1)
    cv2.line(out, (bx, by), (sx, by), (0, 200, 255), 5)
    cv2.circle(out, (sx, by), 7, (0, 200, 255), -1)
    cv2.line(out, (bx, by - 8), (bx, by + 8), (200, 200, 200), 1)

    # Durum metni
    l_txt = "OK" if has_left else "--"
    r_txt = "OK" if has_right else "--"
    mode = "OTONOM" if autonomous else "MANUAL"
    mcol = (0, 255, 0) if autonomous else (0, 0, 255)
    cv2.putText(out, mode, (w - 200, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.90, mcol, 3)
    cv2.putText(out, f"L:{l_txt}  R:{r_txt}  S:{steering:+.2f}",
                (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    return out


# +============================================================+
# |                      ANA DONGU                              |
# +============================================================+

def main():
    motors        = None
    autonomous    = False
    last_error    = 0.0
    steering      = 0.0
    current_speed = 0.0
    no_line_count = 0
    crossing_cooldown = 0
    pid           = PIDController(Kp, Ki, Kd, INTEGRAL_LIMIT, D_FILTER_ALPHA)
    turn_tracker  = TurnTracker()
    lane_mem      = LaneMemory(turn_tracker)
    sw_tracker    = SlidingWindowTracker()
    bright_thr    = BRIGHT_THR
    frame_num     = 0
    fps_t0        = time.monotonic()
    fps_val       = 0.0

    try:
        motors = MotorController()
        motors.steering.center()
        print("[INFO] Motor hazir.")
    except Exception as e:
        print(f"[WARN] Motor yok ({e}), sadece goruntu modu.")

    # -- Kamera Secimi --
    print("\n" + "="*50)
    print("  KAMERA SECIMI")
    print("="*50)
    print("  1) usbcam  -- USB Kamera (cv2.VideoCapture)")
    print("  2) rpi     -- Raspberry Pi Kamera (Picamera2)")
    print("="*50)
    cam_choice = input("Secim [usbcam/rpi]: ").strip().lower()
    use_usb = cam_choice in ("usbcam", "usb", "1")

    cam = None
    cap = None

    if use_usb:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        cap.set(cv2.CAP_PROP_FPS, 40)
        if not cap.isOpened():
            print("[HATA] USB kamera acilamadi!"); sys.exit(1)
        print("[INFO] USB Kamera hazir.")
    else:
        if not HAS_PICAMERA2:
            print("[HATA] Picamera2 bulunamadi!"); sys.exit(1)
        cam = Picamera2()
        cam.configure(cam.create_video_configuration(
            main={"size": (CAM_W, CAM_H), "format": "RGB888"},
            buffer_count=2,
            controls={"FrameRate": 40}
        ))
        cam.start()
        time.sleep(0.5)
        print("[INFO] RPi Kamera hazir.")

    print("[INFO] Tuslar: 'a'=otonom  SPACE=dur  '+'/'-'=esik  'q'=cik")

    def _shutdown(signum=None, frm=None):
        nonlocal autonomous
        autonomous = False
        if motors: motors.close()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    cv2.namedWindow("Kamera",    cv2.WINDOW_NORMAL)
    cv2.namedWindow("Algilama",  cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Kamera",   640, 480)
    cv2.resizeWindow("Algilama", BEV_W, BEV_H)
    cv2.moveWindow("Kamera",    0,   0)
    cv2.moveWindow("Algilama",  0, 510)

    last_det_view = None

    try:
        while True:
            if use_usb:
                ret, frame = cap.read()
                if not ret:
                    print("[HATA] USB kameradan frame okunamadi!"); break
            else:
                img   = cam.capture_array("main")
                frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            frame_num += 1

            # FPS hesabi
            if frame_num % 20 == 0:
                now = time.monotonic()
                fps_val = 20.0 / max(now - fps_t0, 0.001)
                fps_t0 = now

            h_img = frame.shape[0]
            y_off = h_img // CROP_DIV
            bottom = frame[y_off:, :]
            bh_orig, bw_orig = bottom.shape[:2]

            # -- Resize (FPS opt) --
            scale = PROCESS_W / bw_orig
            proc_w = PROCESS_W
            proc_h = int(bh_orig * scale)
            small = cv2.resize(bottom, (proc_w, proc_h),
                               interpolation=cv2.INTER_LINEAR)

            do_viz = (frame_num % VIZ_SKIP == 0)

            # =====================================================
            # ==  BEV + SLIDING WINDOW PIPELINE  ==================
            # =====================================================

            # 1) Bird's Eye View donusumu
            warped = bev_warp(small)

            # 2) Beyaz maske + yaya gecidi filtresi
            mask = create_lane_mask(warped, bright_thr)

            # 2b) Yaya gecidi filtresi — virajda bypass, duz yolda aktif
            in_turn = turn_tracker.turn_direction != 0
            mask, crossing_row_count = filter_pedestrian_crossing(mask, turn_active=in_turn)
            crossing_now = crossing_row_count >= CROSSING_MIN_ROWS
            if crossing_now:
                crossing_cooldown = CROSSING_COOLDOWN
            elif crossing_cooldown > 0:
                crossing_cooldown -= 1
            crossing_active = crossing_cooldown > 0

            # 3) Sliding window ile serit takibi
            left_fit, right_fit, left_y_range, right_y_range, viz_bev = sw_tracker.search(mask, build_viz=do_viz)

            # 4) Merkez hatasi hesapla (dynamic offset + || duzeltmesi)
            raw_error, has_left, has_right = lane_mem.compute_error(
                left_fit, right_fit, left_y_range, right_y_range, BEV_W, BEV_H)

            # -- VIRAJ CIKISI KURTARMA --
            # Iki serit de kayip AMA TurnTracker virajda oldugumuzu biliyor
            if not has_left and not has_right:
                td = turn_tracker.turn_direction
                if td != 0 and turn_tracker.frames_since_dual < TURN_MEMORY_FRAMES:
                    curve_steer = td * SINGLE_LANE_BASE_STEER
                    curve_steer += td * min(0.20, abs(turn_tracker.curvature) * SINGLE_LANE_CURVE_GAIN * 40)
                    raw_error = max(-0.80, min(0.80, curve_steer))

            # -- Cizgi kaybi takibi --
            any_line = has_left or has_right
            if any_line:
                no_line_count = 0
            else:
                no_line_count += 1

            # -- Error EMA yumusatma --
            if any_line:
                error = ALPHA * raw_error + (1.0 - ALPHA) * last_error
                last_error = error
            else:
                error = last_error

            # -- Cizgi kayip guvenligi --
            if no_line_count > NO_LINE_TIMEOUT:
                error *= NO_LINE_STEER_DECAY

            steering = pid.update(error)

            # -- Adaptif hiz --
            current_speed = BASE_SPEED * (1.0 - SPEED_CURVE_FACTOR * abs(steering))
            current_speed = max(MIN_SPEED, current_speed)

            if no_line_count > NO_LINE_TIMEOUT:
                current_speed *= NO_LINE_SPEED_MULT

            # -- Yaya gecidi yavaslamasi --
            if crossing_active:
                current_speed = min(current_speed, CROSSING_SPEED)

            # -- Motor kontrolu --
            if autonomous and motors:
                motors.steering.angle = steering * SERVO_MAX_ANGLE
                motors.set_speed(current_speed)

            # =====================================================
            # ==  VIZUALIZASYON  ==================================
            # =====================================================

            # -- Goruntu Ciktisi --
            tp = getattr(lane_mem, '_track_point', None)
            tc = getattr(lane_mem, '_track_center', None)
            cam_view = draw_camera_overlay(
                frame, left_fit, right_fit, steering,
                has_left, has_right, autonomous, y_off,
                proc_h, proc_w, scale_inv=1.0/scale,
                track_point=tp, track_center=tc)

            # HUD: FPS + hata detaylari
            cv2.putText(cam_view, f"FPS:{fps_val:.0f}",
                        (cam_view.shape[1] - 160, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.90, (180, 180, 180), 2)
            td = turn_tracker.turn_direction
            td_txt = "SOL" if td < 0 else "SAG" if td > 0 else "DUZ"
            td_col = (255, 100, 0) if td < 0 else (0, 100, 255) if td > 0 else (200, 200, 200)
            soff = motors.steering.offset if motors else 0.0
            mode_str = "DUAL" if (has_left and has_right) else "SINGLE" if (has_left or has_right) else "LOST"
            cv2.putText(cam_view,
                        f"E:{error:+.3f}  STR:{steering:+.3f}  {mode_str}",
                        (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (200, 200, 0), 3)
            locked = turn_tracker._locked_side or "-"
            cv2.putText(cam_view,
                        f"OFF:{soff:+.1f}  W:{lane_mem.lane_width:.0f}  V:{current_speed:.2f}  LOCK:{locked}",
                        (8, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.90, td_col, 3)
            if no_line_count > NO_LINE_TIMEOUT:
                cv2.putText(cam_view, "!! CIZGI YOK !!",
                            (cam_view.shape[1]//2 - 140, cam_view.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            if crossing_active:
                cv2.putText(cam_view, "SERIT CIZGI BOZUNUMU ALGILANDI",
                            (cam_view.shape[1]//2 - 320, cam_view.shape[0]//2 + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
            cv2.imshow("Kamera", cam_view)

            # -- Pencere 2: Algilama (BEV + sliding windows) --
            if do_viz and viz_bev is not None:
                last_det_view = viz_bev
            if last_det_view is not None:
                cv2.imshow("Algilama", last_det_view)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('a'):
                autonomous = not autonomous
                if autonomous and motors:
                    motors.set_speed(BASE_SPEED)
                    print(f"[INFO] Otonom ACIK >  hiz={BASE_SPEED:.2f}")
                else:
                    if motors: motors.stop()
                    pid.reset()
                    last_error = 0.0
                    no_line_count = 0
                    print("[INFO] Otonom KAPALI")
            elif key == ord(' '):
                autonomous = False
                if motors: motors.emergency_stop()
                pid.reset()
                no_line_count = 0
                print("[INFO] ACIL DURUS")
            elif key in (ord('+'), ord('=')):
                bright_thr = min(254, bright_thr + 5)
                print(f"[INFO] BRIGHT_THR -> {bright_thr}")
            elif key == ord('-'):
                bright_thr = max(1, bright_thr - 5)
                print(f"[INFO] BRIGHT_THR -> {bright_thr}")
            elif key == ord('x') and motors:
                motors.steering.offset += 1.0
                print(f"[INFO] SERVO_OFFSET -> {motors.steering.offset:+.1f} (saga kaydir)")
            elif key == ord('z') and motors:
                motors.steering.offset -= 1.0
                print(f"[INFO] SERVO_OFFSET -> {motors.steering.offset:+.1f} (sola kaydir)")
            elif key != 255 and key != 0:
                print(f"[KEY] code={key} chr={chr(key) if 32 <= key < 127 else '?'}")

    finally:
        autonomous = False
        if motors: motors.close()
        if cam: cam.stop()
        if cap: cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
