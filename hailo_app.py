import sys, os, threading, time, signal
import numpy as np
import cv2

try:
    from picamera2 import Picamera2
except ImportError:
    print("[HATA] Picamera2 bulunamadı!"); sys.exit(1)

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp
Gst.init(None)

from config import (
    HEF_PATH, MODEL_W, MODEL_H,
    CLASS_NAMES, CLASS_COLORS, QP_SCALE, QP_ZP,
    IOU_THRESHOLD, TOP_K,
    SHOW_FPS, SHOW_LABELS, SHOW_CONFIDENCE,
    WINDOW_TITLE, DISPLAY_W, DISPLAY_H,
    DEBUG_SO, POST_FUNC
)
from motor_control import MotorController
from lane_follower import LaneFollower

# --- NPY yolları: /dev/shm (RAM disk → sıfır disk I/O) ---
_SHM = "/dev/shm"
_BOX_PATH   = os.path.join(_SHM, "model_concat18.npy")
_COEF_PATH  = os.path.join(_SHM, "model_concat17.npy")
_PROTO_PATH = os.path.join(_SHM, "model_conv48.npy")

# --- Per-tensor quantization params ---
# concat18 (box+cls): QP_ZP/QP_SCALE from config
# concat17 (mask coef): zp=169 (median), scale=0.005
# conv48 (proto): zp=8 (median), scale=0.005
_COEF_ZP, _COEF_SC = 169.0, 0.005
_PROTO_ZP, _PROTO_SC = 8.0, 0.005

# --- Sabitler ---
# Uint8 threshold: background median ~155, gerçek detection 210+
# 200 → ~35 aday, NMS sonrası ~5-15
UINT8_SCORE_THR = 200
MIN_CONFIDENCE  = 0.70   # %70 altındaki tespitler atılır
_REG, _STRIDES = 16, [8, 16, 32]
_BINS = np.arange(_REG, dtype=np.float32)

# Anchor grid (52x52 + 26x26 + 13x13 = 3549)
_anchors_cx, _anchors_cy, _anchor_sw, _anchor_sh = [np.empty(3549, np.float32) for _ in range(4)]
_pos = 0
for _s in _STRIDES:
    _h, _w = MODEL_H // _s, MODEL_W // _s
    _ys = (np.arange(_h) + 0.5) / _h
    _xs = (np.arange(_w) + 0.5) / _w
    _cy, _cx = np.meshgrid(_ys.astype(np.float32), _xs.astype(np.float32), indexing="ij")
    _n = _h * _w
    _anchors_cx[_pos:_pos+_n] = _cx.ravel()
    _anchors_cy[_pos:_pos+_n] = _cy.ravel()
    _anchor_sw[_pos:_pos+_n] = _s / MODEL_W
    _anchor_sh[_pos:_pos+_n] = _s / MODEL_H
    _pos += _n


def _nms(boxes, scores, iou_thr):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        order = order[1:][iou <= iou_thr]
    return keep


def _dq(arr_u8, zp, scale):
    return (arr_u8.astype(np.float32) - zp) * scale


def decode_yolov8_seg(box_u8, coef_u8, proto_u8):
    data = box_u8.reshape(-1, 66)
    coefs = coef_u8.reshape(-1, 32)
    protos = proto_u8.reshape(104, 104, 32)

    # --- UINT8 threshold (sigmoid atlanır, daha güvenilir) ---
    cls_u8 = data[:, 64:66]
    cls_max_u8 = cls_u8.max(axis=1)
    cls_ids = cls_u8.argmax(axis=1)

    idx = np.where(cls_max_u8 >= UINT8_SCORE_THR)[0]
    if len(idx) == 0:
        return []
    if len(idx) > TOP_K:
        idx = idx[cls_max_u8[idx].argsort()[::-1][:TOP_K]]

    # Confidence: basit normalize (0-255 → 0-1, background median=155)
    scores = (cls_max_u8[idx].astype(np.float32) - 155.0) / (229.0 - 155.0)
    scores = scores.clip(0, 1)
    cids = cls_ids[idx]

    # Float confidence filtre — düşük güvenilirlik tespitlerini at
    conf_ok = scores >= MIN_CONFIDENCE
    if not conf_ok.any():
        return []
    idx = idx[conf_ok]
    scores = scores[conf_ok]
    cids = cids[conf_ok]

    # Box DFL decode (uint8 → dequant → softmax → offset)
    reg_u8 = data[idx, :64].astype(np.float32)
    reg_f = _dq(data[idx, :64], QP_ZP, QP_SCALE).reshape(-1, 4, _REG)
    soft = np.exp(reg_f - reg_f.max(axis=2, keepdims=True))
    soft /= soft.sum(axis=2, keepdims=True)
    offs = (soft * _BINS).sum(axis=2)

    x1 = (_anchors_cx[idx] - offs[:, 0] * _anchor_sw[idx]).clip(0, 1)
    y1 = (_anchors_cy[idx] - offs[:, 1] * _anchor_sh[idx]).clip(0, 1)
    x2 = (_anchors_cx[idx] + offs[:, 2] * _anchor_sw[idx]).clip(0, 1)
    y2 = (_anchors_cy[idx] + offs[:, 3] * _anchor_sh[idx]).clip(0, 1)

    # NMS
    boxes_arr = np.stack([x1, y1, x2, y2], axis=1)
    keep = _nms(boxes_arr, scores, IOU_THRESHOLD)
    if not keep:
        return []

    # Mask üretimi (NMS sonrası → hızlı, per-tensor quant)
    ki = np.array(keep)
    k_coefs = (coefs[idx[ki]].astype(np.float32) - _COEF_ZP) * _COEF_SC
    k_proto = (protos.astype(np.float32) - _PROTO_ZP) * _PROTO_SC
    masks_raw = k_coefs @ k_proto.reshape(-1, 32).T
    masks_raw = np.clip(masks_raw, -50, 50)
    masks = 1.0 / (1.0 + np.exp(-masks_raw))

    results = []
    for i, k in enumerate(keep):
        results.append((
            (x1[k], y1[k], x2[k], y2[k]),
            scores[k],
            cids[k],
            masks[i].reshape(104, 104)
        ))
    return results


def draw_detections(frame, dets, y_offset=0.0):
    """
    y_offset: modelin başladığı yerin normalize Y konumu (0.5 = alt yarı).
    Deteksyon koordinatlarını tam kamera frame’ine doğru konumlandırır.
    """
    if not dets:
        return frame
    h, w = frame.shape[:2]
    overlay = frame.copy()
    y0 = int(y_offset * h)          # modelin başladığı piksel satırı
    mh = h - y0                     # model bölgesinin yüksekliği piksel olarak
    for (box, conf, cid, mask_104) in dets:
        col = CLASS_COLORS[cid % len(CLASS_COLORS)]
        ix1 = int(box[0] * w)
        iy1 = y0 + int(box[1] * mh)
        ix2 = int(box[2] * w)
        iy2 = y0 + int(box[3] * mh)
        if ix1 >= ix2 or iy1 >= iy2:
            continue

        # Mask’i model bölgesinin boyutuna resize et, tam frame’e yapıştır
        mask_region = cv2.resize(mask_104, (w, mh), interpolation=cv2.INTER_LINEAR)
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        by1 = max(iy1, y0) - y0
        by2 = max(min(iy2, h), y0) - y0
        if by1 < by2:
            roi_mask[y0 + by1:y0 + by2, ix1:ix2] = (
                mask_region[by1:by2, ix1:ix2] > 0.5
            ).astype(np.uint8) * 255

        mask_bool = roi_mask > 0
        overlay[mask_bool] = (overlay[mask_bool].astype(np.float32) * 0.5
                              + np.array(col, dtype=np.float32) * 0.5).astype(np.uint8)

        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, col, 2)

        if SHOW_LABELS:
            lbl = f"{CLASS_NAMES[cid]} {conf:.0%}" if SHOW_CONFIDENCE else CLASS_NAMES[cid]
            cv2.putText(overlay, lbl, (ix1, max(iy1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return overlay


# ── Globals ──
_lock = threading.Lock()
_latest_frame = None
_latest_cam_frame = None   # Kamera ham görüntüsü (fallback)
_pipeline_ok = True
_dbg_count = 0

_follower = LaneFollower(frame_w=MODEL_W, frame_h=MODEL_H)
_motors = None      # main() içinde init edilir
_autonomous = False
_follow_info = {}
_last_steer = 0.0


def _pad_probe_callback(pad, info):
    global _latest_frame, _dbg_count
    buf = info.get_buffer()
    if buf is None:
        return Gst.PadProbeReturn.OK

    res, m_info = buf.map(Gst.MapFlags.READ)
    if not res:
        return Gst.PadProbeReturn.OK
    frame = np.frombuffer(m_info.data[:MODEL_W * MODEL_H * 3], np.uint8).reshape(MODEL_H, MODEL_W, 3).copy()
    buf.unmap(m_info)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    dets = []
    has_all = (os.path.exists(_BOX_PATH) and
               os.path.exists(_COEF_PATH) and
               os.path.exists(_PROTO_PATH))

    if has_all:
        try:
            b = np.load(_BOX_PATH)
            c = np.load(_COEF_PATH)
            p = np.load(_PROTO_PATH)
            os.remove(_BOX_PATH)
            os.remove(_COEF_PATH)
            os.remove(_PROTO_PATH)
            dets = decode_yolov8_seg(b, c, p)
        except Exception as e:
            if _dbg_count < 3:
                print(f"[DBG] decode hata: {e}")

    _dbg_count += 1
    if _dbg_count <= 5:
        print(f"[DBG] frame={_dbg_count} npy={has_all} dets={len(dets)}")

    # Display: tam kamera frame üzerinde deteksiyonları alt yarıya remap ederek çiz
    with _lock:
        disp_base = _latest_cam_frame.copy() if _latest_cam_frame is not None else \
                    cv2.resize(frame, (DISPLAY_W or 640, DISPLAY_H or 480))
    disp_base = draw_detections(disp_base, dets, y_offset=0.5)

    # ── Lane following + motor kontrol ──
    global _follow_info, _last_steer
    steer, spd, info = _follower.compute(dets)
    if _autonomous and _motors:
        _motors.steering.angle = steer * 45
        _last_steer = steer
        if _dbg_count <= 30 or (_dbg_count % 60 == 0):
            print(f"[PD] steer={steer:+.3f} err={info.get('error',0):+.3f} "
                  f"lines={info.get('n_lines',0)}")
    _follow_info = info

    # ── HUD overlay ──
    h, w = disp_base.shape[:2]
    _draw_hud(disp_base, info, w, h)

    with _lock:
        _latest_frame = disp_base
    return Gst.PadProbeReturn.OK


def _draw_hud(frame, info, w, h):
    """On-screen: şerit merkezi, direksiyon, hız."""
    if not info:
        return
    cx = info.get("center", 0.5)
    err = info.get("error", 0.0)
    steer = info.get("steering", 0.0)
    spd = info.get("speed", 0.0)
    n = info.get("n_lines", 0)
    lane = info.get("lane")

    # Şerit merkez çizgisi
    px = int(cx * w)
    cv2.line(frame, (px, h // 2), (px, h), (0, 255, 255), 2)
    cv2.circle(frame, (px, h * 3 // 4), 6, (0, 255, 255), -1)

    # Seçilen şerit kenarları
    if lane:
        lx, rx = int(lane[0] * w), int(lane[1] * w)
        cv2.line(frame, (lx, h // 2), (lx, h), (255, 0, 255), 1)
        cv2.line(frame, (rx, h // 2), (rx, h), (255, 0, 255), 1)

    # Direksiyon göstergesi (üst ortada)
    bar_cx, bar_y = w // 2, 20
    bar_len = w // 4
    sx = int(bar_cx + steer * bar_len)
    cv2.line(frame, (bar_cx, bar_y), (sx, bar_y), (0, 200, 255), 4)
    cv2.circle(frame, (sx, bar_y), 6, (0, 200, 255), -1)

    # Durum metni
    mode = "OTONOM" if _autonomous else "MANUAL"
    col = (0, 255, 0) if _autonomous else (0, 0, 255)
    cv2.putText(frame, mode, (w - 110, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    cv2.putText(frame, f"S:{steer:+.2f} V:{spd:.2f} L:{n}",
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def _poll_bus(bus):
    global _pipeline_ok
    while True:
        msg = bus.pop()
        if msg is None:
            break
        t = msg.type
        if t == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            print(f"[GST HATA] {err.message}")
            if dbg:
                print(f"[GST DEBUG] {dbg}")
            _pipeline_ok = False
        elif t == Gst.MessageType.WARNING:
            w, _ = msg.parse_warning()
            print(f"[GST UYARI] {w.message}")


def _shutdown(signum=None, frame=None):
    """SIGINT/SIGTERM handler — motorları durdur."""
    global _autonomous
    _autonomous = False
    if _motors:
        try:
            _motors.close()
        except Exception:
            pass
    sys.exit(0)


def main():
    global _motors, _autonomous

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # libdebug.so CWD'ye yazar → CWD'yi /dev/shm'e çevir
    os.chdir(_SHM)
    print(f"[INFO] CWD → {os.getcwd()}")

    # Eski npy artıklarını temizle
    for f in [_BOX_PATH, _COEF_PATH, _PROTO_PATH]:
        if os.path.exists(f):
            os.remove(f)

    # Motor kontrol
    try:
        _motors = MotorController()
        print("[INFO] Motor kontrol hazır. Servo sweep testi...")
        _motors.steering.set(-0.3)
        time.sleep(0.3)
        _motors.steering.set(0.3)
        time.sleep(0.3)
        _motors.steering.center()
        time.sleep(0.2)
        print("[INFO] Servo OK.")
    except Exception as e:
        print(f"[WARN] Motor init başarısız ({e}), sadece görüntü modu.")
        _motors = None

    print("[INFO] Kamera başlatılıyor...")
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"size": (1280, 720), "format": "RGB888"},
        controls={"FrameRate": 30}
    ))
    cam.start()
    time.sleep(0.3)
    print("[INFO] Kamera hazır.")

    # Hailo cihaz erişim kontrolü — segfault'u önle
    _hailo_dev = "/dev/hailo0"
    if not os.path.exists(_hailo_dev):
        print(f"[HATA] {_hailo_dev} bulunamadı! Hailo modülü takılı mı?")
        print("[İPUCU] sudo sh -c 'echo 1 > /sys/bus/pci/rescan'")
        return
    try:
        with open(_hailo_dev, "rb") as _f:
            pass
    except (OSError, IOError) as e:
        print(f"[HATA] {_hailo_dev} açılamıyor: {e}")
        print("[İPUCU] PCIe reset deneyin:")
        print("  sudo sh -c 'echo 1 > /sys/bus/pci/devices/0001:01:00.0/remove'")
        print("  sleep 2")
        print("  sudo sh -c 'echo 1 > /sys/bus/pci/rescan'")
        return

    print("[INFO] GStreamer pipeline kuruluyor...")
    try:
        pipe = Gst.parse_launch(
            f"appsrc name=s is-live=true format=time do-timestamp=true "
            f"caps=video/x-raw,format=RGB,width={MODEL_W},height={MODEL_H},framerate=15/1 ! "
            f"queue leaky=downstream max-size-buffers=3 max-size-time=0 max-size-bytes=0 ! "
            f"videoconvert ! video/x-raw,format=RGB ! "
            f"hailonet hef-path={HEF_PATH} batch-size=1 "
            f"scheduler-timeout-ms=1000 scheduler-threshold=1 force-writable=true ! "
            f"queue leaky=no max-size-buffers=3 ! "
            f"hailofilter so-path={DEBUG_SO} function-name={POST_FUNC} ! "
            f"identity name=id ! fakesink sync=false"
        )
    except Exception as e:
        print(f"[HATA] Pipeline oluşturulamadı: {e}")
        return

    bus = pipe.get_bus()
    pipe.get_by_name("id").get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER, _pad_probe_callback
    )

    ret = pipe.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("[HATA] Pipeline başlatılamadı! Hailo cihazını kontrol edin:")
        print("  hailortcli fw-control identify")
        pipe.set_state(Gst.State.NULL)
        return
    print("[INFO] Pipeline çalışıyor.")

    def cam_loop():
        global _latest_cam_frame, _pipeline_ok
        src = pipe.get_by_name("s")
        ts = 0
        dur = 10**9 // 15   # 15 fps → Hailo-8L'i boğmamak için
        interval = 1.0 / 15
        while True:
            t0 = time.monotonic()
            try:
                img = cam.capture_array("main")
            except Exception:
                time.sleep(0.1)
                continue
            rgb_full = img
            h_img = img.shape[0]
            img_bottom = img[h_img // 2:, :]          # sadece alt yarı modele
            rgb = cv2.resize(img_bottom, (MODEL_W, MODEL_H))
            # Ham kamera görüntüsünü sakla (tam frame, display için)
            bgr = cv2.cvtColor(rgb_full, cv2.COLOR_RGB2BGR)
            with _lock:
                _latest_cam_frame = cv2.resize(bgr, (DISPLAY_W or 640, DISPLAY_H or 480))
            # Pipeline hâlâ çalışıyorsa Hailo'ya gönder
            if _pipeline_ok:
                try:
                    buf = Gst.Buffer.new_wrapped(rgb.tobytes())
                    buf.pts = ts
                    buf.duration = dur
                    ts += dur
                    src.emit("push-buffer", buf)
                except Exception as e:
                    print(f"[WARN] push-buffer hata: {e}")
                    _pipeline_ok = False
            # Rate limiting
            elapsed = time.monotonic() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

    threading.Thread(target=cam_loop, daemon=True).start()

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    fps_t0 = time.time()
    fps_cnt = 0
    print("[INFO] Tuşlar: 'a'=otonom, SPACE=acil dur, 'q'=çık")
    try:
        while True:
            _poll_bus(bus)
            with _lock:
                # AI işlenmiş frame varsa onu göster, yoksa ham kamera
                if _latest_frame is not None:
                    frame = _latest_frame.copy()
                elif _latest_cam_frame is not None:
                    frame = _latest_cam_frame.copy()
                    cv2.putText(frame, "KAMERA (AI bekleniyor...)", (8, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                else:
                    frame = None
            if frame is not None:
                fps_cnt += 1
                if SHOW_FPS and (fps_cnt % 15 == 0):
                    fps = fps_cnt / (time.time() - fps_t0 + 1e-9)
                    cv2.putText(frame, f"FPS: {fps:.1f}", (8, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if not _pipeline_ok:
                    cv2.putText(frame, "PIPELINE HATASI", (8, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow(WINDOW_TITLE, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                _autonomous = not _autonomous
                if _autonomous and _motors:
                    _motors.set_speed(_follower.SPEED)
                    print(f"[INFO] Otonom AÇIK ▶  hız={_follower.SPEED:.2f}")
                else:
                    if _motors:
                        _motors.stop()
                    print("[INFO] Otonom KAPALI ■")
            elif key == ord(' '):
                _autonomous = False
                if _motors:
                    _motors.emergency_stop()
                print("[INFO] ACİL DURUŞ")
    finally:
        _autonomous = False
        if _motors:
            _motors.close()
        cam.stop()
        pipe.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
