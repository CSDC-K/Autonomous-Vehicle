import sys, os, threading, time
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


def draw_detections(frame, dets):
    if not dets:
        return frame
    h, w = frame.shape[:2]
    overlay = frame.copy()
    for (box, conf, cid, mask_104) in dets:
        col = CLASS_COLORS[cid % len(CLASS_COLORS)]
        ix1, iy1 = int(box[0] * w), int(box[1] * h)
        ix2, iy2 = int(box[2] * w), int(box[3] * h)
        if ix1 >= ix2 or iy1 >= iy2:
            continue

        mask_full = cv2.resize(mask_104, (w, h), interpolation=cv2.INTER_LINEAR)
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_mask[iy1:iy2, ix1:ix2] = (mask_full[iy1:iy2, ix1:ix2] > 0.5).astype(np.uint8) * 255

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
_dbg_count = 0


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

    frame = draw_detections(frame, dets)

    with _lock:
        _latest_frame = cv2.resize(frame, (DISPLAY_W or 640, DISPLAY_H or 480))
    return Gst.PadProbeReturn.OK


def _poll_bus(bus):
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
            return False
        elif t == Gst.MessageType.WARNING:
            w, _ = msg.parse_warning()
            print(f"[GST UYARI] {w.message}")
    return True


def main():
    # libdebug.so CWD'ye yazar → CWD'yi /dev/shm'e çevir
    os.chdir(_SHM)
    print(f"[INFO] CWD → {os.getcwd()}")

    # Eski npy artıklarını temizle
    for f in [_BOX_PATH, _COEF_PATH, _PROTO_PATH]:
        if os.path.exists(f):
            os.remove(f)

    print("[INFO] Kamera başlatılıyor...")
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"},
        controls={"FrameRate": 30}
    ))
    cam.start()
    time.sleep(0.3)
    print("[INFO] Kamera hazır.")

    print("[INFO] GStreamer pipeline kuruluyor...")
    pipe = Gst.parse_launch(
        f"appsrc name=s is-live=true format=time "
        f"caps=video/x-raw,format=RGB,width={MODEL_W},height={MODEL_H},framerate=30/1 ! "
        f"queue leaky=downstream max-size-buffers=2 ! "
        f"videoconvert ! "
        f"hailonet hef-path={HEF_PATH} force-writable=true ! "
        f"hailofilter so-path={DEBUG_SO} function-name={POST_FUNC} ! "
        f"identity name=id ! fakesink sync=false"
    )

    bus = pipe.get_bus()
    pipe.get_by_name("id").get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER, _pad_probe_callback
    )

    ret = pipe.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("[HATA] Pipeline başlatılamadı!")
        return
    print("[INFO] Pipeline çalışıyor.")

    def cam_loop():
        src = pipe.get_by_name("s")
        ts = 0
        dur = 10**9 // 30
        while True:
            img = cam.capture_array("main")
            rgb = cv2.resize(img, (MODEL_W, MODEL_H))
            buf = Gst.Buffer.new_wrapped(rgb.tobytes())
            buf.pts = ts
            buf.duration = dur
            ts += dur
            src.emit("push-buffer", buf)

    threading.Thread(target=cam_loop, daemon=True).start()

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    fps_t0 = time.time()
    fps_cnt = 0
    print("[INFO] 'q' ile çık.")
    try:
        while True:
            if not _poll_bus(bus):
                break
            with _lock:
                frame = _latest_frame.copy() if _latest_frame is not None else None
            if frame is not None:
                fps_cnt += 1
                if SHOW_FPS and (fps_cnt % 15 == 0):
                    fps = fps_cnt / (time.time() - fps_t0 + 1e-9)
                    cv2.putText(frame, f"FPS: {fps:.1f}", (8, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow(WINDOW_TITLE, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.stop()
        pipe.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
