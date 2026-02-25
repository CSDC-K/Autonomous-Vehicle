import sys
import threading
import time
import cv2
import numpy as np
from picamera2 import Picamera2
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GLib

Gst.init(None)

# --- AYARLAR ---
FRAME_W, FRAME_H = 640, 640
CAMERA_FPS = 30

# Sistemde varsayılan olarak bulunan test modelini kullanıyoruz (Kendi modelimiz gelene kadar)
TEST_HEF_PATH = "/usr/share/hailo-models/yolov8s_h8l.hef"

# Global Değişkenler
_pipeline = None
_picam2 = None
_latest_frame = None
_is_running = False
_lock = threading.Lock()

def _get_numpy_from_buffer_rgb(buffer, width, height):
    """GStreamer buffer'ından RGB numpy dizisi çıkarır"""
    success, map_info = buffer.map(Gst.MapFlags.READ)
    if not success:
        return None
    try:
        expected_size = width * height * 3
        if map_info.size >= expected_size:
            arr = np.frombuffer(map_info.data[:expected_size], dtype=np.uint8)
            return arr.reshape((height, width, 3)).copy()
    finally:
        buffer.unmap(map_info)
    return None

def _pad_probe_callback(pad, info):
    """Hailo NPU'dan çıkan her kare bu fonksiyona düşer"""
    global _latest_frame
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # RGB görüntüyü al ve OpenCV için BGR'ye çevir
    frame_rgb = _get_numpy_from_buffer_rgb(buffer, FRAME_W, FRAME_H)
    if frame_rgb is not None:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Gelecekte PID Hatası (Error) hesaplamalarını tam buraya yazacağız!
        # Şimdilik sadece FPS ve test yazısı basıyoruz.
        cv2.putText(frame_bgr, "Hailo NPU Aktif - Model Bekleniyor", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        with _lock:
            _latest_frame = frame_bgr

    return Gst.PadProbeReturn.OK

def _camera_feed_thread():
    """Picamera2'den görüntü alıp GStreamer'a basan thread"""
    global _is_running
    appsrc = _pipeline.get_by_name("camera_source")
    frame_duration = Gst.util_uint64_scale_int(1, Gst.SECOND, CAMERA_FPS)
    frame_count = 0
    
    while _is_running:
        raw = _picam2.capture_array("main")
        if raw is None: continue
        
        # Modeller genelde 640x640 giriş bekler
        resized = cv2.resize(raw, (FRAME_W, FRAME_H))
        buf = Gst.Buffer.new_wrapped(resized.tobytes())
        buf.pts = frame_count * frame_duration
        buf.duration = frame_duration
        
        appsrc.emit("push-buffer", buf)
        frame_count += 1
        time.sleep(1 / CAMERA_FPS)

def main():
    global _pipeline, _picam2, _is_running

    print("Kamera başlatılıyor...")
    _picam2 = Picamera2()
    config = _picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"},
        controls={"FrameRate": CAMERA_FPS}
    )
    _picam2.configure(config)
    _picam2.start()
    
    # GStreamer Boru Hattı (Pipeline)
    # Hailo ağını test etmek için filtre (so) olmadan doğrudan fakesink'e bağlıyoruz
    pipeline_str = (
        f"appsrc name=camera_source is-live=true format=time ! "
        f"video/x-raw,format=RGB,width={FRAME_W},height={FRAME_H},framerate={CAMERA_FPS}/1 ! "
        f"videoconvert ! "
        f"hailonet hef-path={TEST_HEF_PATH} batch-size=1 force-writable=true ! "
        f"identity name=identity_callback ! "
        f"fakesink sync=false"
    )
    
    _pipeline = Gst.parse_launch(pipeline_str)
    
    # Pad Probe (Görüntü yakalayıcı) bağlantısı
    identity = _pipeline.get_by_name("identity_callback")
    identity.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, _pad_probe_callback)
    
    _pipeline.set_state(Gst.State.PLAYING)
    _is_running = True
    
    threading.Thread(target=_camera_feed_thread, daemon=True).start()
    
    print("Sistem çalışıyor. Çıkmak için ekranda 'q' tuşuna basın.")
    
    try:
        while True:
            with _lock:
                frame = _latest_frame.copy() if _latest_frame is not None else None
                
            if frame is not None:
                cv2.imshow("Otonom Arac - Hailo Kamera Test", frame)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print("Sistem kapatılıyor...")
        _is_running = False
        _pipeline.set_state(Gst.State.NULL)
        _picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()