import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import sys

# GStreamer Başlat
Gst.init(None)

# --- AYARLAR ---
HEF_FILE = "yolov8n-pose.hef"

# Post-Process dosyasını otomatik bulma
def find_post_process_so():
    search_paths = [
        "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so",
        "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_post.so",
        "/usr/lib/hailo/tappas/post_processes/libyolo_hailortpp_post.so"
    ]
    for path in search_paths:
        if os.path.exists(path):
            return path
    return None

so_path = find_post_process_so()
print(f"DEBUG: Bulunan .so dosyası: {so_path}")
print(f"DEBUG: Kullanılan HEF dosyası: {HEF_FILE} ({'Mevcut' if os.path.exists(HEF_FILE) else 'YOK!'})")

if not so_path or not os.path.exists(HEF_FILE):
    print("HATA: Kritik dosyalardan biri eksik! Lütfen yolları kontrol edin.")
    sys.exit(1)

# Pipeline Tanımı
pipeline_str = (
    f"libcamerasrc ! video/x-raw, format=RGB, width=640, height=480 ! "
    f"videoscale ! video/x-raw, width=640, height=640 ! "
    f"hailonet hef-path={HEF_FILE} ! "
    f"hailofilter so-path={so_path} function-name=filter_letterbox_yolov8_pose config-path=null qos=false ! "
    f"videoconvert ! video/x-raw, format=RGBA ! "
    f"appsink name=sink emit-signals=true max-buffers=1 drop=true"
)

try:
    print("DEBUG: Pipeline oluşturuluyor...")
    pipeline = Gst.parse_launch(pipeline_str)
    
    # Hata mesajlarını dinlemek için Bus kullanımı
    bus = pipeline.get_bus()
    
    print("DEBUG: Pipeline durumu PLAYING yapılıyor...")
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("HATA: Pipeline başlatılamadı (State Change Failure).")
        
    # Mesaj döngüsü
    print("DEBUG: Mesaj döngüsü başladı. Hata bekleniyor...")
    while True:
        msg = bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS)
        if msg:
            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                print(f"\n--- KRİTİK GSTREAMER HATASI ---\nError: {err}\nDebug Info: {debug}\n-------------------------------")
                break
            elif msg.type == Gst.MessageType.EOS:
                print("Yayın bitti (EOS).")
                break
except Exception as e:
    print(f"PYTHON HATASI: {e}")
finally:
    if 'pipeline' in locals():
        pipeline.set_state(Gst.State.NULL)
