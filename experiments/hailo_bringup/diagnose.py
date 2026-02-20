"""
Pi'de çalıştır: python3 diagnose.py
hailonet tensor çıkışını kesin olarak doğrular.
"""
import sys, os, threading, time
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
import hailo
import numpy as np

Gst.init(None)

HEF = "/home/kurttepe-2/Desktop/best.hef"
DEBUG_SO = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libdebug.so"

print("="*60)
print("  HAILO TENSOR DIAGNOSTIC")
print("="*60)

# ── hailo modülündeki tensor/meta fonksiyonları ──
tensor_funcs = [x for x in dir(hailo) if any(k in x.lower() for k in ["tensor","meta","buffer"])]
print(f"\n[1] hailo modülü tensor/buffer fonksiyonları:\n    {tensor_funcs}")

# ── HailoRT import denemesi ──
print("\n[2] HailoRT Python API:")
for mod in ["hailo_platform", "hailort", "hailo_platform.pyhailort"]:
    try:
        __import__(mod)
        print(f"    {mod}: OK")
    except Exception as e:
        print(f"    {mod}: FAIL ({e})")

# ── Test 1: dump_tensors_to_npy ──
print("\n[3] dump_tensors_to_npy testi:")
os.chdir("/tmp")
# Eski npy dosyalarını temizle
for f in os.listdir("/tmp"):
    if f.endswith(".npy"):
        os.remove(f"/tmp/{f}")

pipe_dump = Gst.parse_launch(
    f"videotestsrc num-buffers=2 ! video/x-raw,format=RGB,width=416,height=416 ! "
    f"hailonet hef-path={HEF} batch-size=1 force-writable=true ! "
    f"hailofilter so-path={DEBUG_SO} function-name=dump_tensors_to_npy qos=false ! "
    f"fakesink"
)
loop1 = GLib.MainLoop()
bus1 = pipe_dump.get_bus()
bus1.add_signal_watch()
bus1.connect("message", lambda b,m,l: l.quit() if m.type in (Gst.MessageType.EOS, Gst.MessageType.ERROR) else True, loop1)
pipe_dump.set_state(Gst.State.PLAYING)
threading.Timer(10, loop1.quit).start()
loop1.run()
pipe_dump.set_state(Gst.State.NULL)

npy_files = [f for f in os.listdir("/tmp") if f.endswith(".npy")]
if npy_files:
    print(f"    TENSORS VAR! {len(npy_files)} adet .npy dosyası:")
    for f in sorted(npy_files):
        arr = np.load(f"/tmp/{f}")
        print(f"    {f}: shape={arr.shape} dtype={arr.dtype} min={arr.min():.4f} max={arr.max():.4f}")
else:
    print("    TENSOR YOK — hailonet çıktı üretmiyor!")

# ── Test 2: hailofilter identity sonrası ROI ──
print("\n[4] hailofilter(identity) sonrası ROI testi:")
results = []

def on_sample(sink):
    sample = sink.emit("pull-sample")
    if not sample or len(results) >= 3:
        return Gst.FlowReturn.OK
    buf = sample.get_buffer()
    try:
        roi = hailo.get_roi_from_buffer(buf)
        t = roi.get_tensors()
        d = roi.get_objects_typed(hailo.HAILO_DETECTION)
        # Tüm objeleri listele
        all_obj = roi.get_objects()
        entry = {"tensors": len(t), "dets": len(d), "objects": len(all_obj)}
        for tensor in t:
            entry.setdefault("tensor_names", []).append(tensor.name())
        results.append(entry)
    except Exception as e:
        results.append({"error": str(e)})
    return Gst.FlowReturn.OK

pipe2 = Gst.parse_launch(
    f"videotestsrc num-buffers=5 ! video/x-raw,format=RGB,width=416,height=416 ! "
    f"hailonet hef-path={HEF} batch-size=1 force-writable=true ! "
    f"hailofilter so-path={DEBUG_SO} function-name=identity qos=false ! "
    f"appsink name=s emit-signals=true sync=false"
)
sink2 = pipe2.get_by_name("s")
sink2.connect("new-sample", on_sample)
loop2 = GLib.MainLoop()
bus2 = pipe2.get_bus()
bus2.add_signal_watch()
bus2.connect("message", lambda b,m,l: l.quit() if m.type in (Gst.MessageType.EOS, Gst.MessageType.ERROR) else True, loop2)
pipe2.set_state(Gst.State.PLAYING)
threading.Timer(10, loop2.quit).start()
loop2.run()
pipe2.set_state(Gst.State.NULL)

for i, r in enumerate(results):
    print(f"    Frame {i}: {r}")

# ── Test 3: hailonet src pad'den doğrudan okuma (hailofilter YOK) ──
print("\n[5] hailonet src pad probe (hailofilter OLMADAN):")
probe_results = []

def probe_cb(pad, info):
    if len(probe_results) >= 3:
        return Gst.PadProbeReturn.OK
    buf = info.get_buffer()
    entry = {"buf_size": buf.get_size(), "n_meta": 0}
    # GstMeta iterate
    try:
        roi = hailo.get_roi_from_buffer(buf)
        t = roi.get_tensors()
        entry["roi_tensors"] = len(t)
        for tensor in t:
            entry.setdefault("tensor_names", []).append(tensor.name())
    except Exception as e:
        entry["roi_error"] = str(e)
    probe_results.append(entry)
    return Gst.PadProbeReturn.OK

pipe3 = Gst.parse_launch(
    f"videotestsrc num-buffers=5 ! video/x-raw,format=RGB,width=416,height=416 ! "
    f"hailonet name=net hef-path={HEF} batch-size=1 force-writable=true ! "
    f"fakesink"
)
net = pipe3.get_by_name("net")
net.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe_cb)
loop3 = GLib.MainLoop()
bus3 = pipe3.get_bus()
bus3.add_signal_watch()
bus3.connect("message", lambda b,m,l: l.quit() if m.type in (Gst.MessageType.EOS, Gst.MessageType.ERROR) else True, loop3)
pipe3.set_state(Gst.State.PLAYING)
threading.Timer(10, loop3.quit).start()
loop3.run()
pipe3.set_state(Gst.State.NULL)

for i, r in enumerate(probe_results):
    print(f"    Frame {i}: {r}")

# ── Özet ──
print("\n" + "="*60)
if npy_files:
    print("SONUÇ: hailonet TENSOR ÜRETİYOR — sorun metadata erişiminde")
    print("Çözüm: .npy verilerinden Python decoder yazılabilir")
    print("   veya HailoRT Python API ile doğrudan inference yapılabilir")
else:
    print("SONUÇ: hailonet TENSOR ÜRETMİYOR")
    print("Olası nedenler:")
    print("  1. HEF input format uyumsuzluğu")
    print("  2. Model derleme hatası")
    print("  3. HEF dosyası hasarlı")
print("="*60)
