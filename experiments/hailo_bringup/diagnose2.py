"""
diagnose2.py — FLOAT32 output + C++ decoder test
Pi'de çalıştır: python3 diagnose2.py
"""
import os, sys, threading, time
import numpy as np
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
import hailo

Gst.init(None)
HEF = "/home/kurttepe-2/Desktop/best.hef"
YOLO_SO = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so"
DEBUG_SO = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libdebug.so"
LABELS = "/home/kurttepe-2/Desktop/HAILO8L_OTONOM_CODES/labels.json"

print("=" * 60)
print("  HAILO DIAGNOSTIC #2")
print("=" * 60)

def run_pipeline(desc, pipe_str, probe_fn, timeout=15):
    print(f"\n[{desc}]")
    try:
        pipe = Gst.parse_launch(pipe_str)
    except Exception as e:
        print(f"  Parse HATA: {e}"); return
    id_el = pipe.get_by_name("probe_point")
    if id_el:
        id_el.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe_fn)
    loop = GLib.MainLoop()
    bus = pipe.get_bus(); bus.add_signal_watch()
    def on_msg(b, m, l):
        if m.type == Gst.MessageType.ERROR:
            err, dbg = m.parse_error()
            print(f"  BUS HATA: {err}")
        if m.type in (Gst.MessageType.EOS, Gst.MessageType.ERROR):
            l.quit()
        return True
    bus.connect("message", on_msg, loop)
    pipe.set_state(Gst.State.PLAYING)
    threading.Timer(timeout, loop.quit).start()
    loop.run()
    pipe.set_state(Gst.State.NULL)

# --- Test A: UINT8 (eski yöntem) + YOLO filter ---
results = {"A": [], "B": [], "C": {}, "D": []}

def make_probe(key):
    def cb(pad, info):
        if len(results[key]) >= 3: return Gst.PadProbeReturn.OK
        buf = info.get_buffer()
        try:
            roi = hailo.get_roi_from_buffer(buf)
            t = roi.get_tensors()
            d = roi.get_objects_typed(hailo.HAILO_DETECTION)
            entry = {"T": len(t), "D": len(d)}
            for det in d[:5]:
                bb = det.get_bbox()
                entry.setdefault("dets", []).append(
                    f"c{det.get_class_id()} {det.get_confidence():.3f}")
            for tensor in t[:5]:
                entry.setdefault("tens", []).append(
                    f"{tensor.name()} {list(tensor.shape())}")
            results[key].append(entry)
        except Exception as e:
            results[key].append({"err": str(e)})
        return Gst.PadProbeReturn.OK
    return cb

# A: UINT8 + yolo filter (eski)
run_pipeline("A: UINT8 + yolo filter",
    f"videotestsrc num-buffers=5 ! video/x-raw,format=RGB,width=416,height=416 ! "
    f"hailonet hef-path={HEF} batch-size=1 force-writable=true ! "
    f"hailofilter so-path={YOLO_SO} function-name=filter qos=false ! "
    f"identity name=probe_point ! fakesink",
    make_probe("A"))
for r in results["A"]: print(f"  {r}")

# B: FLOAT32 + yolo filter
run_pipeline("B: FLOAT32 + yolo filter",
    f"videotestsrc num-buffers=5 ! video/x-raw,format=RGB,width=416,height=416 ! "
    f"hailonet hef-path={HEF} batch-size=1 force-writable=true "
    f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32 ! "
    f"hailofilter so-path={YOLO_SO} function-name=filter qos=false ! "
    f"identity name=probe_point ! fakesink",
    make_probe("B"))
for r in results["B"]: print(f"  {r}")

# C: FLOAT32 + farklı fonksiyonlar
for fn in ["yolov8s", "yolov8m", "filter_letterbox"]:
    results["C"][fn] = []
    def make_c_probe(fn_name):
        def cb(pad, info):
            if len(results["C"][fn_name]) >= 2: return Gst.PadProbeReturn.OK
            buf = info.get_buffer()
            try:
                roi = hailo.get_roi_from_buffer(buf)
                t = roi.get_tensors(); d = roi.get_objects_typed(hailo.HAILO_DETECTION)
                results["C"][fn_name].append({"T": len(t), "D": len(d)})
            except Exception as e:
                results["C"][fn_name].append({"err": str(e)})
            return Gst.PadProbeReturn.OK
        return cb
    run_pipeline(f"C: FLOAT32 + {fn}",
        f"videotestsrc num-buffers=3 ! video/x-raw,format=RGB,width=416,height=416 ! "
        f"hailonet hef-path={HEF} batch-size=1 force-writable=true "
        f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32 ! "
        f"hailofilter so-path={YOLO_SO} function-name={fn} qos=false ! "
        f"identity name=probe_point ! fakesink",
        make_c_probe(fn))
    for r in results["C"][fn]: print(f"  {r}")

# D: FLOAT32 + dump (float .npy analizi)
os.chdir("/dev/shm")
for f in os.listdir("/dev/shm"):
    if f.endswith(".npy"): os.remove(f"/dev/shm/{f}")

run_pipeline("D: FLOAT32 + dump_tensors_to_npy",
    f"videotestsrc num-buffers=3 ! video/x-raw,format=RGB,width=416,height=416 ! "
    f"hailonet hef-path={HEF} batch-size=1 force-writable=true "
    f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32 ! "
    f"hailofilter so-path={DEBUG_SO} function-name=dump_tensors_to_npy qos=false ! "
    f"fakesink", None)

npy_files = sorted([f for f in os.listdir("/dev/shm") if f.endswith(".npy")])
for f in npy_files:
    arr = np.load(f"/dev/shm/{f}")
    print(f"  {f}: shape={arr.shape} dtype={arr.dtype} min={arr.min():.4f} max={arr.max():.4f}")
    if "concat18" in f:
        raw = arr.reshape(-1, 66)
        cls = raw[:, 64:]
        if arr.dtype == np.float32:
            sig = 1.0 / (1.0 + np.exp(-np.clip(cls, -50, 50)))
            print(f"    sigmoid stats: min={sig.min():.4f} max={sig.max():.4f} mean={sig.mean():.4f}")
            print(f"    anchors > 0.25: {(sig.max(axis=1) > 0.25).sum()}")
            print(f"    anchors > 0.10: {(sig.max(axis=1) > 0.10).sum()}")
        else:
            print(f"    (uint8: quant params gerekli)")

print("\n" + "=" * 60)
print("ÖZET:")
has_dets = any(r.get("D", 0) > 0 for r in results["A"] + results["B"])
for fn, res in results["C"].items():
    if any(r.get("D", 0) > 0 for r in res): has_dets = True
if has_dets:
    print("C++ .so DETECTION buldu! → Direkt kullanılabilir")
else:
    print("C++ .so 0 detection → Python decode gerekli")
    if npy_files:
        for f in npy_files:
            if "concat18" in f:
                arr = np.load(f"/dev/shm/{f}")
                if arr.dtype == np.float32:
                    print("FLOAT32 tensorler VAR → Python DFL decoder çalışır!")
                else:
                    print("UINT8 tensorlar → quant param tahmin gerekli")
print("=" * 60)
