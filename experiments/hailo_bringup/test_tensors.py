"""
Pi'de çalıştır: python3 test_tensors.py
hailonet'in tensor üretip üretmediğini kontrol eder.
"""
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
import hailo, threading

Gst.init(None)

HEF = "/home/kurttepe-2/Desktop/best.hef"

PIPE = (
    "videotestsrc num-buffers=5 pattern=2 ! "
    "video/x-raw,format=RGB,width=416,height=416 ! "
    f"hailonet hef-path={HEF} batch-size=1 force-writable=true ! "
    "appsink name=s emit-signals=true sync=false"
)

results = []

def on_sample(sink):
    sample = sink.emit("pull-sample")
    if not sample:
        return Gst.FlowReturn.OK
    buf = sample.get_buffer()
    roi = hailo.get_roi_from_buffer(buf)
    tensors = roi.get_tensors()
    entry = {"count": len(tensors), "shapes": []}
    for t in tensors:
        try:
            entry["shapes"].append(f"{t.name()} {list(t.shape())} scale={t.qp_scale():.6f} zp={t.qp_zp()}")
        except Exception as e:
            entry["shapes"].append(f"ERR: {e}")
    results.append(entry)
    print(f"[FRAME {len(results)}] tensors={entry['count']}")
    for s in entry["shapes"]:
        print(f"   {s}")
    return Gst.FlowReturn.OK

loop = GLib.MainLoop()
pipe = Gst.parse_launch(PIPE)
sink = pipe.get_by_name("s")
sink.connect("new-sample", on_sample)

bus = pipe.get_bus()
bus.add_signal_watch()
def on_bus(bus, msg, loop):
    if msg.type in (Gst.MessageType.ERROR, Gst.MessageType.EOS):
        err = msg.parse_error()[0] if msg.type == Gst.MessageType.ERROR else "EOS"
        print(f"[BUS] {err}")
        loop.quit()
bus.connect("message", on_bus, loop)

pipe.set_state(Gst.State.PLAYING)
print("[INFO] Pipeline başlatıldı, 8 saniye bekleniyor...")
threading.Timer(8, loop.quit).start()
loop.run()
pipe.set_state(Gst.State.NULL)

print(f"\n=== SONUÇ: {len(results)} kare işlendi ===")
if results:
    print(f"İlk karede tensors={results[0]['count']}")
    if results[0]['count'] == 0:
        print("[SORUN] hailo.get_roi_from_buffer() tensor görmüyor!")
        print("  → Python decoder gerekecek (ham buffer byte okuma)")
    else:
        print("[TAMAM] Tensorlar erişilebilir, Python decoder yazılabilir")
else:
    print("[SORUN] Hiç kare gelmiyor!")
