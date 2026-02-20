#!/usr/bin/env python3
"""
qparams.py — Quantization parametrelerini bul
Pi'de çalıştır: python3 qparams.py
"""
import subprocess, os, numpy as np

HEF  = "/home/kurttepe-2/Desktop/best.hef"
NPY  = "/dev/shm/best_concat18.npy"

# ─── 1. hailortcli çıktısı ───
print("=" * 60)
print("1) hailortcli parse-hef")
print("=" * 60)
r = subprocess.run(["hailortcli", "parse-hef", HEF],
                    capture_output=True, text=True)
print(r.stdout[:3000])

# ─── 2. Python HEF API (quant params) ───
print("=" * 60)
print("2) Python HEF API")
print("=" * 60)
for imp in [
    "from hailo_platform import HEF",
    "from hailo_platform.pyhailort import HEF",
    "from hailort import HEF",
    "from hailort._pyhailort import HEF",
]:
    try:
        exec(imp)
        hef = eval(f'HEF("{HEF}")')
        for info in hef.get_output_vstream_infos():
            qi = info.quant_info
            print(f"  {info.name}: zp={qi.qp_zp} scale={qi.qp_scale}")
        print("  BAŞARILI!")
        break
    except Exception as e:
        print(f"  {imp} → {e}")

# ─── 3. NPY analizi ───
print("\n" + "=" * 60)
print("3) NPY data analizi + zero-point tarama")
print("=" * 60)
if not os.path.exists(NPY):
    print(f"  {NPY} yok! Önce hailo_app.py çalıştır."); exit()

arr = np.load(NPY).reshape(-1, 66).astype(np.float32)
cls = arr[:, 64:]           # (3549, 2) class scores
cls_max = cls.max(axis=1)   # (3549,) max score per anchor
reg = arr[:, :64]           # (3549, 64) DFL regs

print(f"  Anchors: {len(cls_max)}")
print(f"  Class scores: min={cls.min():.0f} max={cls.max():.0f} "
      f"mean={cls.mean():.1f} median={np.median(cls):.1f}")
print(f"  DFL regs:     min={reg.min():.0f} max={reg.max():.0f} "
      f"mean={reg.mean():.1f} median={np.median(reg):.1f}")

# Histogram
print(f"\n  CLS histogram:")
for lo in range(0, 256, 16):
    hi = min(lo + 16, 256)
    n = int(((cls_max >= lo) & (cls_max < hi)).sum())
    bar = "#" * min(50, n * 50 // max(1, len(cls_max)))
    print(f"    [{lo:3d}-{hi:3d}) {n:5d} {bar}")

# Zero-point tarama (sigmoid threshold 0.5 → float > 0)
print(f"\n  Zero-point tarama (anchors with max_cls > zp):")
for zp in range(0, 256):
    n = int((cls_max > zp).sum())
    if 0 < n <= 200:
        marker = " ← !!!" if 1 <= n <= 30 else ""
        print(f"    zp={zp:3d} → {n:4d} det{marker}")
    elif n == 0 and int((cls_max > (zp - 1)).sum()) > 0:
        print(f"    zp={zp:3d} → 0 det (tüm sınıflar burada biter)")
        break

# Scale tarama (en iyi zp civarında)
print(f"\n  Scale tarama (CONF_THRESHOLD=0.25):")
# Find best zp first (closest to 5 dets at sig>0.5)
diffs = [(abs(int((cls_max > zp).sum()) - 5), zp) for zp in range(256)]
diffs.sort()
best_zp = diffs[0][1]
print(f"  En iyi zp tahmini: {best_zp}")

for sc in [0.003, 0.005, 0.008, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
    deq = (cls - best_zp) * sc
    sig = 1.0 / (1.0 + np.exp(-np.clip(deq, -50, 50)))
    n = int((sig.max(axis=1) > 0.25).sum())
    print(f"    zp={best_zp} scale={sc:.3f} → {n:4d} det (threshold=0.25)")

print("\n" + "=" * 60)
print("BİTTİ — Çıktıyı paylaş!")
print("=" * 60)
