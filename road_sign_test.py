import os
import cv2
import time
import numpy as np
from ultralytics import YOLO

# 1. MODEL YÜKLEME (Tüm modlar için ortak)
print("YOLOv8-Seg Modeli Yükleniyor...")
model = YOLO('road_model.pt')

def run_photo_mode():
    folder = "photo"
    if not os.path.exists(folder):
        print(f"[Hata] '{folder}' klasörü bulunamadı. Lütfen oluşturup içine fotoğraf atın.")
        return

    # Sadece fotoğraf dosyalarını al
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print(f"[Bilgi] '{folder}' klasörü boş.")
        return

    print("\n--- FOTOĞRAF MODU ---")
    print("-> Fotoğraf başına 5 saniye beklenir.")
    print("-> 'SPACE' (Boşluk) tuşuna basarak hemen sonrakine geçebilirsin.")
    print("-> 'q' tuşuna basarak moddan çıkabilirsin.")

    for f in files:
        path = os.path.join(folder, f)
        frame = cv2.imread(path)
        
        # CPU'da çalışırken verbose=False ile terminali temiz tut
        results = model.predict(source=frame, conf=0.4, device='cpu', verbose=False)
        
        # Ultralytics'in kendi çizim metodunu kullan (Kutular ve etiketler dahil)
        annotated_frame = results[0].plot()
        
        cv2.imshow("Otonom Test - Foto Mode", annotated_frame)
        
        # 5000 ms (5 saniye) bekle. Tuşa basılırsa ascii kodunu al
        key = cv2.waitKey(5000) & 0xFF
        if key == 32: # SPACE tuşu
            continue
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Fotoğraf testi tamamlandı.")

def run_video_mode():
    folder = "video"
    if not os.path.exists(folder):
        print(f"[Hata] '{folder}' klasörü bulunamadı. Lütfen oluşturup içine video atın.")
        return

    # Sadece video dosyalarını al
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if not files:
        print(f"[Bilgi] '{folder}' klasörü boş.")
        return

    print("\n--- VİDEO MODU ---")
    print("-> Video sonuna kadar oynatılır.")
    print("-> 'SPACE' (Boşluk) tuşuna basarak sıradaki videoya atlayabilirsin.")
    print("-> 'q' tuşuna basarak moddan çıkabilirsin.")

    FRAME_SKIP = 2 # CPU performansını artırmak için

    for f in files:
        path = os.path.join(folder, f)
        cap = cv2.VideoCapture(path)
        
        frame_count = 0
        last_results = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # Video bitti
            
            frame_count += 1
            
            # Frame atlama algoritması (Performans için)
            if frame_count % FRAME_SKIP == 0 or last_results is None:
                last_results = model.predict(source=frame, conf=0.4, device='cpu', verbose=False)
            
            # Kutuları çiz
            annotated_frame = last_results[0].plot()
            
            cv2.imshow("Otonom Test - Video Mode", annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32: # SPACE tuşu (Sıradaki videoya atla)
                break
            elif key == ord('q'): # q tuşu (Komple çık)
                cap.release()
                cv2.destroyAllWindows()
                return
                
        cap.release()
        
    cv2.destroyAllWindows()
    print("Video testi tamamlandı.")

def run_live_mode():
    # Sadece bu mod seçilirse libcamera yüklenir (PC'de test ederken çökmemesi için)
    try:
        from picamera2 import Picamera2
        from libcamera import controls, Transform
    except ImportError:
        print("[HATA] picamera2 kütüphanesi bulunamadı! Bu mod sadece Raspberry Pi üzerinde çalışır.")
        return

    print("\n--- CANLI (LIVE) MOD BAŞLIYOR ---")
    print("AI Camera Başlatılıyor (Kamera 1)...")
    picam2 = Picamera2(camera_num=0)

    # Sensör modlarını listele — en geniş FOV için en büyük sensör alanını seç
    sensor_modes = picam2.sensor_modes
    for i, mode in enumerate(sensor_modes):
        crop = mode.get("crop_limits", mode.get("ScalerCropMaximum", "?"))
        print(f"  Mod {i}: {mode['size']} crop={crop}")

    # En büyük sensör modu = en az crop = en geniş FOV
    best_mode = max(sensor_modes, key=lambda m: m["size"][0] * m["size"][1])
    print(f"Seçilen sensör modu: {best_mode['size']}")

    # Geniş açı: 16:9 oranı ile sensör alanını daha verimli kullan (4:3 yerine)
    config = picam2.create_video_configuration(
        main={"format": "BGR888", "size": (640, 360)},  # 16:9 → daha geniş yatay FOV
        raw={"size": best_mode["size"]},
        buffer_count=2,
        queue=False,
    )
    picam2.configure(config)

    # ScalerCrop: sensörün tüm alanını kullanmaya zorla (crop'u kaldır)
    try:
        sensor_size = best_mode["size"]
        picam2.set_controls({
            "ScalerCrop": (0, 0, sensor_size[0], sensor_size[1]),
        })
        print(f"ScalerCrop: tam sensör alanı ({sensor_size[0]}x{sensor_size[1]})")
    except Exception as e:
        print(f"ScalerCrop ayarlanamadı: {e}")

    # Continuous autofocus (destekliyorsa)
    try:
        picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
    except Exception:
        pass

    picam2.start()

    time.sleep(0.5)
    try:
        metadata = picam2.capture_metadata()
        actual_crop = metadata.get("ScalerCrop", "bilinmiyor")
        print(f"Gerçek ScalerCrop: {actual_crop}")
    except Exception:
        pass

    # Mesafe Ölçüm Parametreleri
    GERCEK_BOYUT_CM = 13.0  # Levhanın gerçek yüksekliği (cm)
    KALIBRASYON_DOSYASI = "kamera_kalibrasyon.txt"
    KALIBRASYON_MESAFE_CM = 50.0

    odak_uzunlugu = None
    if os.path.exists(KALIBRASYON_DOSYASI):
        with open(KALIBRASYON_DOSYASI, "r") as f:
            odak_uzunlugu = float(f.read().strip())
        print(f"Kalibrasyon yüklendi: odak_uzunlugu = {odak_uzunlugu:.1f}")
    else:
        print("=" * 50)
        print("KALIBRASYON GEREKLİ!")
        print("Levhayı kameradan TAM 50 cm uzağa koy.")
        print("Levha algılandığında 'c' tuşuna bas.")
        print("=" * 50)

    FRAME_SKIP = 2
    frame_count = 0
    last_results = None
    kalibrasyon_tamamlandi = odak_uzunlugu is not None

    if kalibrasyon_tamamlandi:
        print("AI Camera Aktif! Tabelalar taranıyor... Çıkmak için 'q' basın.")
    else:
        print("Kalibrasyon bekleniyor... Levhayı 50cm'ye koy ve 'c' bas.")

    try:
        while True:
            frame = picam2.capture_array()
            # libcamera BGR888 dese bile RGB veriyor, kanalları takas et
            frame = frame[:, :, ::-1].copy()

            frame_count += 1

            if frame_count % FRAME_SKIP == 0 or last_results is None:
                last_results = model.predict(
                    source=frame,
                    conf=0.4,
                    device='cpu',
                    verbose=False,
                    imgsz=640,
                    max_det=10,
                )

            if last_results:
                for result in last_results:
                    if result.boxes:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = box.conf[0]
                            label = model.names[int(box.cls[0])]
                            h_pixel = y2 - y1

                            if kalibrasyon_tamamlandi and h_pixel > 0:
                                dist = (GERCEK_BOYUT_CM * odak_uzunlugu) / h_pixel
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                text = f"{label} %{conf:.2f} | {dist:.1f}cm"
                                cv2.putText(frame, text, (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            else:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                text = f"[KAL] {label} h={h_pixel}px"
                                cv2.putText(frame, text, (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                cv2.putText(frame, "Levhayi 50cm'ye koy, 'c' bas",
                                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Otonom Test - Live Mode", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and not kalibrasyon_tamamlandi:
                if last_results:
                    max_h = 0
                    for result in last_results:
                        if result.boxes:
                            for box in result.boxes:
                                _, y1k, _, y2k = map(int, box.xyxy[0])
                                max_h = max(max_h, y2k - y1k)
                    if max_h > 0:
                        odak_uzunlugu = (max_h * KALIBRASYON_MESAFE_CM) / GERCEK_BOYUT_CM
                        kalibrasyon_tamamlandi = True
                        with open(KALIBRASYON_DOSYASI, "w") as f:
                            f.write(str(odak_uzunlugu))
                        print(f"Kalibrasyon tamamlandı! odak_uzunlugu = {odak_uzunlugu:.1f}")
                    else:
                        print("Levha algılanamadı! Levhayı kameranın önüne getir.")

    finally:
        picam2.stop()
        cv2.destroyAllWindows()

# --- ANA MENÜ (Programın Başlangıç Noktası) ---
while True:
    print("\n" + "="*40)
    print("   OTONOM ARAÇ YOLOv8 TEST SİSTEMİ")
    print("="*40)
    print(" Lütfen test modunu seçin:")
    print(" 1 - Fotoğraf Testi (foto)")
    print(" 2 - Video Testi (video)")
    print(" 3 - Canlı Kamera (live)")
    print(" q - Çıkış")
    
    secim = input("Seçiminiz (foto/video/live/q): ").strip().lower()

    if secim == 'foto' or secim == '1':
        run_photo_mode()
    elif secim == 'video' or secim == '2':
        run_video_mode()
    elif secim == 'live' or secim == '3':
        run_live_mode()
    elif secim == 'q':
        print("Sistem kapatılıyor. İyi çalışmalar!")
        break
    else:
        print("Geçersiz seçim! Lütfen foto, video veya live yazın.")