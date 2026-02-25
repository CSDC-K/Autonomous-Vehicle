import cv2
from ultralytics import YOLO
import time
from picamera2 import Picamera2

# 1. Modeli yükle
model = YOLO("best.pt")

# 2. Picamera2 Yapılandırması
picam2 = Picamera2()
# OpenCV doğrudan BGR formatı kullanır, kamerayı buna göre yapılandırıyoruz
config = picam2.create_preview_configuration(main={"format": "BGR888", "size": (640, 640)})
picam2.configure(config)
picam2.start()

print("Otonom Araç Canlı Test Başladı. Çıkmak için 'q' tuşuna basın.")

try:
    while True:
        # Görüntüyü doğrudan NumPy dizisi olarak yakala (BGR formatında gelecek)
        frame = picam2.capture_array()

        t_start = time.time()

        # 3. Predict 
        results = model.predict(source=frame, device='cpu', imgsz=480, conf=0.4, verbose=False)
        
        t_end = time.time()
        fps = 1 / (t_end - t_start)

        # 4. Sonuçları Çiz
        annotated_frame = results[0].plot()

        # FPS bilgisini ekle
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Görüntüyü göster
        cv2.imshow("Otonom Arac - Pi 5 Test", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Kamera sensörünü ve pencereleri düzgünce kapat
    picam2.stop()
    cv2.destroyAllWindows()