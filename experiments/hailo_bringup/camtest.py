import time
import cv2
import numpy as np
import sys

# Picamera2 import kontrolü
try:
    from picamera2 import Picamera2
except ImportError:
    print("HATA: Picamera2 kütüphanesi yüklü değil.")
    print("Lütfen şu komutu çalıştırın: sudo apt install python3-libcamera python3-opencv")
    sys.exit(1)

def main():
    print("Kamera başlatılıyor... (Lütfen bekleyin)")
    
    try:
        # 1. Picamera2 Nesnesini Oluştur
        picam2 = Picamera2()

        # 2. Konfigürasyon (Pi 5 ISP donanımını kullanır)
        # Çözünürlük: 640x480 (Yapay zeka için ideal hız)
        # Format: BGR888 (OpenCV'nin sevdiği format, renk dönüşümü gerektirmez)
        config = picam2.create_video_configuration(
            main={"size": (640, 480), "format": "BGR888"}
        )
        picam2.configure(config)
        picam2.start()

        print("Kamera aktif! Çıkmak için 'q' tuşuna basın.")
        print("Pencere açılmazsa HDMI ekranınızın bağlı olduğundan emin olun.")

        # FPS Hesaplama için değişkenler
        prev_frame_time = 0
        new_frame_time = 0

        while True:
            # 3. Görüntüyü Yakala (Zero-copy / En hızlı yöntem)
            frame = picam2.capture_array()

            # FPS Hesabı
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps_text = f"FPS: {int(fps)}"

            # Ekrana FPS Yazdır
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)

            # 4. Görüntüyü Göster
            cv2.imshow("Pi5 Camera Test - Picamera2", frame)

            # 'q' tuşu ile çıkış
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Bir hata oluştu: {e}")

    finally:
        # 5. Temizlik (Kaynakları serbest bırak)
        if 'picam2' in locals() and picam2 is not None:
            picam2.stop()
            picam2.close()
        cv2.destroyAllWindows()
        print("Test sonlandırıldı.")

if __name__ == "__main__":
    main()
