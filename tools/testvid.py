import cv2
import numpy as np

# Video ayarları
width, height = 640, 360  # Aracın kamera çözünürlüğü (16:9)
fps = 30
duration = 5  # saniye
total_frames = fps * duration

# Video kaydediciyi başlat
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('sollama_simulasyonu.mp4', fourcc, fps, (width, height))

# Renkler
COLOR_BG = (30, 30, 30)       # Siyah MDF zemin (Çok hafif gri ki çizgiler belli olsun)
COLOR_LINE = (255, 255, 255)  # Beyaz şeritler
COLOR_ORANGE = (0, 100, 255)  # Turuncu Kutu (OpenCV BGR formatı kullanır)

# Şerit perspektif noktaları (Kamera açısı)
vanish_y = int(height * 0.4)
center_x = width // 2

for frame_idx in range(total_frames):
    # Boş siyah zemin oluştur
    frame = np.full((height, width, 3), COLOR_BG, dtype=np.uint8)
    
    # Zaman ilerleyişi (0.0 ile 1.0 arası)
    progress = frame_idx / total_frames
    
    # --- KAMERA (ARAÇ) HAREKETİ ---
    # Aracın sollama hareketi için x ekseninde kayma miktarı (Kamera sola kayar, şeritler sağa)
    camera_shift_x = 0
    if 0.3 < progress < 0.7:
        # Sollama anı (Yumuşak geçiş için sinüs eğrisi)
        # İlerleme 0.3'ten 0.7'ye giderken araç sol şeride geçer
        t = (progress - 0.3) / 0.4
        camera_shift_x = int(np.sin(t * np.pi / 2) * 200) # 200 piksel sola kay
    elif progress >= 0.7:
        camera_shift_x = 200 # Sol şeritte kalmaya devam et
        
    # --- ŞERİTLERİ ÇİZ ---
    # Çizgilerin ufuktan alt köşelere doğru açılması
    lines = [
        (center_x - 300, height), # Sol dış çizgi
        (center_x - 100, height), # Orta sol çizgi
        (center_x + 100, height), # Orta sağ çizgi (Çift çizgi 1)
        (center_x + 130, height), # Orta sağ çizgi (Çift çizgi 2)
        (center_x + 350, height)  # Sağ dış çizgi
    ]
    
    for bottom_x in lines:
        # Kamera kaydıkça şeritler ters yöne (sağa) kayar
        shifted_bottom_x = bottom_x + camera_shift_x
        shifted_vanish_x = center_x + (camera_shift_x // 3) # Ufuk noktası daha az kayar (Perspektif)
        
        cv2.line(frame, (shifted_vanish_x, vanish_y), (shifted_bottom_x, height), COLOR_LINE, 4)

    # --- TURUNCU KUTUYU ÇİZ ---
    # Kutu yavaş yavaş yaklaşacak (Büyüyecek ve aşağı inecek)
    # Kutu orijinal (sağ) şeritte duruyor
    box_start_progress = 0.0
    box_end_progress = 0.8 # Videonun %80'inde kameranın altından çıkıp gidecek
    
    if progress < box_end_progress:
        # Yaklaşma oranı (Büyüme)
        scale = 1.0 + (progress / box_end_progress) * 10.0 
        
        # Kutunun boyutu (Yaklaştıkça artar)
        box_w = int(25 * scale)
        box_h = int(20 * scale)
        
        # Kutunun Y pozisyonu (Ufuktan ekranın altına doğru iner)
        box_y = vanish_y + int((height - vanish_y) * (progress / box_end_progress))
        
        # Kutunun X pozisyonu (Kendi şeridi olan sağ şeridin ortası)
        # Kamera sola kaydığı için kutu da ekranda sağa doğru kaymış gibi görünür
        base_box_x = center_x + 50 
        box_x = base_box_x + camera_shift_x - (box_w // 2)
        
        # Eğer kutu hala ekranın içindeyse çiz
        if box_y - box_h < height:
            top_left = (box_x, box_y - box_h)
            bottom_right = (box_x + box_w, box_y)
            cv2.rectangle(frame, top_left, bottom_right, COLOR_ORANGE, -1)
            # Kutunun hatları daha net anlaşılsın diye siyah çerçeve
            cv2.rectangle(frame, top_left, bottom_right, (0,0,0), 2)

    # Kareyi videoya yazdır
    out.write(frame)

# Bitir ve kaydet
out.release()
print("Simülasyon videosu başarıyla oluşturuldu: sollama_simulasyonu.mp4")