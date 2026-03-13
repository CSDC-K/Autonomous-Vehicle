import os
# Raspberry Pi 5 için pin fabrikasını 'lgpio' olarak zorluyoruz (Sessiz çökmeyi kesin engeller)
os.environ["GPIOZERO_PIN_FACTORY"] = "lgpio"

from gpiozero import Motor, PWMOutputDevice
from time import sleep

# Görseldeki (maxresdefault.jpg) bağlantılara göre BCM Pin Tanımlamaları:
# IN1 -> BCM 23 (Fiziksel Pin 16)
# IN2 -> BCM 24 (Fiziksel Pin 18)
motor_yon = Motor(forward=23, backward=24) 

# ENA (Hız Kontrolü) -> BCM 18 (Fiziksel Pin 12)
motor_hiz = PWMOutputDevice(18)            

def dc_motor_test():
    print("DC Motor Testi Başlıyor... Lütfen aracın tekerleklerini havada tutun!")
    sleep(2)
    
    print(">>> İleri Yön - Orta Hız (%50)")
    motor_hiz.value = 1.0  # Hız ayarı (0.0 ile 1.0 arası)
    motor_yon.forward()
    sleep(10)
    
    print(">>> Motor Frenliyor...")
    motor_yon.stop()
    motor_hiz.value = 0.0
    sleep(1.5)
    
    print(">>> Geri Yön - Tam Güç (%100)")
    motor_hiz.value = 1.0  # Tam güç
    motor_yon.backward()
    sleep(3)
    
    print(">>> Geri Yön - Yavaş Sürüş (%30 - Yaya Geçidi/Park Hızı)")
    motor_hiz.value = 0.3  # Düşük hız
    sleep(3)
    
    print("Test Başarıyla Tamamlandı. Güç kesiliyor.")
    motor_yon.stop()
    motor_hiz.value = 0.0

if __name__ == "__main__":
    dc_motor_test()
