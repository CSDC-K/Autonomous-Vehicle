import os
import warnings

# Pin fabrikasini lgpio olarak zorluyoruz (Pi 5 icin en saglikli yontem)
os.environ["GPIOZERO_PIN_FACTORY"] = "lgpio"

# PWMSoftwareFallback uyarisini ekranda gormemek icin gizliyoruz
warnings.filterwarnings("ignore")

from gpiozero import AngularServo
from time import sleep

# BCM 12 (Fiziksel Pin 32)
# Not: min ve max_pulse_width degerlerini 0.0005 ve 0.0025 yaparak 
# motorun tum hareket kapasitesini (genis acisini) aktif ediyoruz.
servo = AngularServo(12, min_angle=-90, max_angle=90, 
                     min_pulse_width=0.0005, max_pulse_width=0.0025)

servo.angle = 0

if __name__ == "__main__":
    try:
        while True:
            derece = input("Derece: ")
            try:
                servo.angle = int(derece)
            except:
                servo.angle = float(derece)
            print(f"Servo aci: {servo.angle}")
            sleep(2)
            servo.detach()
    except KeyboardInterrupt:
        print("Program kullanici tarafindan durduruldu.")