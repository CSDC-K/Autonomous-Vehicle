from rpi_hardware_pwm import HardwarePWM
import time

# Donanımsal PWM Kanal 0 (Fiziksel Pin 32 / BCM 12)
# Standart servo frekansı 50 Hz'dir
servo = HardwarePWM(pwm_channel=0, hz=50)

def aci_ayarla(duty_cycle):
    servo.change_duty_cycle(duty_cycle)

def test_baslat():
    print("Servo testi başlıyor... Lütfen servoyu yormamak için elinizde serbest tutun.")
    time.sleep(2)
    
    # %7.5 Genellikle servo motorların tam merkez (90 derece düz) noktasıdır
    print(">>> Merkez Pozisyonu (%7.5)")
    servo.start(7.5) 
    time.sleep(2)
    
    print(">>> Hafif Sola Dönüş (%6.0)")
    aci_ayarla(6.0)
    time.sleep(2)
    
    print(">>> Hafif Sağa Dönüş (%9.0)")
    aci_ayarla(9.0)
    time.sleep(2)
    
    print(">>> Tekrar Merkeze Dönüyor (%7.5) ve kapanıyor.")
    aci_ayarla(7.5)
    time.sleep(1)
    
    servo.stop()
    print("Test tamamlandı.")

if __name__ == "__main__":
    test_baslat()